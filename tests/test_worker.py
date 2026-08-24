"""Worker glue: the progress model the UI bar is driven by, and retention."""

from __future__ import annotations

import time

import pytest

from server import worker
from server.store import DONE


class TestOverallProgress:
    def test_monotonic_across_the_whole_run(self):
        seen = []
        for stage in ("probe", "transcribe", "score", "render"):
            for pct in (0.0, 0.25, 0.5, 0.75, 1.0):
                seen.append(worker.overall_progress(stage, pct))
        assert seen == sorted(seen)

    def test_starts_at_zero_and_ends_at_one(self):
        assert worker.overall_progress("probe", 0.0) == 0.0
        assert worker.overall_progress("render", 1.0) == 1.0
        assert worker.overall_progress("done", 1.0) == 1.0

    def test_weights_sum_to_one(self):
        assert sum(w for _, w in worker.STAGE_WEIGHTS) == pytest.approx(1.0)

    def test_out_of_range_input_is_clamped(self):
        assert 0.0 <= worker.overall_progress("render", 5.0) <= 1.0
        assert 0.0 <= worker.overall_progress("render", -1.0) <= 1.0

    def test_unknown_stage_does_not_crash(self):
        assert 0.0 <= worker.overall_progress("nonsense", 0.5) <= 1.0


class TestOptionsFrom:
    def test_clip_count_is_clamped_to_a_sane_range(self):
        assert worker.options_from({"clips": 10_000}).clips == 20
        assert worker.options_from({"clips": 0}).clips == 1

    def test_auto_language_becomes_none_for_whisper(self):
        for value in ("auto", "detect", "", "AUTO"):
            assert worker.options_from({"lang": value}).lang is None

    def test_a_real_language_is_lowercased_and_kept(self):
        assert worker.options_from({"lang": "AR"}).lang == "ar"

    def test_garbage_numbers_fall_back_to_the_default(self):
        assert worker.options_from({"clips": "many"}).clips == 5
        assert worker.options_from({"min_score": None}).min_score == 60

    def test_empty_payload_is_valid(self):
        opts = worker.options_from({})
        assert opts.clips == 5 and opts.faces is True


class TestSweep:
    def test_removes_finished_jobs_past_retention(self, store, tmp_path):
        jobs_dir = tmp_path / "jobs"
        job = store.create("a.mp4", 1, {})
        (jobs_dir / job.id).mkdir(parents=True)
        (jobs_dir / job.id / "source.mp4").write_bytes(b"x")
        store.finish(job.id, DONE)
        store.update(job.id, finished_at=time.time() - 10_000)

        assert worker.sweep(store, jobs_dir, retention_hours=1) == 1
        assert store.get(job.id) is None
        assert not (jobs_dir / job.id).exists()

    def test_leaves_recent_jobs_alone(self, store, tmp_path):
        job = store.create("a.mp4", 1, {})
        store.finish(job.id, DONE)
        assert worker.sweep(store, tmp_path / "jobs", retention_hours=72) == 0
        assert store.get(job.id) is not None

    def test_retention_off_is_a_no_op(self, store, tmp_path):
        job = store.create("a.mp4", 1, {})
        store.finish(job.id, DONE)
        store.update(job.id, finished_at=0)
        assert worker.sweep(store, tmp_path / "jobs", retention_hours=0) == 0
        assert store.get(job.id) is not None

    def test_missing_source_fails_the_job_cleanly(self, store, tmp_path):
        job = store.create("gone.mp4", 1, {})
        store.claim_next()
        w = worker.Worker(store, tmp_path / "jobs")
        w.run_job(job.id)
        got = store.get(job.id)
        assert got.status == "error" and "no longer on disk" in got.error
