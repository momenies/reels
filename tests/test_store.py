"""Job persistence and the state transitions the worker depends on."""

from __future__ import annotations

import threading
import time

from server.store import CANCELED, DONE, ERROR, PROCESSING, QUEUED


class TestLifecycle:
    def test_create_starts_queued(self, store):
        job = store.create("a.mp4", 100, {"clips": 3})
        assert job.status == QUEUED and job.options == {"clips": 3}

    def test_round_trip_preserves_types(self, store):
        job = store.create("a.mp4", 100, {"clips": 3, "faces": True})
        got = store.get(job.id)
        assert got.options["faces"] is True and got.size_bytes == 100

    def test_get_unknown_is_none(self, store):
        assert store.get("nope") is None

    def test_update_writes_through(self, store):
        job = store.create("a.mp4", 1, {})
        store.update(job.id, stage="render", progress=0.42, message="3/5")
        got = store.get(job.id)
        assert got.stage == "render" and got.progress == 0.42

    def test_finish_records_clips_and_a_timestamp(self, store):
        job = store.create("a.mp4", 1, {})
        store.finish(job.id, DONE, clips=[{"file": "x.mp4"}], elapsed=12.5)
        got = store.get(job.id)
        assert got.status == DONE and got.clips[0]["file"] == "x.mp4"
        assert got.finished_at is not None

    def test_delete(self, store):
        job = store.create("a.mp4", 1, {})
        assert store.delete(job.id) and store.get(job.id) is None

    def test_delete_unknown_is_false(self, store):
        assert not store.delete("nope")


class TestClaim:
    def test_claim_moves_to_processing(self, store):
        job = store.create("a.mp4", 1, {})
        claimed = store.claim_next()
        assert claimed.id == job.id and claimed.status == PROCESSING

    def test_a_job_is_only_claimed_once(self, store):
        store.create("a.mp4", 1, {})
        assert store.claim_next() is not None
        assert store.claim_next() is None

    def test_empty_queue(self, store):
        assert store.claim_next() is None

    def test_oldest_first(self, store):
        first = store.create("a.mp4", 1, {})
        time.sleep(0.01)
        store.create("b.mp4", 1, {})
        assert store.claim_next().id == first.id

    def test_concurrent_claims_never_hand_out_the_same_job(self, store):
        for i in range(8):
            store.create(f"{i}.mp4", 1, {})
        claimed, lock = [], threading.Lock()

        def worker():
            while True:
                job = store.claim_next()
                if job is None:
                    return
                with lock:
                    claimed.append(job.id)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(claimed) == len(set(claimed)) == 8


class TestCancelAndRequeue:
    def test_queued_job_can_be_cancelled(self, store):
        job = store.create("a.mp4", 1, {})
        assert store.cancel(job.id)
        assert store.get(job.id).status == CANCELED

    def test_running_job_cannot_be_cancelled(self, store):
        store.create("a.mp4", 1, {})
        job = store.claim_next()
        assert not store.cancel(job.id)
        assert store.get(job.id).status == PROCESSING

    def test_restart_requeues_interrupted_jobs(self, store):
        store.create("a.mp4", 1, {})
        store.claim_next()
        assert store.requeue_stale() == 1
        assert store.get(store.list()[0].id).status == QUEUED

    def test_requeue_leaves_finished_jobs_alone(self, store):
        job = store.create("a.mp4", 1, {})
        store.finish(job.id, DONE)
        assert store.requeue_stale() == 0


class TestQueries:
    def test_list_is_newest_first(self, store):
        store.create("old.mp4", 1, {})
        time.sleep(0.01)
        store.create("new.mp4", 1, {})
        assert [j.filename for j in store.list()] == ["new.mp4", "old.mp4"]

    def test_list_paginates(self, store):
        for i in range(5):
            store.create(f"{i}.mp4", 1, {})
            time.sleep(0.002)
        assert len(store.list(limit=2)) == 2
        assert len(store.list(limit=2, offset=4)) == 1

    def test_counts_by_status(self, store):
        a = store.create("a.mp4", 1, {})
        store.create("b.mp4", 1, {})
        store.finish(a.id, ERROR, error="boom")
        assert store.counts() == {QUEUED: 1, ERROR: 1}

    def test_older_than_only_returns_finished_jobs(self, store):
        old = store.create("old.mp4", 1, {})
        store.finish(old.id, DONE)
        store.update(old.id, finished_at=time.time() - 10_000)
        store.create("fresh.mp4", 1, {})
        stale = list(store.older_than(time.time() - 3600))
        assert [j.id for j in stale] == [old.id]


class TestUploadHandoff:
    """A worker must not see a job while its bytes are still arriving.

    Claiming one mid-upload runs ffprobe on a partial file, which fails with
    "moov atom not found" — indistinguishable from a genuinely corrupt upload.
    """

    def test_an_uploading_job_is_not_claimable(self, store):
        from server.store import UPLOADING

        store.create("a.mp4", 0, {}, status=UPLOADING)
        assert store.claim_next() is None

    def test_mark_queued_publishes_it(self, store):
        from server.store import UPLOADING

        job = store.create("a.mp4", 0, {}, status=UPLOADING)
        assert store.mark_queued(job.id)
        assert store.claim_next().id == job.id

    def test_mark_queued_only_fires_once(self, store):
        from server.store import UPLOADING

        job = store.create("a.mp4", 0, {}, status=UPLOADING)
        assert store.mark_queued(job.id)
        assert not store.mark_queued(job.id)

    def test_mark_queued_cannot_resurrect_a_running_job(self, store):
        store.create("a.mp4", 0, {})
        job = store.claim_next()
        assert not store.mark_queued(job.id)
        assert store.get(job.id).status == PROCESSING

    def test_restart_fails_an_interrupted_upload(self, store):
        from server.store import UPLOADING

        job = store.create("a.mp4", 0, {}, status=UPLOADING)
        store.requeue_stale()
        got = store.get(job.id)
        assert got.status == ERROR and "interrupted" in got.error

    def test_an_uploading_job_can_still_be_cancelled(self, store):
        from server.store import UPLOADING

        job = store.create("a.mp4", 0, {}, status=UPLOADING)
        assert store.cancel(job.id)
        assert store.get(job.id).status == CANCELED
