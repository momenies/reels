"""The HTTP surface: validation, auth, path safety, and the job endpoints.

The worker thread is left running but never gets a real video to chew on —
these tests are about the API contract, not the pipeline.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REELS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REELS_API_KEY", "")
    from pipeline.config import settings

    settings(refresh=True)
    from server.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def keyed_client(tmp_path, monkeypatch):
    monkeypatch.setenv("REELS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REELS_API_KEY", "s3cret")
    from pipeline.config import settings

    settings(refresh=True)
    from server.app import app

    with TestClient(app) as c:
        yield c


def upload(client, name="a.mp4", body=b"\x00" * 2048, **form):
    return client.post(
        "/api/jobs",
        files={"file": (name, body, "video/mp4")},
        data={"clips": "2", "min_score": "10", **form},
    )


class TestMeta:
    def test_health_reports_the_machine_not_a_constant(self, client):
        import shutil

        body = client.get("/api/health").json()
        assert body["version"]
        # `ok` mirrors whether ffmpeg is installed here, which is a property of
        # the machine, not of the API. CI's unit runner has no ffmpeg on
        # purpose — asserting True would only prove the runner was configured
        # the way this test wanted.
        assert body["ok"] == body["ffmpeg"] == (shutil.which("ffmpeg") is not None)

    def test_health_goes_not_ok_when_ffmpeg_is_missing(self, client, monkeypatch):
        from server import app as server_app

        monkeypatch.setattr(server_app.shutil, "which", lambda name: None)
        body = client.get("/api/health").json()
        assert body["ok"] is False and body["ffmpeg"] is False

    def test_styles_are_exposed_with_css_colours(self, client):
        styles = client.get("/api/styles").json()["styles"]
        assert {s["name"] for s in styles} >= {"pop", "beast", "clean"}
        assert all(s["highlight"].startswith("#") for s in styles)

    def test_ass_colour_conversion_swaps_bgr_to_rgb(self):
        from server.app import _css

        assert _css("&H0000E5FF") == "#FFE500"   # ASS is &HAABBGGRR
        assert _css("&H00FFFFFF") == "#FFFFFF"

    def test_index_serves_the_studio(self, client):
        assert "reels" in client.get("/").text.lower()


class TestUploadValidation:
    def test_rejects_a_non_video_extension(self, client):
        r = upload(client, name="notes.txt")
        assert r.status_code == 415

    def test_rejects_an_empty_file(self, client):
        r = upload(client, body=b"")
        assert r.status_code == 400

    def test_rejects_an_unknown_caption_style(self, client):
        assert upload(client, style="rainbow").status_code == 400

    def test_rejects_a_file_over_the_limit(self, client, monkeypatch):
        from pipeline.config import settings

        monkeypatch.setattr(settings(), "max_upload_mb", 0, raising=False)
        assert upload(client, body=b"\x00" * 4096).status_code == 413

    def test_an_oversized_upload_leaves_no_job_behind(self, client, monkeypatch):
        from pipeline.config import settings

        monkeypatch.setattr(settings(), "max_upload_mb", 0, raising=False)
        upload(client, body=b"\x00" * 4096)
        assert client.get("/api/jobs").json()["jobs"] == []

    def test_accepts_a_video_and_returns_201(self, client):
        r = upload(client)
        assert r.status_code == 201
        assert r.json()["status"] == "queued"
        assert r.json()["size_bytes"] == 2048

    def test_out_of_range_options_are_clamped_not_rejected(self, client):
        from server.worker import options_from

        assert options_from({"clips": 999}).clips == 20
        assert options_from({"clips": -5}).clips == 1
        assert options_from({"min_score": "abc"}).min_score == 60


class TestJobEndpoints:
    def test_get_a_job(self, client):
        job_id = upload(client).json()["id"]
        assert client.get(f"/api/jobs/{job_id}").status_code == 200

    def test_unknown_job_is_404(self, client):
        assert client.get("/api/jobs/deadbeef0000").status_code == 404

    def test_malformed_job_id_is_404_not_500(self, client):
        for bad in ["../../etc", "'; DROP TABLE jobs;--", "x" * 200]:
            assert client.get(f"/api/jobs/{bad}").status_code == 404

    def test_list_includes_counts(self, client):
        upload(client)
        body = client.get("/api/jobs").json()
        assert len(body["jobs"]) == 1 and body["counts"]

    def test_delete_removes_the_job_and_its_directory(self, client, tmp_path):
        job_id = upload(client).json()["id"]
        assert client.delete(f"/api/jobs/{job_id}").status_code == 200
        assert client.get(f"/api/jobs/{job_id}").status_code == 404
        assert not (tmp_path / "data" / "jobs" / job_id).exists()

    def test_source_is_served_back(self, client):
        job_id = upload(client).json()["id"]
        assert client.get(f"/api/jobs/{job_id}/source").status_code == 200

    def test_download_with_no_clips_is_404(self, client):
        job_id = upload(client).json()["id"]
        assert client.get(f"/api/jobs/{job_id}/download").status_code == 404


class TestPathSafety:
    @pytest.mark.parametrize(
        "name",
        [
            "../../../etc/passwd",
            "..%2F..%2Fjobs.db",
            "....//....//etc/passwd",
            "/etc/passwd",
        ],
    )
    def test_traversal_never_escapes_the_job_directory(self, client, name):
        job_id = upload(client).json()["id"]
        r = client.get(f"/api/jobs/{job_id}/files/{name}")
        assert r.status_code == 404

    def test_resolver_rejects_an_escape(self, client, tmp_path):
        from fastapi import HTTPException
        from server.app import _safe_output_path

        # `..` and a bare `.` never arrive over HTTP — a client normalises them
        # out of the URL — so the resolver is where they have to be refused.
        for name in ["..", "../../../../etc/passwd", "/etc/passwd", ".", "", "../out"]:
            with pytest.raises(HTTPException):
                _safe_output_path("abc123", name)

    def test_arabic_filename_in_content_disposition_is_ascii(self):
        from server.app import _ascii

        assert _ascii("مقطع رائع").isascii()
        assert _ascii("podcast episode 3") == "podcast-episode-3"


class TestAuth:
    def test_open_by_default(self, client):
        assert client.get("/api/jobs").status_code == 200

    def test_key_is_required_when_configured(self, keyed_client):
        assert keyed_client.get("/api/jobs").status_code == 401

    def test_correct_key_in_a_header_passes(self, keyed_client):
        r = keyed_client.get("/api/jobs", headers={"x-api-key": "s3cret"})
        assert r.status_code == 200

    def test_wrong_key_is_rejected(self, keyed_client):
        r = keyed_client.get("/api/jobs", headers={"x-api-key": "guess"})
        assert r.status_code == 401

    def test_health_stays_open_for_load_balancers(self, keyed_client):
        assert keyed_client.get("/api/health").status_code == 200


class TestUploadRace:
    """The upload must be complete on disk before any worker can see the job."""

    def test_job_is_queued_only_after_the_bytes_land(self, client):
        body = client.post(
            "/api/jobs",
            files={"file": ("a.mp4", b"\x00" * 500_000, "video/mp4")},
            data={"clips": "1"},
        ).json()
        # by the time the response is written the file is whole and published
        assert body["status"] in ("queued", "processing")
        assert body["size_bytes"] == 500_000

    def test_no_part_file_is_left_behind(self, client, tmp_path):
        job_id = upload(client).json()["id"]
        leftovers = list((tmp_path / "data" / "jobs" / job_id).glob("*.part"))
        assert leftovers == []

    def test_a_partial_file_is_never_offered_to_the_pipeline(self, tmp_path):
        from server.worker import _find_source

        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "source.mp4.part").write_bytes(b"half a video")
        assert _find_source(job_dir) is None

        (job_dir / "source.mp4").write_bytes(b"a whole video")
        assert _find_source(job_dir).name == "source.mp4"


class TestUrlIngest:
    """A link is an alternative to an upload, not a second endpoint."""

    def test_a_link_and_a_file_together_is_rejected(self, client):
        r = client.post(
            "/api/jobs",
            files={"file": ("a.mp4", b"\x00" * 1024, "video/mp4")},
            data={"url": "https://example.com/v"},
        )
        assert r.status_code == 400
        assert "not both" in r.json()["detail"]

    def test_neither_is_rejected(self, client):
        assert client.post("/api/jobs", data={"clips": "2"}).status_code == 400

    def test_a_non_link_is_rejected(self, client):
        r = client.post("/api/jobs", data={"url": "just some words"})
        assert r.status_code == 400

    def test_a_link_needs_yt_dlp_and_says_so(self, client, monkeypatch):
        from pipeline import fetch

        monkeypatch.setattr(fetch, "available", lambda: False)
        r = client.post("/api/jobs", data={"url": "https://example.com/v"})
        assert r.status_code == 503 and "yt-dlp" in r.json()["detail"]

    def test_a_link_queues_a_job_when_yt_dlp_is_present(self, client, monkeypatch):
        from pipeline import fetch

        monkeypatch.setattr(fetch, "available", lambda: True)
        r = client.post(
            "/api/jobs", data={"url": "https://example.com/v", "clips": "2"}
        )
        assert r.status_code == 201
        assert r.json()["options"]["url"] == "https://example.com/v"

    def test_health_reports_whether_links_can_be_fetched(self, client):
        assert "url_ingest" in client.get("/api/health").json()


class TestWorkerFetch:
    def test_a_failed_fetch_reports_the_reason_not_a_traceback(self, store, tmp_path, monkeypatch):
        from pipeline.fetch import FetchError
        from server import worker as w

        job = store.create("https://example.com/v", 0, {"url": "https://example.com/v"})
        store.claim_next()
        (tmp_path / "jobs" / job.id).mkdir(parents=True)

        def boom(url, dest, **kw):
            raise FetchError("that video is private", kind="PRIVATE")

        monkeypatch.setattr(w, "download", boom)
        w.Worker(store, tmp_path / "jobs").run_job(job.id)

        got = store.get(job.id)
        assert got.status == "error" and got.error == "that video is private"
