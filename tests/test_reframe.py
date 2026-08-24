"""The crop planner. Face positions are injected, so none of this needs
mediapipe, opencv, or a real video."""

from __future__ import annotations

import pytest

from pipeline import reframe
from pipeline.reframe import VideoInfo


@pytest.fixture
def landscape():
    return VideoInfo(width=1920, height=1080, fps=25.0, duration=8.0)


@pytest.fixture(autouse=True)
def no_detector(monkeypatch):
    """Default to 'no faces found' unless a test injects some."""
    monkeypatch.setattr(reframe, "_detect_face_centers", lambda *a, **k: [])


def inject(monkeypatch, samples):
    monkeypatch.setattr(reframe, "_detect_face_centers", lambda *a, **k: samples)


class TestCropWidth:
    def test_landscape_gets_a_916_window(self, landscape):
        assert reframe.crop_width_for(landscape) == 606

    def test_portrait_source_uses_its_full_width(self):
        info = VideoInfo(width=1080, height=1920, fps=30, duration=5)
        assert reframe.crop_width_for(info) == 1080

    def test_square_source_uses_its_full_width(self):
        info = VideoInfo(width=1080, height=1080, fps=30, duration=5)
        assert reframe.crop_width_for(info) == 607 // 2 * 2

    def test_width_is_always_even(self):
        for w, h in [(1919, 1081), (721, 481), (100, 57)]:
            info = VideoInfo(width=w, height=h, fps=25, duration=5)
            assert reframe.crop_width_for(info) % 2 == 0


class TestEvenOffset:
    @pytest.mark.parametrize("value", [0, 1, 2, 3, 99.9, 1000.5])
    def test_always_even(self, value):
        assert reframe.even_offset(value) % 2 == 0


class TestPlan:
    def test_no_faces_falls_back_to_centre(self, landscape):
        crop_w, plan = reframe.build_crop_plan("x.mp4", 0, 8, landscape)
        assert plan == [(0.0, reframe.even_offset((1920 - crop_w) / 2))]

    def test_portrait_never_pans(self):
        info = VideoInfo(width=1080, height=1920, fps=30, duration=5)
        _, plan = reframe.build_crop_plan("x.mp4", 0, 5, info)
        assert plan == [(0.0, 0)]

    def test_still_subject_produces_a_single_position(self, landscape, monkeypatch):
        inject(monkeypatch, [(i * 0.25, 0.3) for i in range(32)])
        _, plan = reframe.build_crop_plan("x.mp4", 0, 8, landscape)
        assert len(plan) == 1

    def test_moving_subject_is_followed(self, landscape, monkeypatch):
        samples = (
            [(i * 0.25, 0.18) for i in range(8)]
            + [(2.0 + i * 0.25, 0.18 + (i + 1) / 12 * 0.6) for i in range(12)]
            + [(5.0 + i * 0.25, 0.78) for i in range(8)]
        )
        inject(monkeypatch, samples)
        crop_w, plan = reframe.build_crop_plan("x.mp4", 0, 8, landscape)
        xs = [x for _, x in plan]
        assert max(xs) - min(xs) > crop_w // 2

    def test_plan_timestamps_strictly_increase(self, landscape, monkeypatch):
        samples = [(i * 0.25, 0.1 + i * 0.03) for i in range(32)]
        inject(monkeypatch, samples)
        _, plan = reframe.build_crop_plan("x.mp4", 0, 8, landscape)
        assert all(b[0] > a[0] for a, b in zip(plan, plan[1:]))

    def test_offsets_stay_inside_the_frame(self, landscape, monkeypatch):
        # deliberately out-of-range detections
        inject(monkeypatch, [(i * 0.25, -0.5 if i % 2 else 1.9) for i in range(32)])
        crop_w, plan = reframe.build_crop_plan("x.mp4", 0, 8, landscape)
        max_x = 1920 - crop_w
        assert all(0 <= x <= max_x for _, x in plan)

    def test_dropped_detections_carry_the_last_known_position(self, landscape, monkeypatch):
        inject(monkeypatch, [(0.0, 0.8), (0.25, None), (0.5, None), (0.75, 0.8)])
        _, plan = reframe.build_crop_plan("x.mp4", 0, 1, landscape)
        assert len(plan) == 1     # a gap must not read as a move

    def test_static_plan_is_centred(self, landscape):
        crop_w, plan = reframe.static_plan(landscape)
        assert plan[0][1] == reframe.even_offset((1920 - crop_w) / 2)


class TestSendcmd:
    def test_first_position_is_not_written(self, tmp_path):
        path = reframe.write_sendcmd([(0.0, 100), (1.0, 200), (2.0, 300)], tmp_path / "c")
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2 and "100" not in path.read_text()

    def test_format_is_what_the_filter_expects(self, tmp_path):
        path = reframe.write_sendcmd([(0.0, 0), (1.5, 240)], tmp_path / "c")
        assert path.read_text().strip() == "1.500 crop x 240;"

    def test_single_position_writes_no_commands(self, tmp_path):
        path = reframe.write_sendcmd([(0.0, 100)], tmp_path / "c")
        assert path.read_text().strip() == ""


class TestSmooth:
    def test_outlier_is_suppressed(self):
        values = [0.5] * 10
        values[5] = 0.99                        # one bad detection
        out = reframe._smooth(values)
        assert max(out) < 0.6

    def test_empty(self):
        assert reframe._smooth([]) == []
