import cv2
import numpy as np
import pytest

import fabric_pipeline as fpipe
from ui import fabric_app


class FakeDet:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, img):
        return self.dets


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(fpipe.CFG["paths"], "db", str(tmp_path / "t.sqlite"))
    monkeypatch.setitem(fpipe.FAB["fabrics"]["blue"]["anomaly"], "enabled", False)   # real model off in UI tests
    monkeypatch.chdir(tmp_path) if False else None
    fabric_app.STATE.update({"rec": None, "action": None, "error": None})
    app = fabric_app.app
    app.testing = True
    return app.test_client()


def photo():
    rng = np.random.default_rng(0)
    img = (rng.normal(128, 40, (1080, 1920, 3))).clip(0, 255).astype(np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def post(client, monkeypatch, dets):
    monkeypatch.setattr(fpipe, "get_detector", lambda fabric: FakeDet(dets))
    return client.post("/inspect", data={"source": "upload", "photo": (__import__("io").BytesIO(photo()), "a.png")},
                       content_type="multipart/form-data", follow_redirects=True)


def test_hole_page(client, monkeypatch):
    big = {"type": "Hole", "confidence": 0.88, "box": (900, 500, 1100, 700), "polygon": None}
    r = post(client, monkeypatch, [big])
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "REJECT" in html and "Recommended action" in html and "Hole" in html
    assert "not calibrated" not in html


def test_pass_page(client, monkeypatch):
    html = post(client, monkeypatch, []).get_data(as_text=True)
    assert "PASS" in html and "No defects found" in html


def test_no_file(client):
    html = client.post("/inspect", data={"source": "upload"}, follow_redirects=True).get_data(as_text=True)
    assert "Choose a photo first" in html


def test_label_and_history(client, monkeypatch):
    post(client, monkeypatch, [])
    client.post("/label", data={"kind": "correct"})
    html = client.get("/").get_data(as_text=True)
    assert "correct" in html


def test_anomaly_only_flag_shows_review_and_note(client, monkeypatch):
    monkeypatch.setitem(fpipe.FAB["fabrics"]["blue"]["anomaly"], "enabled", True)
    class Hot:
        def heatmap(self, img):
            hm = np.zeros(img.shape[:2], np.float32); hm[200:600, 300:900] = 9.0
            return hm
    monkeypatch.setitem(fpipe._anomaly, "blue", Hot())
    html = post(client, monkeypatch, []).get_data(as_text=True)
    assert "REVIEW" in html and "Anomaly" in html and "second-opinion" in html
    assert "Anomaly heatmap" in html
    img = client.get("/heatmap")
    assert img.status_code == 200 and img.mimetype == "image/png"


def test_dashboard_shows_summary_and_device_time(client):
    html = client.get("/").get_data(as_text=True)
    assert "Total device use" in html and "Inspected today" in html


def test_quality_shown_and_feedback_detail_exported(client, monkeypatch):
    big = {"type": "Hole", "confidence": 0.88, "box": (900, 500, 1100, 700), "polygon": None}
    html = post(client, monkeypatch, [big]).get_data(as_text=True)
    assert "Cloth quality" in html and "Grade D" in html
    client.post("/label", data={"kind": "wrong", "corrected": "REWORK", "reason": "Wrong severity",
                                "note": "small hole"}, follow_redirects=True)
    csv_text = client.get("/export.csv").get_data(as_text=True)
    assert "REWORK" in csv_text and "Wrong severity" in csv_text and "small hole" in csv_text


def test_device_usage_accumulates(client):
    sid = fpipe.usage_start()
    fpipe.usage_tick(sid, 3600)
    assert fpipe.usage_total_seconds() >= 3600


def test_why_verdict_id_and_ai_badge(client, monkeypatch):
    big = {"type": "Hole", "confidence": 0.88, "box": (900, 500, 1100, 700), "polygon": None}
    html = post(client, monkeypatch, [big]).get_data(as_text=True)
    assert "Why this verdict" in html and "REJECT because" in html
    assert "Inspection #" in html and "AI verdict" in html
    assert "How the quality score and grade are worked out" in html and "Grade A = 90 or more" in html
    assert "Hole" in html and "W &times; H (mm)" in html and "Hole ×1" in html


def test_operator_confirmation_shown(client, monkeypatch):
    post(client, monkeypatch, [])
    html = client.post("/label", data={"kind": "correct"}, follow_redirects=True).get_data(as_text=True)
    assert "Operator confirmed" in html and "AI verdict" not in html
    html = client.post("/label", data={"kind": "wrong", "corrected": "REWORK"}, follow_redirects=True).get_data(as_text=True)
    assert "Operator changed to REWORK" in html


def test_model_and_frame_info_shown(client, monkeypatch):
    html = post(client, monkeypatch, []).get_data(as_text=True)
    assert "best_a2.onnx" in html and "Camera frame" in html and "1920 &times; 1080" in html


# ---------------------------------------------------------------- live camera and pattern page

class FakeCap:
    def __init__(self):
        self.n = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.n += 1
        img = np.full((540, 960, 3), 128, np.uint8)
        cv2.putText(img, str(self.n), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        return True, img

    def release(self):
        self.released = True


def test_live_view_start_capture_stop(client, monkeypatch):
    from ui.live_camera import LiveCamera
    cap = FakeCap()
    live = LiveCamera(opener=lambda i: cap, fps=100)
    monkeypatch.setattr(fabric_app, "LIVE", live)
    assert "Start live view" in client.get("/").get_data(as_text=True)
    html = client.post("/live/start", follow_redirects=True).get_data(as_text=True)
    assert "Capture &amp; select area" in html and "/live.mjpg" in html
    monkeypatch.setattr(fpipe, "get_detector", lambda fabric: FakeDet([]))
    html = client.post("/inspect", data={"source": "live"}, follow_redirects=True).get_data(as_text=True)
    assert "PASS" in html or "REVIEW" in html or "INVALID" in html
    first = next(live.mjpeg())
    assert first.startswith(b"--frame") and b"image/jpeg" in first
    client.post("/live/stop")
    assert not live.running and cap.released


def test_live_capture_without_live_view_shows_message(client, monkeypatch):
    from ui.live_camera import LiveCamera
    monkeypatch.setattr(fabric_app, "LIVE", LiveCamera(opener=lambda i: FakeCap()))
    html = client.post("/inspect", data={"source": "live"}, follow_redirects=True).get_data(as_text=True)
    assert "Start the live view" in html


def test_live_camera_not_found(client, monkeypatch):
    from ui.live_camera import LiveCamera

    class NoCam:
        def isOpened(self):
            return False

        def release(self):
            pass
    monkeypatch.setattr(fabric_app, "LIVE", LiveCamera(opener=lambda i: NoCam()))
    html = client.post("/live/start", follow_redirects=True).get_data(as_text=True)
    assert "Camera not found" in html


def grid_png(missing=()):
    img = np.full((1326, 1740, 3), 225, np.uint8)
    for r in range(6):
        for c in range(8):
            if (r, c) in missing:
                continue
            cv2.circle(img, (130 + c * 210, 130 + r * 210), 70, (60, 60, 120), -1)
    return cv2.imencode(".png", img)[1].tobytes()


def test_pattern_page_refs_and_check(client, monkeypatch, tmp_path):
    import fabric_pattern as fpat
    monkeypatch.setattr(fpat, "REF_DIR", tmp_path / "refs")
    html = client.get("/pattern").get_data(as_text=True)
    assert "Reference photos" in html and "Add reference photos" in html
    for _ in range(2):
        client.post("/pattern/refs", data={"action": "add", "refs": (__import__("io").BytesIO(grid_png()), "a.png")},
                    content_type="multipart/form-data")
    html = client.get("/pattern").get_data(as_text=True)
    assert "ref_01.png" in html and "ref_02.png" in html and "Learned from 2 photo(s)" in html
    assert client.get("/pattern/ref/ref_01.png").mimetype == "image/jpeg"
    r = client.post("/pattern/check", data={"source": "upload", "compare": "1",
                    "photo": (__import__("io").BytesIO(grid_png(missing={(2, 3)})), "t.png")},
                    content_type="multipart/form-data", follow_redirects=True)
    html = r.get_data(as_text=True)
    assert "1 place(s) where a motif seems to be missing" in html and "REVIEW: look at the marked spots" in html
    assert "Comparison with reference photo" in html
    assert client.get("/pattern/img/gaps").mimetype == "image/png"
    r = client.post("/pattern/check", data={"source": "upload", "photo": (__import__("io").BytesIO(grid_png()), "g.png")},
                    content_type="multipart/form-data", follow_redirects=True)
    assert "No missing motif found" in r.get_data(as_text=True)
    client.post("/pattern/refs", data={"action": "delete", "name": "ref_01.png"})
    assert "ref_01.png" not in client.get("/pattern").get_data(as_text=True)


def test_pattern_page_no_file(client):
    html = client.post("/pattern/check", data={"source": "upload"}, follow_redirects=True).get_data(as_text=True)
    assert "Choose a photo first" in html


def test_fabric_selector(client):
    html = client.get("/").get_data(as_text=True)
    assert "<select name=\"fabric\"" in html and "Pattern check" in html


def test_live_camera_black_frames_gives_clear_message(client, monkeypatch):
    from ui.live_camera import LiveCamera

    class BlackCap(FakeCap):
        def read(self):
            return True, np.zeros((480, 640, 3), np.uint8)
    cap = BlackCap()
    monkeypatch.setattr(fabric_app, "LIVE", LiveCamera(opener=lambda i: cap))
    html = client.post("/live/start", follow_redirects=True).get_data(as_text=True)
    assert "sends a black picture" in html and cap.released


def _select(client, monkeypatch):
    monkeypatch.setattr(fpipe, "get_detector", lambda fabric: FakeDet([]))
    return client.post("/select", data={"source": "upload", "photo": (__import__("io").BytesIO(photo()), "a.png")})


def test_select_page_shows_photo_and_rectangle_tool(client, monkeypatch):
    html = _select(client, monkeypatch).get_data(as_text=True)
    assert "Crop an area" in html and "Inspect full photo" in html and "Inspect this part" in html
    assert client.get("/select/img").mimetype == "image/jpeg"


def test_inspect_selected_area_uses_only_that_rectangle(client, monkeypatch):
    _select(client, monkeypatch)
    html = client.post("/inspect/area", data={"fx": "0.1", "fy": "0.2", "fw": "0.5", "fh": "0.6"},
                       follow_redirects=True).get_data(as_text=True)
    rec = fabric_app.STATE["rec"]
    assert rec["crop"]["frac"] == [0.1, 0.2, 0.5, 0.6] and rec["crop"]["small"] is False
    assert "Area inspected" in html and fabric_app.STATE["pending"] is None
    w, h = rec["crop"]["size"]
    ws = fpipe.FAB["fabrics"]["blue"]["warp_size"]
    assert (w, h) == (round(0.5 * ws[0]), round(0.6 * ws[1]))


def test_each_photo_needs_its_own_rectangle(client, monkeypatch):
    _select(client, monkeypatch)
    client.post("/inspect/area", data={"fx": "0.1", "fy": "0.1", "fw": "0.5", "fh": "0.5"})
    r = client.post("/inspect/area", data={"fx": "0.1", "fy": "0.1", "fw": "0.5", "fh": "0.5"}, follow_redirects=True)
    assert "No photo waiting" in r.get_data(as_text=True)


def test_whole_photo_and_tiny_area(client, monkeypatch):
    _select(client, monkeypatch)
    client.post("/inspect/area", data={"whole": "1"})
    assert "crop" not in fabric_app.STATE["rec"]
    _select(client, monkeypatch)
    r = client.post("/inspect/area", data={"fx": "0.1", "fy": "0.1", "fw": "0.001", "fh": "0.001"}, follow_redirects=True)
    assert "too small" in r.get_data(as_text=True)


def test_report_from_log_has_summary_for_each_defect(client, monkeypatch, tmp_path):
    monkeypatch.setattr(fpipe, "REPORT_DIR", tmp_path / "rep")
    big = {"type": "Hole", "confidence": 0.88, "box": (900, 500, 1100, 700), "polygon": None}
    post(client, monkeypatch, [big])
    rid = fabric_app.STATE["rec"]["id"]
    assert f"/report/{rid}" in client.get("/").get_data(as_text=True)
    html = client.get(f"/report/{rid}").get_data(as_text=True)
    assert "Inspection report" in html and "Summary of each defect" in html
    assert "Hole" in html and "Why this severity" in html and "REJECT" in html and "88%" in html
    assert client.get(f"/report/{rid}/picture.png").mimetype == "image/png"
    assert client.get("/report/99999").status_code == 404
    assert str(rid) in client.get("/log").get_data(as_text=True)
