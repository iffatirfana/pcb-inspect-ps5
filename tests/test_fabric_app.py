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
