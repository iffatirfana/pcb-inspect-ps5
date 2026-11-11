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
    assert "not calibrated" in html


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
