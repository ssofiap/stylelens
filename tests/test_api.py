import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as m


class FakeSession:
    def run(self, _outputs, feeds):
        (x,) = feeds.values()
        assert x.shape == (1, 3, 224, 224)
        return [np.full((1, 512), 2.0, np.float32)]  # deliberately NOT unit-norm


class Hit:
    def __init__(self, id_, score):
        self.id = id_
        self.score = score
        self.payload = {"articleType": "Tshirts"}


class FakeResult:
    def __init__(self, points):
        self.points = points


class FakeQdrant:
    def __init__(self):
        self.query = None

    def query_points(self, collection, query, limit):
        self.query = np.asarray(query)
        return FakeResult([Hit(i, 0.9 - 0.01 * i) for i in range(limit)])


@pytest.fixture
def client(monkeypatch):
    q = FakeQdrant()
    monkeypatch.setitem(m.state, "sess", FakeSession())
    monkeypatch.setitem(m.state, "input", "x")
    monkeypatch.setitem(m.state, "qdrant", q)
    monkeypatch.setitem(m.state, "version", "test")
    # No with: lifespan does not run, so no MLflow/Qdrant connection is attempted.
    return TestClient(m.app), q


def png_bytes(w=300, h=200):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 40, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_healthz_reports_version(client):
    c, _ = client
    assert c.get("/healthz").json() == {"status": "ok", "model_version": "test"}


def test_search_returns_k_and_normalizes_query(client):
    c, q = client
    r = c.post("/search?k=3", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3
    assert np.isclose(np.linalg.norm(q.query), 1.0, atol=1e-5)  # must match indexer


def test_invalid_image_is_400(client):
    c, _ = client
    r = c.post("/search", files={"file": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400


@pytest.mark.parametrize("k", [0, 51])
def test_k_bounds(client, k):
    c, _ = client
    r = c.post(f"/search?k={k}", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 422


def test_metrics_exposed(client):
    c, _ = client
    assert "stylelens_requests_total" in c.get("/metrics").text
