"""
Tests for backend API endpoints.
Run from AarogyaAI/backend/:
    pytest ../tests/test_api.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import Base, engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_db():
    """Recreate tables before each test for isolation."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


VALID_PROFILE = {
    "name": "Test User",
    "age": 30,
    "gender": "male",
    "height_cm": 170.0,
    "weight_kg": 70.0,
    "activity_level": "moderate",
    "health_goal": "general_wellness",
    "dietary_preference": "none",
}


def test_create_profile():
    res = client.post("/api/v1/profile", json=VALID_PROFILE)
    assert res.status_code == 201
    data = res.json()
    assert data["id"] == 1
    assert data["name"] == "Test User"


def test_get_profile():
    create_res = client.post("/api/v1/profile", json=VALID_PROFILE)
    user_id = create_res.json()["id"]

    res = client.get(f"/api/v1/profile/{user_id}")
    assert res.status_code == 200
    assert res.json()["name"] == "Test User"


def test_get_profile_not_found():
    res = client.get("/api/v1/profile/9999")
    assert res.status_code == 404


def test_get_recommendations():
    create_res = client.post("/api/v1/profile", json=VALID_PROFILE)
    user_id = create_res.json()["id"]

    res = client.get(f"/api/v1/recommendations/{user_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == user_id
    assert "yoga" in data
    assert "fitness" in data
    assert "ahar" in data


def test_get_recommendations_not_found():
    res = client.get("/api/v1/recommendations/9999")
    assert res.status_code == 404


def test_catalog_yoga():
    res = client.get("/api/v1/catalog/yoga")
    assert res.status_code == 200
    assert len(res.json()["items"]) > 0


def test_catalog_fitness():
    res = client.get("/api/v1/catalog/fitness")
    assert res.status_code == 200
    assert len(res.json()["items"]) > 0


def test_catalog_ahar():
    res = client.get("/api/v1/catalog/ahar")
    assert res.status_code == 200
    assert len(res.json()["items"]) > 0


def test_invalid_profile_age():
    bad = {**VALID_PROFILE, "age": 200}
    res = client.post("/api/v1/profile", json=bad)
    assert res.status_code == 422


def test_invalid_profile_gender():
    bad = {**VALID_PROFILE, "gender": "robot"}
    res = client.post("/api/v1/profile", json=bad)
    assert res.status_code == 422
