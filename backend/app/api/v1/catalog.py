"""Static catalog endpoints — returns knowledge base items (no DB needed)."""
from fastapi import APIRouter
from app.services.recommender import YOGA_CATALOG, FITNESS_CATALOG, AHAR_CATALOG

router = APIRouter()


@router.get("/yoga")
def list_yoga():
    return {"items": [{"name": i["name"], "goals": i["goals"], "levels": i["levels"]} for i in YOGA_CATALOG]}


@router.get("/fitness")
def list_fitness():
    return {"items": [{"name": i["name"], "goals": i["goals"], "levels": i["levels"]} for i in FITNESS_CATALOG]}


@router.get("/ahar")
def list_ahar():
    return {"items": [{"name": i["name"], "goals": i["goals"], "diet": i["diet"]} for i in AHAR_CATALOG]}
