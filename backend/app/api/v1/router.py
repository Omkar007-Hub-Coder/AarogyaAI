from fastapi import APIRouter

from app.api.v1 import profiles, recommendations, catalog, ml_endpoints, personalization, feedback

api_router = APIRouter()
api_router.include_router(profiles.router,          prefix="/profile",          tags=["profiles"])
api_router.include_router(recommendations.router,   prefix="/recommendations",  tags=["recommendations"])
api_router.include_router(catalog.router,           prefix="/catalog",          tags=["catalog"])
api_router.include_router(ml_endpoints.router,      prefix="/ml",               tags=["ml"])
api_router.include_router(personalization.router,   prefix="/personalization",  tags=["personalization"])
api_router.include_router(feedback.router,          prefix="/feedback",         tags=["feedback"])
