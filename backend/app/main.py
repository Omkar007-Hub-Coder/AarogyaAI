from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.db.session import init_db


@asynccontextmanager
async def lifespan(application: FastAPI):  # noqa: ARG001
    init_db()
    yield


app = FastAPI(
    title="AarogyaAI API",
    description="Personalized Yoga, Fitness & Ahar Recommendation System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "service": "AarogyaAI"}


app.include_router(api_router, prefix="/api/v1")
