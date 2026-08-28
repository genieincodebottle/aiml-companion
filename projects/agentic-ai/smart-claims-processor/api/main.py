"""
Smart Claims Processor - FastAPI backend.

Run locally:
    uvicorn api.main:app --port 8000
"""
from __future__ import annotations #For Python 3.10+ to allow forward references in type hints without quotes Ensure this is at the top of the file before any imports or code.

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from api.db import engine, init_db
from api.routes_analytics import router as analytics_router
from api.routes_appeals import router as appeals_router
from api.routes_auth import router as auth_router
from api.routes_claims import router as claims_router
from api.routes_hitl import router as hitl_router
from api.routes_policies import router as policies_router
from api.routes_settings import router as settings_router
from api.security import seed_admin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create tables + seed dev users. Shutdown: nothing special."""
    init_db()
    with Session(engine) as session:
        seed_admin(session)

    # Seed fraud knowledge base into ChromaDB (idempotent)
    try:
        from src.memory.manager import memory
        from src.tools.fraud_patterns import get_patterns
        patterns = [
            {"id": p["id"], "name": p["name"], "description": p["description"],
             "risk_weight": p["risk_weight"]}
            for p in get_patterns()
        ]
        memory.seed_fraud_knowledge(patterns)
    except Exception as e:
        logger.debug("Fraud knowledge seeding skipped: %s", e)

    logger.info(
        "Smart Claims API ready. Dev users: admin/admin123 (admin), "
        "reviewer1/review123 + reviewer2/review123 (reviewer), claimant/claim123 (user)."
    )
    yield


app = FastAPI(title="Smart Claims Processor API", version="1.0.0", lifespan=lifespan)

# In dev we accept ANY localhost / 127.0.0.1 port so a Vite port change
# doesn't break the frontend with a preflight 400. An explicit
# API_CORS_ORIGINS env overrides the regex with an exact list for production.
# The regex allows both http and https to work in dev, since some users have https://localhost setups.
_cors_env = os.getenv("API_CORS_ORIGINS", "").strip()
if _cors_env:
    _cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
    _cors_regex = None
else:
    _cors_origins = []
    _cors_regex = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

# CORS setup: allow frontend origins to access the API. In production, set API_CORS_ORIGINS to a comma-separated list of allowed origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "smart-claims-processor", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}

# Register API routers for different modules (auth, claims, HITL, appeals, analytics, policies, settings). 
# Each router is defined in its respective module under api/routes_*.py and contains related endpoints. 
# This modular structure keeps the code organized and maintainable as the application grows.
app.include_router(auth_router)
app.include_router(claims_router)
app.include_router(hitl_router)
app.include_router(appeals_router)
app.include_router(analytics_router)
app.include_router(policies_router)
app.include_router(settings_router)
