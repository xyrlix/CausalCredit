"""FastAPI application entry point.

CausalCredit — Causal Inference Enhanced Credit Scoring System API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

app = FastAPI(
    title="CausalCredit API",
    description="Causal Inference Enhanced Credit Scoring System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""


app.include_router(router)
