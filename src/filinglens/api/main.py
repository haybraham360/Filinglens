"""FastAPI application (see specs/06)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from filinglens import __version__
from filinglens.config import get_settings


class Health(BaseModel):
    """Liveness response."""

    status: str
    version: str


def create_app() -> FastAPI:
    """Build the FastAPI app with CORS restricted to configured origins."""
    settings = get_settings()
    api = FastAPI(title="FilingLens API", version=__version__)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @api.get("/health", response_model=Health)
    def health() -> Health:
        """Report that the service is up."""
        return Health(status="ok", version=__version__)

    return api


app = create_app()
