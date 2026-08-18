"""Entrypoint: uvicorn main:app --reload --port 8000"""

from src.api.server import app

__all__ = ["app"]
