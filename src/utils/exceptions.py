"""Exception handling utilities."""

from typing import Any, Dict, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger


class RecommendationException(Exception):
    """Base exception for recommendation system."""

    def __init__(self, message: str, detail: Optional[str] = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class DataLoadException(RecommendationException):
    """Exception raised when data loading fails."""

    pass


class UserNotFoundException(RecommendationException):
    """Exception raised when user is not found."""

    pass


class ProductNotFoundException(RecommendationException):
    """Exception raised when product is not found."""

    pass


async def recommendation_exception_handler(
    request: Request, exc: RecommendationException
):
    """Handle recommendation system exceptions."""
    logger.error(f"Recommendation error: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "detail": exc.detail,
            "type": exc.__class__.__name__,
        },
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
        },
    )
