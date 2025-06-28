"""Main application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .core.config import settings
from .api.routes import router
from .utils.logging import configure_logging
from .utils.exceptions import (
    RecommendationException,
    recommendation_exception_handler,
    general_exception_handler,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Configure logging
    configure_logging()

    # Create FastAPI app
    app = FastAPI(
        title=settings.api.title,
        description=settings.api.description,
        version=settings.api.version,
        debug=settings.api.debug,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.allow_origins,
        allow_credentials=settings.api.allow_credentials,
        allow_methods=settings.api.allow_methods,
        allow_headers=settings.api.allow_headers,
    )

    # Mount static files
    app.mount(
        f"/{settings.static.static_dir}",
        StaticFiles(directory=settings.static.static_dir),
        name="static",
    )

    # Include API routes
    app.include_router(router)

    # Add exception handlers
    app.add_exception_handler(RecommendationException, recommendation_exception_handler) # type: ignore
    app.add_exception_handler(Exception, general_exception_handler)

    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        """Application startup event."""
        logger.info("Starting E-commerce Facecare Recommender API")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"Debug mode: {settings.api.debug}")

        # Pre-load data on startup to catch any issues early
        try:
            from .services.data_service import data_service

            logger.info("Pre-loading data services...")

            # Trigger data loading
            _ = data_service.popular_products
            _ = data_service.combined_products
            _ = data_service.purchase_data
            _ = data_service.vietnamese_embedding
            _ = data_service.cosine_similarity_matrix
            _ = data_service.product_indices

            logger.info("Data services loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load data services: {e}")
            raise

    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event."""
        logger.info("Shutting down E-commerce Facecare Recommender API")

    return app


# Create the app instance
app = create_app()
