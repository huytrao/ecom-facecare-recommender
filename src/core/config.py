"""Configuration settings for the e-commerce recommendation system."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    popular_products_path: str = os.getenv(
        "DATABASE__POPULAR_PRODUCTS_PATH", "./data/top_brand_products.csv"
    )
    combined_products_path: str = os.getenv(
        "DATABASE__COMBINED_PRODUCTS_PATH", "./data/data_product_combine.csv"
    )
    purchase_data_path: str = os.getenv(
        "DATABASE__PURCHASE_DATA_PATH", "./data/data_purchase.csv"
    )
    vietnamese_embedding_path: str = os.getenv(
        "DATABASE__VIETNAMESE_EMBEDDING_PATH", "./embeddings/vietnamese_embedding.npy"
    )


class APIConfig(BaseModel):
    """API configuration settings."""

    title: str = os.getenv("API__TITLE", "E-commerce Facecare Recommender")
    description: str = os.getenv(
        "API__DESCRIPTION",
        "AI-powered product recommendation system for skincare products",
    )
    version: str = os.getenv("API__VERSION", "1.0.0")
    debug: bool = os.getenv("API__DEBUG", "false").lower() == "true"

    # CORS settings
    allow_origins: list[str] = os.getenv("API__ALLOW_ORIGINS", "*").split(",")
    allow_credentials: bool = (
        os.getenv("API__ALLOW_CREDENTIALS", "true").lower() == "true"
    )
    allow_methods: list[str] = os.getenv("API__ALLOW_METHODS", "*").split(",")
    allow_headers: list[str] = os.getenv("API__ALLOW_HEADERS", "*").split(",")


class RecommendationConfig(BaseModel):
    """Recommendation engine configuration."""

    default_num_recommendations: int = int(
        os.getenv("RECOMMENDATION__DEFAULT_NUM_RECOMMENDATIONS", "15")
    )
    max_recommendations: int = int(
        os.getenv("RECOMMENDATION__MAX_RECOMMENDATIONS", "50")
    )
    similarity_weight: float = float(
        os.getenv("RECOMMENDATION__SIMILARITY_WEIGHT", "0.7")
    )
    popularity_weight: float = float(
        os.getenv("RECOMMENDATION__POPULARITY_WEIGHT", "0.3")
    )
    top_similar_products: int = int(
        os.getenv("RECOMMENDATION__TOP_SIMILAR_PRODUCTS", "100")
    )


class StaticConfig(BaseModel):
    """Static files configuration."""

    static_dir: str = os.getenv("STATIC__STATIC_DIR", "static")
    templates_dir: str = os.getenv("STATIC__TEMPLATES_DIR", "templates")
    image_dir: str = os.getenv("STATIC__IMAGE_DIR", "static/image")


class Settings(BaseModel):
    """Main application settings."""

    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Component configurations
    database: DatabaseConfig = DatabaseConfig()
    api: APIConfig = APIConfig()
    recommendation: RecommendationConfig = RecommendationConfig()
    static: StaticConfig = StaticConfig()


# Global settings instance
settings = Settings()
