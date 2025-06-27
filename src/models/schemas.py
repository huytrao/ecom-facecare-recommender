"""Data models for the recommendation system."""

from typing import Optional
from pydantic import BaseModel


class Product(BaseModel):
    """Product data model."""

    product_id: int
    product_name: str
    brand: Optional[str] = None
    origin: Optional[str] = None
    type: Optional[str] = None
    skin_kind: Optional[str] = None
    price: Optional[float] = None
    avg_star: Optional[float] = None
    num_rating: Optional[int] = None
    num_sold_time: Optional[int] = None
    image_path: Optional[str] = None
    popularity_score: Optional[float] = None
    processed_description: Optional[str] = None
    combined_text: Optional[str] = None


class Purchase(BaseModel):
    """Purchase data model."""

    user_id: int
    product_id: int
    cmt_date: str


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""

    user_id: int
    num_recommendations: Optional[int] = 15


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""

    products: list[Product]
    based_on_product: Optional[Product] = None
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
