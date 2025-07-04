"""Search service for retrieving and processing search results."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .data_service import data_service
from ..models.schemas import Product

vietnamese_bi_encoder = SentenceTransformer(
    "bkai-foundation-models/vietnamese-bi-encoder", cache_folder="./models"
)


class SearchService:
    """Service for handling product search functionality."""

    def __init__(self):
        self._vie_bi_encode_embedding: Optional[np.ndarray] = None

    @property
    def vie_bi_encode_embedding(self) -> np.ndarray:
        """Get Vietnamese bi-encoder embedding matrix."""
        if self._vie_bi_encode_embedding is None:
            self._load_vie_bi_encode_embedding()
        assert self._vie_bi_encode_embedding is not None
        return self._vie_bi_encode_embedding

    def _load_vie_bi_encode_embedding(self) -> None:
        """Load Vietnamese bi-encoder embedding matrix."""
        try:
            logger.info("Loading Vietnamese bi-encoder embedding matrix...")
            embedding = np.load("./embeddings/vie_bi_encode_embedding.npy")
            self._vie_bi_encode_embedding = embedding
            logger.info(f"Loaded embedding matrix with shape {embedding.shape}")
        except Exception as e:
            logger.error(f"Failed to load Vietnamese bi-encoder embedding: {e}")
            raise

    def search_products(self, query: str, num_results: int = 10) -> List[Product]:
        """
        Search for products using semantic similarity.

        Args:
            query: Search query from user
            num_results: Number of results to return

        Returns:
            List of matching products sorted by similarity score
        """
        try:
            logger.info(f"Searching for products with query: '{query}'")

            # Encode the search query
            query_embedding = vietnamese_bi_encoder.encode([query])

            # Load product embeddings
            product_embeddings = self.vie_bi_encode_embedding

            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, product_embeddings)[0]

            # Get indices of most similar products
            similar_indices = np.argsort(similarities)[::-1][:num_results]

            # Get product data
            products_df = data_service.combined_products

            # Convert to Product models
            products = []
            for idx in similar_indices:
                if idx < len(products_df):
                    row = products_df.iloc[idx]
                    product = Product(
                        product_id=int(row["product_id"]),
                        product_name=row["product_name"],
                        brand=row.get("brand"),
                        origin=row.get("origin"),
                        type=row.get("type"),
                        skin_kind=row.get("skin_kind"),
                        price=float(row["price"]) if pd.notna(row["price"]) else None,
                        avg_star=(
                            float(row["avg_star"])
                            if pd.notna(row["avg_star"])
                            else None
                        ),
                        num_rating=(
                            int(row["num_rating"])
                            if pd.notna(row["num_rating"])
                            else None
                        ),
                        num_sold_time=(
                            int(row["num_sold_time"])
                            if pd.notna(row["num_sold_time"])
                            else None
                        ),
                        image_path=row.get("image_path"),
                        popularity_score=(
                            float(row["popularity_score"])
                            if pd.notna(row["popularity_score"])
                            else None
                        ),
                        processed_description=row.get("processed_description"),
                        combined_text=row.get("combined_text"),
                    )
                    products.append(product)

            logger.info(f"Found {len(products)} similar products")
            return products

        except Exception as e:
            logger.error(f"Error searching products: {str(e)}")
            raise


# Global search service instance
search_service = SearchService()
