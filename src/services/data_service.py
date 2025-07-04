"""Data loading and management service."""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger
from sklearn.metrics.pairwise import linear_kernel

from ..core.config import settings


class DataService:
    """Service for managing data loading and access."""

    def __init__(self):
        self._popular_products: Optional[pd.DataFrame] = None
        self._combined_products: Optional[pd.DataFrame] = None
        self._purchase_data: Optional[pd.DataFrame] = None
        self._vietnamese_embedding: Optional[np.ndarray] = None
        self._cosine_similarity_matrix: Optional[np.ndarray] = None
        self._product_indices: Optional[pd.Series] = None

    @property
    def popular_products(self) -> pd.DataFrame:
        """Get popular products dataframe."""
        if self._popular_products is None:
            self._load_popular_products()
        assert self._popular_products is not None
        return self._popular_products

    @property
    def combined_products(self) -> pd.DataFrame:
        """Get combined products dataframe."""
        if self._combined_products is None:
            self._load_combined_products()
        assert self._combined_products is not None
        return self._combined_products

    @property
    def purchase_data(self) -> pd.DataFrame:
        """Get purchase data dataframe."""
        if self._purchase_data is None:
            self._load_purchase_data()
        assert self._purchase_data is not None
        return self._purchase_data

    @property
    def vietnamese_embedding(self) -> np.ndarray:
        """Get Vietnamese embedding matrix."""
        if self._vietnamese_embedding is None:
            self._load_vietnamese_embedding()
        assert self._vietnamese_embedding is not None
        return self._vietnamese_embedding

    @property
    def cosine_similarity_matrix(self) -> np.ndarray:
        """Get cosine similarity matrix."""
        if self._cosine_similarity_matrix is None:
            self._compute_cosine_similarity()
        assert self._cosine_similarity_matrix is not None
        return self._cosine_similarity_matrix

    @property
    def product_indices(self) -> pd.Series:
        """Get product indices mapping."""
        if self._product_indices is None:
            self._create_product_indices()
        assert self._product_indices is not None
        return self._product_indices

    def _load_popular_products(self) -> None:
        """Load popular products data."""
        try:
            logger.info("Loading popular products data...")
            self._popular_products = pd.read_csv(
                settings.database.popular_products_path
            )
            logger.info(f"Loaded {len(self._popular_products)} popular products")
        except Exception as e:
            logger.error(f"Failed to load popular products: {e}")
            raise

    def _load_combined_products(self) -> None:
        """Load combined products data."""
        try:
            logger.info("Loading combined products data...")
            self._combined_products = pd.read_csv(
                settings.database.combined_products_path
            )
            logger.info(f"Loaded {len(self._combined_products)} combined products")
        except Exception as e:
            logger.error(f"Failed to load combined products: {e}")
            raise

    def _load_purchase_data(self) -> None:
        """Load purchase data."""
        try:
            logger.info("Loading purchase data...")
            self._purchase_data = pd.read_csv(settings.database.purchase_data_path)
            logger.info(f"Loaded {len(self._purchase_data)} purchase records")
        except Exception as e:
            logger.error(f"Failed to load purchase data: {e}")
            raise

    def _load_vietnamese_embedding(self) -> None:
        """Load Vietnamese embedding matrix."""
        try:
            logger.info("Loading Vietnamese embedding matrix...")
            self._vietnamese_embedding = np.load(
                settings.database.vietnamese_embedding_path
            )
            logger.info(
                f"Loaded embedding matrix with shape {self._vietnamese_embedding.shape}"
            )
        except Exception as e:
            logger.error(f"Failed to load Vietnamese embedding: {e}")
            raise

    def _compute_cosine_similarity(self) -> None:
        """Compute cosine similarity matrix."""
        try:
            logger.info("Computing cosine similarity matrix...")
            embedding = self.vietnamese_embedding
            self._cosine_similarity_matrix = linear_kernel(embedding, embedding)
            logger.info(
                f"Computed similarity matrix with shape {self._cosine_similarity_matrix.shape}"
            )
        except Exception as e:
            logger.error(f"Failed to compute cosine similarity: {e}")
            raise

    def _create_product_indices(self) -> None:
        """Create product indices mapping."""
        try:
            logger.info("Creating product indices mapping...")
            df = self.combined_products
            self._product_indices = pd.Series(df.index, index=df["product_id"])
            logger.info(f"Created indices for {len(self._product_indices)} products")
        except Exception as e:
            logger.error(f"Failed to create product indices: {e}")
            raise

    def get_product_by_id(self, product_id: int) -> Optional[pd.Series]:
        """Get product by ID."""
        try:
            idx = self.product_indices[product_id]
            return self.combined_products.iloc[idx]
        except KeyError:
            logger.warning(f"Product ID {product_id} not found")
            return None

    def get_user_purchases(self, user_id: int) -> pd.DataFrame:
        """Get user purchase history."""
        return self.purchase_data[self.purchase_data["user_id"] == user_id]

    def reload_data(self) -> None:
        """Reload all data."""
        logger.info("Reloading all data...")
        self._popular_products = None
        self._combined_products = None
        self._purchase_data = None
        self._vietnamese_embedding = None
        self._cosine_similarity_matrix = None
        self._product_indices = None
        logger.info("Data reload completed")


# Global data service instance
data_service = DataService()
