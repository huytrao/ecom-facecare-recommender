"""Recommendation service for product recommendations."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from loguru import logger

from ..core.config import settings
from ..models.schemas import Product, RecommendationResponse
from .data_service import data_service


class RecommendationService:
    """Service for generating product recommendations."""

    def __init__(self):
        self.data_service = data_service

    def get_popular_products(self, num_products: int = 10) -> List[Product]:
        """
        Get popular products.

        Args:
            num_products: Number of products to return

        Returns:
            List of popular products
        """
        try:
            logger.info(f"Getting {num_products} popular products")

            # Get popular products dataframe
            df_popular = self.data_service.popular_products

            # Randomly sample from popular products
            num_products = min(num_products, len(df_popular))
            selected_products = df_popular.sample(n=num_products)

            # Convert to Product models
            products = []
            for _, row in selected_products.iterrows():
                product = Product(**row.to_dict())
                products.append(product)

            logger.debug(f"Retrieved {len(products)} popular products")
            return products

        except Exception as e:
            logger.error(f"Error getting popular products: {e}")
            raise

    def get_content_based_recommendations(
        self, user_id: int, num_products: int = 15
    ) -> RecommendationResponse:
        """
        Get content-based recommendations for a user.

        Args:
            user_id: User ID to get recommendations for
            num_products: Number of products to recommend

        Returns:
            RecommendationResponse with products and metadata
        """
        try:
            logger.info(f"Getting content-based recommendations for user {user_id}")

            # Get user's purchase history
            user_purchases = self.data_service.get_user_purchases(user_id)

            if user_purchases.empty:
                logger.warning(f"No purchase history found for user {user_id}")
                popular_products = self.get_popular_products(num_products)
                return RecommendationResponse(
                    products=popular_products,
                    message="No purchase history found. Showing popular products instead.",
                )

            # Get most recent purchase
            most_recent_purchase = user_purchases.sort_values(
                "cmt_date", ascending=False
            ).iloc[0]
            last_product_id = most_recent_purchase["product_id"]

            logger.info(f"User's most recent purchase: product ID {last_product_id}")

            # Get base product information
            base_product_row = self.data_service.get_product_by_id(last_product_id)
            if base_product_row is None:
                logger.error(f"Product ID {last_product_id} not found in dataset")
                raise ValueError(f"Product ID {last_product_id} not found")

            base_product = Product(**base_product_row.to_dict())

            # Get recommendations based on the last purchased product
            recommended_indices = self._get_similar_products(
                last_product_id, num_products
            )

            # Get recommended products
            df_products = self.data_service.combined_products
            recommended_products_df = df_products.iloc[recommended_indices]

            # Convert to Product models
            recommended_products = []
            for _, row in recommended_products_df.iterrows():
                product = Product(**row.to_dict())
                recommended_products.append(product)

            logger.debug(
                f"Generated {len(recommended_products)} content-based recommendations"
            )

            return RecommendationResponse(
                products=recommended_products, based_on_product=base_product
            )

        except Exception as e:
            logger.error(f"Error generating content-based recommendations: {e}")
            raise

    def _get_similar_products(self, product_id: int, num_products: int) -> List[int]:
        """
        Get similar products based on content similarity.

        Args:
            product_id: Base product ID
            num_products: Number of similar products to return

        Returns:
            List of product indices
        """
        try:
            # Get product index
            product_indices = self.data_service.product_indices
            idx = product_indices[product_id]

            # Get similarity scores
            similarity_matrix = self.data_service.cosine_similarity_matrix
            sim_scores = list(enumerate(similarity_matrix[idx]))

            # Sort by similarity score (excluding the product itself)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[
                1 : settings.recommendation.top_similar_products + 1
            ]

            # Get product indices and similarity scores
            df_indices = [i[0] for i in sim_scores]

            # Get popularity scores for ranking
            df_products = self.data_service.combined_products
            popularity_scores = df_products.loc[df_indices, "popularity_score"]

            # Create temporary dataframe for scoring
            temp_df = pd.DataFrame(sim_scores, columns=["index", "score"])
            temp_df = temp_df.set_index("index")

            # Combine similarity and popularity scores
            similarity_weight = settings.recommendation.similarity_weight
            popularity_weight = settings.recommendation.popularity_weight

            combined_score = (
                temp_df["score"] * similarity_weight
                + popularity_scores * popularity_weight
            )

            # Sort by combined score and return top products
            combined_score = combined_score.sort_values(ascending=False)
            product_indices_result = combined_score.index[:num_products].tolist()

            return product_indices_result

        except Exception as e:
            logger.error(f"Error getting similar products: {e}")
            raise


# Global recommendation service instance
recommendation_service = RecommendationService()
