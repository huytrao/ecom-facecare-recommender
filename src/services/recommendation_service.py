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

    def get_hybrid_recommendations(
        self, user_id: int, num_products: int = 20
    ) -> RecommendationResponse:
        """
        Get hybrid recommendations combining content-based and collaborative filtering.

        Args:
            user_id: User ID to get recommendations for
            num_products: Number of products to recommend

        Returns:
            RecommendationResponse with products and metadata
        """
        try:
            logger.info(f"Getting hybrid recommendations for user {user_id}")

            # Get user's purchase history
            user_purchases = self.data_service.get_user_purchases(user_id)

            # Initialize results
            content_based_product_ids = []
            user_based_product_ids = []
            base_product = None

            # Get content-based recommendations if user has purchase history
            if not user_purchases.empty:
                # Get the most recent purchase
                most_recent_purchase = user_purchases.sort_values(
                    "cmt_date", ascending=False
                ).iloc[0]
                last_product_id = most_recent_purchase["product_id"]

                logger.info(
                    f"User's most recent purchase: product ID {last_product_id}"
                )

                # Get base product information
                base_product_row = self.data_service.get_product_by_id(last_product_id)
                if base_product_row is not None:
                    base_product = Product(**base_product_row.to_dict())

                # Get content-based recommendations
                try:
                    recommended_indices = self._get_similar_products(
                        last_product_id, num_products
                    )
                    df_products = self.data_service.combined_products
                    content_based_product_ids = df_products.iloc[recommended_indices][
                        "product_id"
                    ].tolist()
                    logger.debug(
                        f"Content-based recommendations generated: {len(content_based_product_ids)} products"
                    )
                except Exception as e:
                    logger.warning(f"Error in content-based recommendation: {str(e)}")

            # Get user-based recommendations if user exists in prediction matrix
            try:
                user_based_product_ids = self._get_user_based_recommendations(
                    user_id, num_products // 2
                )
                logger.debug(
                    f"User-based recommendations generated: {len(user_based_product_ids)} products"
                )
            except Exception as e:
                logger.warning(f"Error in user-based recommendation: {str(e)}")

            # Combine both recommendation types
            hybrid_product_ids = []

            # If we have user-based recommendations, prioritize them
            if user_based_product_ids:
                hybrid_product_ids = user_based_product_ids.copy()

                # Add content-based recommendations that aren't already in the list
                for product_id in content_based_product_ids:
                    if product_id not in hybrid_product_ids:
                        hybrid_product_ids.append(product_id)
            else:
                # If no user-based recommendations, use only content-based
                hybrid_product_ids = content_based_product_ids

            # Limit to requested number of products
            hybrid_product_ids = hybrid_product_ids[:num_products]

            if not hybrid_product_ids:
                logger.warning(f"No hybrid recommendations found for user {user_id}")
                popular_products = self.get_popular_products(num_products)
                return RecommendationResponse(
                    products=popular_products,
                    message="No personalized recommendations found. Showing popular products instead.",
                )

            # Get full product details for the recommended product IDs
            df_products = self.data_service.combined_products
            recommended_products_df = df_products[
                df_products["product_id"].isin(hybrid_product_ids)
            ].copy()

            # Sort products to match the order in hybrid_product_ids
            recommended_products_df["sort_order"] = recommended_products_df[
                "product_id"
            ].apply(
                lambda x: (
                    hybrid_product_ids.index(x)
                    if x in hybrid_product_ids
                    else len(hybrid_product_ids)
                )
            )
            recommended_products_df = recommended_products_df.sort_values("sort_order")
            recommended_products_df = recommended_products_df.drop("sort_order", axis=1)

            # Convert to Product models
            recommended_products = []
            for _, row in recommended_products_df.iterrows():
                product = Product(**row.to_dict())
                recommended_products.append(product)

            logger.info(
                f"Hybrid recommendations generated: {len(recommended_products)} products"
            )

            # Create appropriate message based on what was available
            if user_based_product_ids and content_based_product_ids:
                message = "Recommendations generated using hybrid method (content-based + collaborative filtering)"
            elif user_based_product_ids:
                message = "Recommendations generated using collaborative filtering only"
            elif content_based_product_ids:
                message = "Recommendations generated using content-based method only (collaborative filtering unavailable)"
            else:
                message = "Showing popular products (no personalized recommendations available)"

            return RecommendationResponse(
                products=recommended_products,
                based_on_product=base_product,
                message=message,
            )

        except Exception as e:
            logger.error(f"Error generating hybrid recommendations: {e}")
            raise

    def _get_user_based_recommendations(self, user_id: int, k: int = 10) -> List[int]:
        """
        Get user-based collaborative filtering recommendations.

        Args:
            user_id: User ID to get recommendations for
            k: Number of products to recommend

        Returns:
            List of product IDs
        """
        try:
            # Load prediction matrix if not already loaded
            prediction_matrix = self.data_service.prediction_matrix

            if prediction_matrix is None:
                logger.warning("Prediction matrix not available")
                return []

            # Check if user exists in prediction matrix
            if user_id not in prediction_matrix.index:
                logger.info(f"User {user_id} not found in prediction matrix")
                return []

            logger.info(
                f"User {user_id} found in prediction matrix, generating user-based recommendations"
            )

            # Get all predictions for user
            user_predictions = prediction_matrix.loc[user_id].copy()

            # Remove NaN values (no predictions)
            user_predictions = user_predictions.dropna()

            # If no predictions, return empty list
            if len(user_predictions) == 0:
                return []

            # Sort by predicted rating descending and get top-k
            top_products = user_predictions.nlargest(k).index.tolist()
            return top_products

        except Exception as e:
            logger.error(
                f"Error getting user-based recommendations for user {user_id}: {e}"
            )
            return []


# Global recommendation service instance
recommendation_service = RecommendationService()
