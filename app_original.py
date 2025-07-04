from typing import Optional
from loguru import logger
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize


# Initialize FastAPI application
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# load data from CSV file
df_population = pd.read_csv("./data/top_brand_products.csv")

IMG_PATH = "static/image/"


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# avg_star,num_sold_time,price,product_id,product_name,brand,origin,type,skin_kind,
# num_rating,processed_description,image_path,popularity_score


@app.get("/recommend/popular", summary="Get Popular Product Recommendations")
async def recommend_popular_products(num: Optional[int] = 10):
    """
    Returns a list of popular products randomly selected from the dataset.
    Popular products are determined by their popularity_score.
    """
    try:
        logger.info(f"Generating {num} popular product recommendations")

        # Randomly select from the popular products
        selected_products = df_population.sample(n=min(num, len(df_population)))

        # Return structured product data as JSON
        products = selected_products.to_dict(orient="records")

        logger.debug(f"Popular recommendations generated: {len(products)} products")
        return JSONResponse(content={"products": products})
    except Exception as e:
        logger.error(f"Error generating popular recommendations: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate recommendations: {str(e)}"},
        )


df_product = pd.read_csv(
    "./data/data_product_combine.csv"
)  # Columns: avg_star,num_sold_time,price,product_id,product_name,brand,origin,type,skin_kind,num_rating,processed_description,image_path,popularity_score,combined_text


id_prod = df_product["product_id"]
indices = pd.Series(df_product.index, index=id_prod)

vietnamese_embedding = np.load("./embeddings/vietnamese_embedding.npy")
cosine_similarity_matrix = linear_kernel(vietnamese_embedding, vietnamese_embedding)
cosine_similarity_matrix


def content_base_recommendation(product_id, num_products=15):

    idx = indices[product_id]
    sim_scores = list(enumerate(cosine_similarity_matrix[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:101]
    df_indices = [i[0] for i in sim_scores]

    popularity_scores = df_product.loc[df_indices, "popularity_score"]
    temp_df = pd.DataFrame(sim_scores, columns=["index", "score"])
    temp_df.index = temp_df["index"]
    temp_df.drop(columns=["index"], axis=1, inplace=True)

    score = temp_df["score"] * 0.7 + popularity_scores * 0.3
    score = score.sort_values(ascending=False)
    product_indices = score.index[:num_products]

    return product_indices


df_purchase = pd.read_csv(
    "data/data_purchase.csv"
)  # Columns: user_id,product_id,cmt_date


# Content-Based Recommendations
@app.post(
    "/recommend/content_based",
    summary="Get Content-Based Product Recommendations",
    response_class=HTMLResponse,
)
async def recommend_content_based_products(request: Request, user_id: int = Form(...)):
    """
    Returns product recommendations based on a user's most recently purchased product.

    Parameters:
    - user_id: The ID of the user to get recommendations for

    Returns:
    - HTML page with product recommendations similar to the user's last purchased product
    """
    try:
        if not user_id:
            return templates.TemplateResponse(
                "content_based.html",
                {"request": request, "error": "User ID is required"},
            )

        logger.info(f"Generating content-based recommendations for user {user_id}")

        # Get the user's purchase history
        user_purchases = df_purchase[df_purchase["user_id"] == user_id]

        if user_purchases.empty:
            logger.warning(
                f"No purchase history found for user {user_id}, showing popular products instead"
            )

            # Set show_popular flag to true to trigger the popular products display in the template
            return templates.TemplateResponse(
                "content_based.html",
                {
                    "request": request,
                    "user_id": user_id,
                    "show_popular": True,
                    "message": "No purchase history found. Showing popular products instead.",
                },
            )

        # Sort by comment date to get the most recent purchase
        most_recent_purchase = user_purchases.sort_values(
            "cmt_date", ascending=False
        ).iloc[0]
        last_product_id = most_recent_purchase["product_id"]

        logger.info(f"User's most recent purchase was product ID: {last_product_id}")

        # Get recommendations based on the last purchased product
        try:
            recommended_indices = content_base_recommendation(
                last_product_id, num_products=15
            )
            recommended_products = df_product.iloc[recommended_indices]

            # Prepare the response
            recommendations = recommended_products.to_dict(orient="records")

            logger.debug(
                f"Content-based recommendations generated: {len(recommendations)} products"
            )

            # Get the base product information
            base_product = df_product.loc[indices[last_product_id]].to_dict()

            return templates.TemplateResponse(
                "content_based.html",
                {
                    "request": request,
                    "products": recommendations,
                    "based_on_product": base_product,
                    "user_id": user_id,
                },
            )

        except KeyError:
            logger.error(f"Product ID {last_product_id} not found in dataset")
            return templates.TemplateResponse(
                "content_based.html",
                {
                    "request": request,
                    "error": f"Product ID {last_product_id} not found in dataset",
                },
            )

    except Exception as e:
        logger.error(f"Error generating content-based recommendations: {str(e)}")
        return templates.TemplateResponse(
            "content_based.html",
            {
                "request": request,
                "error": f"Failed to generate recommendations: {str(e)}",
            },
        )


with open("data/prediction_matrix.npy", "rb") as f:
    loaded_data = np.load(f, allow_pickle=True).item()

prediction_matrix = pd.DataFrame.from_dict(loaded_data["prediction_matrix"])
train_matrix = pd.DataFrame.from_dict(loaded_data["train_matrix"])


def user_based_recommendation(user_id, prediction_matrix, k=10):
    try:
        # Kiểm tra xem user có trong ma trận không
        if user_id not in prediction_matrix.index:
            print(f"User ID {user_id} không tồn tại trong dữ liệu.")
            return []

        # Lấy tất cả dự đoán cho user
        user_predictions = prediction_matrix.loc[user_id].copy()

        # Loại bỏ các giá trị NaN (không có dự đoán)
        user_predictions = user_predictions.dropna()

        # Nếu không có dự đoán nào, trả về danh sách rỗng
        if len(user_predictions) == 0:
            return []

        # Sắp xếp theo điểm dự đoán giảm dần và lấy top-k
        top_products = user_predictions.nlargest(k).index.tolist()
        return top_products
    except Exception as e:
        print(f"Lỗi khi gợi ý cho user {user_id}: {e}")
        return []


def hybrid_recommendation(user_id, k=20):
    """
    Hybrid recommendation function that combines content-based and user-based filtering.
    
    Checks if user exists in prediction matrix before attempting user-based recommendations.
    Falls back to content-based only if user isn't in the collaborative filtering model.

    Args:
        user_id: ID of the user to recommend products for
        k: Total number of products to recommend (will be split between methods)

    Returns:
        List of product_id's recommended for the user
    """
    try:
        logger.info(f"Generating hybrid recommendations for user {user_id}")

        # Get user's purchase history
        user_purchases = df_purchase[df_purchase["user_id"] == user_id]

        # Initialize results
        content_based_products = []
        user_based_products = []

        # Get content-based recommendations if user has purchase history
        if not user_purchases.empty:
            # Get the most recent purchase
            most_recent_purchase = user_purchases.sort_values(
                "cmt_date", ascending=False
            ).iloc[0]
            last_product_id = most_recent_purchase["product_id"]

            # Get content-based recommendations
            try:
                recommended_indices = content_base_recommendation(
                    last_product_id, num_products=k
                )
                content_based_products = df_product.iloc[recommended_indices][
                    "product_id"
                ].tolist()
                logger.debug(
                    f"Content-based recommendations generated: {len(content_based_products)} products"
                )
            except KeyError as e:
                logger.warning(f"Error in content-based recommendation: {str(e)}")

        # Check if user exists in prediction matrix before getting user-based recommendations
        if user_id in prediction_matrix.index:
            logger.info(f"User {user_id} found in prediction matrix, generating user-based recommendations")
            user_based_products = user_based_recommendation(
                user_id, prediction_matrix, k=k // 2
            )
            logger.debug(
                f"User-based recommendations generated: {len(user_based_products)} products"
            )
        else:
            logger.info(f"User {user_id} not found in prediction matrix, using only content-based recommendations")

        # Combine both recommendation types
        hybrid_recommendations = []
        
        # If we have user-based recommendations, prioritize them
        if user_based_products:
            hybrid_recommendations = user_based_products.copy()
            
            # Add content-based recommendations that aren't already in the list
            for product_id in content_based_products:
                if product_id not in hybrid_recommendations:
                    hybrid_recommendations.append(product_id)
        else:
            # If no user-based recommendations, use only content-based
            hybrid_recommendations = content_based_products

        # Limit to k recommendations
        hybrid_recommendations = hybrid_recommendations[:k]

        logger.info(
            f"Hybrid recommendations generated: {len(hybrid_recommendations)} products"
        )
        return hybrid_recommendations

    except Exception as e:
        logger.error(f"Error generating hybrid recommendations: {str(e)}")
        return []


@app.post(
    "/recommend/hybrid",
    summary="Get Hybrid Product Recommendations",
    response_class=HTMLResponse,
)
async def recommend_hybrid_products(request: Request, user_id: int = Form(...)):
    """
    Returns product recommendations using a hybrid approach that combines
    content-based and collaborative filtering methods.

    Parameters:
    - user_id: The ID of the user to get recommendations for

    Returns:
    - HTML page with hybrid product recommendations
    """
    try:
        if not user_id:
            return templates.TemplateResponse(
                "content_based.html",
                {"request": request, "error": "User ID is required"},
            )

        logger.info(f"Processing hybrid recommendation request for user {user_id}")

        # Get hybrid recommendations
        recommended_product_ids = hybrid_recommendation(user_id, k=20)

        if not recommended_product_ids:
            logger.warning(
                f"No hybrid recommendations found for user {user_id}, showing popular products instead"
            )
            return templates.TemplateResponse(
                "content_based.html",
                {
                    "request": request,
                    "user_id": user_id,
                    "show_popular": True,
                    "message": "No personalized recommendations found. Showing popular products instead.",
                },
            )

        # Get full product details for the recommended product IDs
        recommended_products = df_product[
            df_product["product_id"].isin(recommended_product_ids)
        ]

        # Sort products to match the order in recommended_product_ids
        recommended_products["sort_order"] = recommended_products["product_id"].apply(
            lambda x: (
                recommended_product_ids.index(x)
                if x in recommended_product_ids
                else len(recommended_product_ids)
            )
        )
        recommended_products = recommended_products.sort_values("sort_order")
        recommended_products = recommended_products.drop("sort_order", axis=1)

        # Prepare the response
        recommendations = recommended_products.to_dict(orient="records")

        logger.debug(
            f"Hybrid recommendations to display: {len(recommendations)} products"
        )

        return templates.TemplateResponse(
            "content_based.html",
            {
                "request": request,
                "products": recommendations,
                "user_id": user_id,
                "recommendation_type": "hybrid",
                "message": "Recommendations generated using hybrid method (content-based + collaborative filtering)",
            },
        )

    except Exception as e:
        logger.error(f"Error in hybrid recommendation endpoint: {str(e)}")
        return templates.TemplateResponse(
            "content_based.html",
            {
                "request": request,
                "error": f"Failed to generate hybrid recommendations: {str(e)}",
            },
        )
