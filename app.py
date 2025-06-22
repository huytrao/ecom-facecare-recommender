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
        

