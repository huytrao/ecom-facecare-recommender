"""API routes for the recommendation system."""

from typing import Optional
from fastapi import APIRouter, Depends, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

from ..core.config import settings
from ..models.schemas import RecommendationResponse, ErrorResponse
from ..services.recommendation_service import recommendation_service
from ..services.search_service import search_service

# Initialize templates
templates = Jinja2Templates(directory=settings.static.templates_dir)

# Create router
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page endpoint."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "E-commerce Facecare Recommender",
        "version": settings.api.version,
    }


@router.get(
    "/recommend/popular",
    summary="Get Popular Product Recommendations",
    response_model=dict,
)
async def get_popular_products(num: Optional[int] = 10):
    """
    Get popular product recommendations.

    Args:
        num: Number of products to return (default: 10)

    Returns:
        JSON response with popular products
    """
    try:
        # Validate input
        if num is None or num <= 0:
            num = settings.recommendation.default_num_recommendations

        num = min(num, settings.recommendation.max_recommendations)

        logger.info(f"Generating {num} popular product recommendations")

        # Get popular products
        products = recommendation_service.get_popular_products(num)

        # Convert to dict for JSON response
        products_dict = [product.model_dump() for product in products]

        logger.debug(
            f"Popular recommendations generated: {len(products_dict)} products"
        )
        return JSONResponse(content={"products": products_dict})

    except Exception as e:
        logger.error(f"Error generating popular recommendations: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate recommendations: {str(e)}"},
        )


@router.post(
    "/recommend/hybrid",
    summary="Get Hybrid Product Recommendations",
    response_class=HTMLResponse,
)
async def get_hybrid_recommendations(request: Request, user_id: int = Form(...)):
    """
    Get hybrid product recommendations combining content-based and collaborative filtering.

    Args:
        request: FastAPI request object
        user_id: User ID to get recommendations for

    Returns:
        HTML page with recommendations
    """
    try:
        if not user_id:
            return templates.TemplateResponse(
                "content_based.html",
                {"request": request, "error": "User ID is required"},
            )

        logger.info(f"Generating hybrid recommendations for user {user_id}")

        # Get hybrid recommendations
        recommendation_response = recommendation_service.get_hybrid_recommendations(
            user_id, settings.recommendation.default_num_recommendations
        )

        # Convert products to dict for template
        products_dict = [
            product.model_dump() for product in recommendation_response.products
        ]

        # Prepare template context
        context = {
            "request": request,
            "products": products_dict,
            "user_id": user_id,
            "recommendation_type": "hybrid",
            "message": recommendation_response.message,
        }

        # Add base product if available
        if recommendation_response.based_on_product:
            context["based_on_product"] = (
                recommendation_response.based_on_product.model_dump()
            )

        # Check if we're showing popular products
        if (
            recommendation_response.message
            and "popular products" in recommendation_response.message.lower()
        ):
            context["show_popular"] = True

        logger.debug(f"Hybrid recommendations generated: {len(products_dict)} products")

        return templates.TemplateResponse("content_based.html", context)

    except Exception as e:
        logger.error(f"Error generating hybrid recommendations: {str(e)}")
        return templates.TemplateResponse(
            "content_based.html",
            {
                "request": request,
                "error": f"Failed to generate recommendations: {str(e)}",
            },
        )


@router.get("/search", response_class=HTMLResponse)
async def search_products_page(request: Request, query: Optional[str] = None):
    """
    Search products page endpoint.

    Args:
        request: FastAPI request object
        query: Search query parameter

    Returns:
        HTML page with search results
    """
    try:
        context: dict = {"request": request}

        if query:
            logger.info(f"Processing search request for query: '{query}'")

            # Search for products
            products = search_service.search_products(query, num_results=20)

            # Convert to dict for template
            products_dict = [product.model_dump() for product in products]

            context["query"] = query
            context["products"] = products_dict
            context["results_count"] = len(products_dict)

            logger.info(f"Search completed: {len(products_dict)} products found")

        return templates.TemplateResponse("search.html", context)

    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        error_context: dict = {
            "request": request,
            "query": query,
            "error": f"Search error: {str(e)}",
        }
        return templates.TemplateResponse("search.html", error_context)


@router.post("/api/search")
async def api_search_products(request: Request):
    """
    API endpoint for product search.

    Args:
        request: FastAPI request object with JSON body containing query

    Returns:
        JSON response with search results
    """
    try:
        body = await request.json()
        query = body.get("query", "").strip()

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query parameter is required"},
            )

        logger.info(f"API search request for query: '{query}'")

        # Search for products
        products = search_service.search_products(query, num_results=20)

        # Convert to dict for JSON response
        products_dict = [product.model_dump() for product in products]

        logger.info(f"API search completed: {len(products_dict)} products found")

        return JSONResponse(
            content={
                "query": query,
                "products": products_dict,
                "results_count": len(products_dict),
            }
        )

    except Exception as e:
        logger.error(f"Error processing API search request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Search error: {str(e)}"},
        )
