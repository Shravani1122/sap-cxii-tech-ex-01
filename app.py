"""
FastAPI Microservice for Product Similarity Search

Endpoints:
- GET /find_similar_products: Find similar products by product ID
- GET /health: Health check endpoint
- GET /product/{product_id}: Get product details
- GET /similarity: Calculate similarity between two products
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import numpy as np
from pydantic import BaseModel
import config

# Always use the enhanced multimodal version
from similarity_search_enhanced import (
    get_enhanced_similarity_search as get_similarity_search_instance,
    EnhancedProductSimilaritySearch as ProductSimilaritySearch,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Lifespan — replaces the deprecated @app.on_event("startup") pattern
# (FastAPI ≥ 0.93 emits a deprecation warning for on_event; lifespan is
# the recommended approach and works identically for our purposes).
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the multimodal similarity search engine before the first request."""
    try:
        logger.info("Initializing multimodal similarity search engine...")
        logger.info(f"Configuration: {config.get_config_summary()}")

        # Initialize enhanced multimodal engine
        get_similarity_search_instance(
            data_path=config.DATA_PATH,
            use_pca=True,
            pca_components=config.PCA_COMPONENTS,
        )

        engine = get_similarity_search_instance()
        logger.info(f"Successfully loaded {len(engine.df)} products")

        # Log multimodal features
        pca_var = engine.get_pca_explained_variance()
        pca_info = f", PCA explained variance: {pca_var:.2%}" if pca_var else ""
        image_status = "enabled" if engine.use_image_features else "disabled"
        logger.info(
            f"Multimodal features: PCA={config.PCA_COMPONENTS}D, "
            f"one-hot encoding, image features {image_status}{pca_info}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize similarity search: {e}")
        raise

    yield  # application runs here

    logger.info("Shutting down similarity search engine")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="Product Similarity Search API",
    description="Find similar products using advanced similarity search with FAISS",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class SimilarProductsResponse(BaseModel):
    query_product_id: str
    num_requested: int
    similar_products: List[str]
    count: int


class ProductDetailsResponse(BaseModel):
    product_id: str
    details: dict


class SimilarityScoreResponse(BaseModel):
    product_id1: str
    product_id2: str
    similarity_score: float


class HealthResponse(BaseModel):
    status: str
    total_products: int
    index_dimension: int
    pca_explained_variance: Optional[float] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Product Similarity Search API",
        "version": "1.0.0",
        "endpoints": {
            "/find_similar_products": "Find similar products",
            "/product/{product_id}": "Get product details",
            "/similarity": "Calculate similarity between two products",
            "/health": "Health check",
        },
    }


@app.get("/find_similar_products", response_model=SimilarProductsResponse)
async def get_similar_products(
    product_id: str = Query(..., description="The unique ID of the product"),
    num_similar: int = Query(
        10, ge=1, le=100, description="Number of similar products to return"
    ),
) -> SimilarProductsResponse:
    """
    Find similar products based on product ID.

    Results are sorted by descending cosine similarity.  Ties are broken by
    rating (desc) then sales_price (asc).

    Args:
        product_id: The unique ID of the product to find similarities for
        num_similar: Number of similar products to return (1–100)

    Returns:
        List of similar product IDs sorted by similarity

    Raises:
        HTTPException 404: product not found
        HTTPException 500: unexpected server error
    """
    try:
        logger.info(f"Finding {num_similar} similar products for {product_id}")
        engine = get_similarity_search_instance()
        similar_products = engine.find_similar_products(product_id, num_similar)
        logger.info(f"Found {len(similar_products)} similar products")
        return SimilarProductsResponse(
            query_product_id=product_id,
            num_requested=num_similar,
            similar_products=similar_products,
            count=len(similar_products),
        )
    except ValueError:
        logger.warning(f"Product not found: {product_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Product ID '{product_id}' not found in dataset",
        )
    except Exception as e:
        logger.error(f"Error finding similar products: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/product/{product_id}", response_model=ProductDetailsResponse)
async def get_product_details(product_id: str) -> ProductDetailsResponse:
    """
    Get detailed information about a product.

    Args:
        product_id: The unique ID of the product

    Returns:
        Product details including all attributes

    Raises:
        HTTPException 404: product not found
    """
    try:
        engine = get_similarity_search_instance()
        details = engine.get_product_details(product_id)

        if details is None:
            raise HTTPException(
                status_code=404,
                detail=f"Product ID '{product_id}' not found",
            )

        # Scrub NaN / Infinity before JSON serialisation
        cleaned = {}
        for key, value in details.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                cleaned[key] = None
            else:
                cleaned[key] = value

        return ProductDetailsResponse(product_id=product_id, details=cleaned)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting product details: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/similarity", response_model=SimilarityScoreResponse)
async def get_similarity_score(
    product_id1: str = Query(..., description="First product ID"),
    product_id2: str = Query(..., description="Second product ID"),
) -> SimilarityScoreResponse:
    """
    Calculate cosine similarity between two products.

    Args:
        product_id1: First product ID
        product_id2: Second product ID

    Returns:
        Similarity score (higher = more similar)

    Raises:
        HTTPException 404: either product not found
    """
    try:
        engine = get_similarity_search_instance()
        # get_similarity_score now raises ValueError for missing products,
        # so we no longer need the fragile "score == 0.0 means not found" check.
        score = engine.get_similarity_score(product_id1, product_id2)
        return SimilarityScoreResponse(
            product_id1=product_id1,
            product_id2=product_id2,
            similarity_score=score,
        )
    except ValueError as e:
        # The error message names the missing product ID
        missing_id = str(e).split("'")[1] if "'" in str(e) else "unknown"
        logger.warning(f"Product not found during similarity check: {missing_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint (suitable for Kubernetes liveness/readiness probes).

    Returns system status, product count, index dimension, and — when PCA is
    enabled — the cumulative explained variance so operators can verify how
    much signal was retained after dimensionality reduction.
    """
    try:
        engine = get_similarity_search_instance()
        pca_var = (
            engine.get_pca_explained_variance()
            if hasattr(engine, 'get_pca_explained_variance')
            else None
        )
        return HealthResponse(
            status="healthy",
            total_products=len(engine.df),
            index_dimension=engine.feature_matrix.shape[1],
            pca_explained_variance=pca_var,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# ------------------------------------------------------------------
# Error handlers
# ------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "detail": str(exc.detail)},
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
