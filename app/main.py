import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from util.recom import Recommender
import uvicorn
from app.loader import settings as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize recommender
try:
    recommender = Recommender(
        model_name=config['model']['name'],
        qdrant_url=config['qdrant']['url'],
        qdrant_api_key=config['qdrant']['api_key'],
        collection_name=config['qdrant'].get('collection_name', 'products')
    )
except (ValueError, ConnectionError, RuntimeError) as e:
    logger.error(f"Failed to initialize Recommender: {e}")
    recommender = None

app = FastAPI(title=config['app']['title'])

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if recommender else "degraded",
        "recommender_ready": recommender is not None
    }


@app.get("/")
async def recommend_endpoint(
    product_name: str = Query(None, description="Name of the product to get recommendations for"),
    q: str = Query(None, description="Alias for product_name (deprecated)")
):
    if recommender is None:
        raise HTTPException(status_code=503, detail={
            "message": "Recommender service is not available. Check server logs.",
            "success": False
        })

    search_query = product_name or q
    if not search_query:
        raise HTTPException(status_code=400, detail={
            "message": "Query parameter 'product_name' or 'q' is required.",
            "success": False
        })

    try:
        data = recommender.recommend(search_query)
        return {
            "data": data,
            "success": True
        }
    except Exception as e:
        logger.error(f"Recommendation error for query '{search_query}': {e}")
        raise HTTPException(status_code=500, detail={
            "message": f"Internal error: {str(e)}",
            "success": False
        })


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config['app']['host'],
        port=config['app']['port'],
        reload=config['app']['reload']
    )
