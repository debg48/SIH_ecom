from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from util.recom import Recommender
import uvicorn
import yaml
import os

# Load config
def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize recommender
recommender = Recommender(
    model_name=config['model']['name'],
    emb_path=config['data']['embeddings'],
    prod_path=config['data']['products']
)

app = FastAPI(title=config['app']['title'])

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def recommend_endpoint(product_name: str = Query(None, description="Name of the product to get recommendations for"), q: str = Query(None, description="Alias for product_name (deprecated)")):
    try:
        search_query = product_name or q
        if not search_query:
            return {
                "message": "Query parameter 'product_name' or 'q' is required",
                "success": False
            }
        
        data = recommender.recommend(search_query)
        
        return {
            "data": data,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "message": f"Error: {str(e)}",
            "success": False
        })

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=config['app']['host'], 
        port=config['app']['port'], 
        reload=config['app']['reload']
    )
