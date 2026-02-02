from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from recom import recommend
import uvicorn

app = FastAPI(title="SIH_ecom Recommender API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index(q: str = Query(None)):
    try:
        if not q:
            return {
                "message": "Query parameter 'q' is required",
                "success": False
            }
            
        data = recommend(q)
        
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
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)