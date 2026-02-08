from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import time
import httpx

app = FastAPI(title="RAG Service - Document Retrieval")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "arabic_university_docs"
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
EMBEDDING_STORE_URL = "http://embedding-store:5003"
MAX_RETRIES = 5
RETRY_DELAY = 2

# Global variables
qdrant_client = None
EMBEDDING_DIM = 1024  # BGE-M3 dimension


class QueryRequest(BaseModel):
    query: str
    k: int = 8
    min_score: float = 0.3
    faculty: Optional[str] = None
    doc_category: Optional[str] = None
    year: Optional[str] = None
    semester: Optional[str] = None


def init_qdrant_client(max_retries=MAX_RETRIES):
    """Initialize Qdrant client with retry logic."""
    for attempt in range(max_retries):
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            client.get_collections()
            logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
            return client
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise


def get_query_embedding(text: str) -> List[float]:
    """Generate query embedding by calling Embedding-Store service."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{EMBEDDING_STORE_URL}/embed",
                json={"texts": [text]}
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def retrieve_documents(
    query_embedding: List[float], 
    k: int, 
    min_score: float
) -> List[Dict]:
    """Vector similarity search."""
    try:
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=k,
            score_threshold=min_score
        )
        
        results = []
        for hit in search_results:
            results.append({
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {}),
                "score": float(hit.score),
                "chunk_id": hit.payload.get("chunk_id")
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise


@app.get("/health")
def health_check():
    """Health check endpoint."""
    qdrant_status = "connected" if qdrant_client else "disconnected"
    
    # Check if Embedding-Store service is reachable
    embedding_service_status = "unknown"
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{EMBEDDING_STORE_URL}/health")
            if response.status_code == 200:
                embedding_service_status = "connected"
            else:
                embedding_service_status = "error"
    except:
        embedding_service_status = "disconnected"
    
    return {
        "status": "ok",
        "message": "RAG Service is running!",
        "qdrant": qdrant_status,
        "embedding_service": embedding_service_status,
        "model": "BAAI/bge-m3 (via Embedding-Store)",
        "collection": COLLECTION_NAME
    }


@app.post("/retrieve")
def retrieve(request: QueryRequest):
    """
    Main retrieval endpoint.
    Delegates to Embedding-Store's Hybrid Search.
    """
    try:
        logger.info(f"📥 Query: {request.query[:50]}...")
        
        # Call Embedding-Store Search Endpoint
        params = {
            "query": request.query,
            "limit": request.k,
            "min_score": request.min_score
        }
        
        # Add filters if provided
        if request.faculty:
            params["faculty"] = request.faculty
        if request.doc_category:
            params["doc_category"] = request.doc_category
            
        # Add hierarchical filters
        for key in ["year", "semester"]:
            if hasattr(request, key) and getattr(request, key):
                params[key] = getattr(request, key)
            elif key in request.__dict__ and request.__dict__[key]:
                 params[key] = request.__dict__[key]
                 
        # Actually, let's just use a more robust way to capture dynamic filters
        params.update({k: v for k, v in request.dict().items() if v and k not in ["query", "k", "min_score"]})
        
        response = httpx.get(
            f"{EMBEDDING_STORE_URL}/search",
            params=params,
            timeout=30.0
        )
        
        if response.status_code != 200:
            logger.error(f"Embedding-Store returned {response.status_code}: {response.text}")
            raise HTTPException(status_code=500, detail="Search service failed")
            
        data = response.json()
        
        if not data.get("success"):
            raise HTTPException(status_code=500, detail=data.get("error", "Unknown error"))
            
        results = data.get("results", [])
        logger.info(f"✅ Retrieved {len(results)} results via Hybrid Search")
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/search-quality")
def search_quality(query: str, limit: int = 10):
    """Analyze search quality for a query."""
    try:
        query_embedding = get_query_embedding(query)
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=0.0
        )
        
        scores = [float(r.score) for r in results]
        
        return {
            "success": True,
            "query": query,
            "total_results": len(results),
            "score_stats": {
                "max": max(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
                "avg": sum(scores) / len(scores) if scores else 0.0,
                "high_relevance": len([s for s in scores if s >= 0.7]),
                "medium_relevance": len([s for s in scores if 0.3 <= s < 0.7]),
                "low_relevance": len([s for s in scores if s < 0.3])
            }
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/collection-stats")
def collection_stats():
    """Get collection statistics."""
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        
        try:
            count = qdrant_client.count(
                collection_name=COLLECTION_NAME,
                exact=True
            )
            point_count = count.count
        except:
            point_count = 0
        
        return {
            "success": True,
            "collection_name": COLLECTION_NAME,
            "vector_dimension": info.config.params.vectors.size,
            "total_documents": point_count,
            "distance_metric": str(info.config.params.vectors.distance)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global qdrant_client
    
    logger.info("🚀 RAG Service starting up...")
    logger.info("📡 Using Embedding-Store service for embeddings")
    
    # Initialize Qdrant client
    try:
        qdrant_client = init_qdrant_client()
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        raise
    
    # Verify Embedding-Store service is reachable
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{EMBEDDING_STORE_URL}/health")
            if response.status_code == 200:
                logger.info(f"✅ Embedding-Store service connected")
            else:
                logger.warning(f"⚠️ Embedding-Store returned status {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Could not reach Embedding-Store: {e}")
    
    logger.info("✅ RAG Service initialized successfully!")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004)