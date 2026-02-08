from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import uuid
import time
import gc
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from FlagEmbedding import BGEM3FlagModel

app = FastAPI(title="Embedding & Vector Store Service (Hybrid)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "arabic_university_docs"

QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333

MAX_RETRIES = 5
RETRY_DELAY = 2

# Safer defaults to avoid OOM
EMBED_BATCH_SIZE = 4          # outer batching in our loop (safe for WSL/Docker)
MODEL_BATCH_SIZE = 4          # internal batch in model.encode (keep same as outer)
MODEL_MAX_LENGTH = 2048       # reduce from 8192 to prevent RAM spikes

UPSERT_BATCH_SIZE = 100       # Qdrant upsert batch size


class DocumentModel(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}
    chunk_id: Optional[int] = None


class EmbedRequest(BaseModel):
    documents: List[DocumentModel]
    collection_name: str = COLLECTION_NAME


class EmbedTextRequest(BaseModel):
    texts: List[str]


# Globals
qdrant_client: Optional[QdrantClient] = None
embedding_model: Optional[BGEM3FlagModel] = None
EMBEDDING_DIM = 1024


def init_qdrant_client(max_retries: int = MAX_RETRIES) -> QdrantClient:
    """Initialize Qdrant client with connection retry."""
    for attempt in range(max_retries):
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            client.get_collections()
            logger.info(f"Successfully connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
            return client
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to connect to Qdrant: {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Failed to connect to Qdrant after all retries")
                raise


def init_embedding_model() -> bool:
    """Initialize the BGE-M3 embedding model."""
    global embedding_model, EMBEDDING_DIM

    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        embedding_model = BGEM3FlagModel(
            EMBEDDING_MODEL_NAME,
            use_fp16=True
        )

        test_output = embedding_model.encode(
            ["test"],
            max_length=64,
            return_dense=True,
            return_sparse=True, # Enable Sparse
            return_colbert_vecs=False,
        )

        EMBEDDING_DIM = len(test_output["dense_vecs"][0])
        logger.info(f"Embedding model loaded. Dimension: {EMBEDDING_DIM}")
        return True
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return False


def ensure_collection_exists(collection_name: str) -> None:
    """
    Ensure the Qdrant collection exists with HYBRID config (Dense + Sparse).
    """
    if qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized")

    try:
        qdrant_client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' exists")
    except UnexpectedResponse:
        logger.info(f"Creating collection '{collection_name}' with Hybrid Config")
        
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            }
        )
        logger.info(f"Collection '{collection_name}' created")


def get_embeddings_batch(
    texts: List[str],
    batch_size: int = EMBED_BATCH_SIZE,
    model_batch_size: int = MODEL_BATCH_SIZE,
    max_length: int = MODEL_MAX_LENGTH,
) -> Tuple[List[List[float]], List[models.SparseVector]]:
    """
    Generate Hybrid embeddings (Dense + Sparse) in batches.
    Returns: (dense_vectors, sparse_vectors)
    """
    if not texts:
        return [], []

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")

    all_dense: List[List[float]] = []
    all_sparse: List[models.SparseVector] = []
    
    total = len(texts)
    total_batches = (total - 1) // batch_size + 1

    logger.info(f"Starting hybrid embedding: total={total}")

    for b, i in enumerate(range(0, total, batch_size), start=1):
        batch = texts[i:i + batch_size]

        output = embedding_model.encode(
            batch,
            batch_size=model_batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )

        # Dense Processing
        batch_dense = output["dense_vecs"].tolist()
        all_dense.extend(batch_dense)
        
        # Sparse Processing (Lexical Weights)
        # BGE-M3 returns a list of dicts {word_id: weight} for sparse
        lexical_weights = output["lexical_weights"]
        
        for weights in lexical_weights:
            # Convert str keys to int indices if needed, BGE-M3 uses token IDs (int)
            # Actually BGEM3FlagModel.encode returns dict of {str(token_id): weight} or {token_id: weight}
            # qdrant expects integer indices.
            # BGE-M3 library sparse output keys are strings of token IDs.
            
            indices = []
            values = []
            for k, v in weights.items():
                indices.append(int(k))
                values.append(float(v))
            
            all_sparse.append(models.SparseVector(indices=indices, values=values))

        # Explicit cleanup
        del output
        del batch_dense
        del lexical_weights
        gc.collect()

        if total > 50 and b % 5 == 0:
            logger.info(f"Progress: {len(all_dense)}/{total} processed")

    return all_dense, all_sparse


@app.get("/health")
def health_check():
    """Health check endpoint."""
    qdrant_status = "connected" if qdrant_client else "disconnected"
    model_status = "loaded" if embedding_model else "not loaded"
    return {
        "status": "ok",
        "service": "Embedding Store (Hybrid)",
        "qdrant": qdrant_status,
        "embedding_model": model_status,
    }


@app.post("/embed")
def embed_texts(request: EmbedTextRequest):
    """
    Generate embeddings without storing.
    Returns both dense and sparse for inspection.
    """
    try:
        dense, sparse = get_embeddings_batch(request.texts)
        
        # Convert sparse to serializable
        sparse_serializable = []
        for sv in sparse:
            sparse_serializable.append({
                "indices": sv.indices,
                "values": sv.values
            })

        return {
            "success": True,
            "dense_embeddings": dense,
            "sparse_embeddings": sparse_serializable,
            "count": len(dense),
        }

    except Exception as e:
        logger.error(f"Error in embed: {e}")
        return {"success": False, "error": str(e)}


@app.post("/embed-and-store")
def embed_and_store(request: EmbedRequest):
    """Embed documents and store in Qdrant (Hybrid)."""
    try:
        if qdrant_client is None:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")

        valid_docs = []
        texts = []
        for doc in request.documents:
            content = (doc.page_content or "").strip()
            if content and len(content) >= 10:
                valid_docs.append(doc)
                texts.append(content)

        if not texts:
            return {"success": False, "error": "No valid documents"}

        # Generate Hybrid Embeddings
        dense_vecs, sparse_vecs = get_embeddings_batch(texts)
        
        ensure_collection_exists(request.collection_name)

        points: List[models.PointStruct] = []
        for doc, dense, sparse in zip(valid_docs, dense_vecs, sparse_vecs):
            point_id = str(uuid.uuid4())
            payload = {
                "content": doc.page_content,
                "metadata": {
                    **(doc.metadata or {}),
                    "content_length": len(doc.page_content),
                },
                "chunk_id": doc.chunk_id,
            }
            
            # Hybrid Point
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense,
                        "sparse": sparse
                    },
                    payload=payload,
                )
            )

        # Upsert in batches
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[i:i + UPSERT_BATCH_SIZE]
            qdrant_client.upsert(
                collection_name=request.collection_name,
                points=batch
            )

        return {
            "success": True,
            "stored_count": len(points),
            "mode": "hybrid",
        }

    except Exception as e:
        logger.error(f"Error in embed_and_store: {e}")
        return {"success": False, "error": str(e)}


@app.get("/search")
def search(
    query: str, 
    limit: int = 8, 
    min_score: float = 0.3,
    faculty: Optional[str] = None,
    doc_category: Optional[str] = None,
    year: Optional[str] = None,
    semester: Optional[str] = None
):
    """
    Hybrid Search with optional metadata filtering.
    Filters by faculty and/or document category to prevent cross-faculty confusion.
    """
    try:
        if qdrant_client is None or embedding_model is None:
            raise HTTPException(status_code=500, detail="Service not ready")

        # Encode Query (Hybrid)
        output = embedding_model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        
        query_dense = output["dense_vecs"][0].tolist()
        
        # Helper for sparse query conversion
        sparse_weights = output["lexical_weights"][0]
        query_sparse = models.SparseVector(
            indices=[int(k) for k in sparse_weights.keys()],
            values=[float(v) for v in sparse_weights.values()]
        )
        
        # Build metadata filter
        filter_conditions = []
        if faculty:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.faculty",
                    match=models.MatchValue(value=faculty)
                )
            )
        if doc_category:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.doc_category",
                    match=models.MatchValue(value=doc_category)
                )
            )
            
        # Self-Querying: Map Year and Semester to headers
        # Year usually appears in header_2, Semester in header_3 based on MD structure
        if year:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.header_2",
                    match=models.MatchText(text=year)
                )
            )
        if semester:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.header_3",
                    match=models.MatchText(text=semester)
                )
            )
        
        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
        
        # Execute Hybrid Search with filters
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=limit * 2,
                    filter=query_filter,  # Apply filter to dense search
                ),
                models.Prefetch(
                    query=query_sparse,
                    using="sparse",
                    limit=limit * 2,
                    filter=query_filter,  # Apply filter to sparse search
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        ).points

        # Format results
        results = []
        for hit in search_results:
            results.append({
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {}),
                "score": float(hit.score),
                "chunk_id": hit.payload.get("chunk_id"),
            })

        return {"success": True, "query": query, "results": results, "count": len(results)}

    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        return {"success": False, "error": str(e)}


@app.get("/collections")
def get_collections():
    """List all collections."""
    try:
        if qdrant_client is None:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        collections = qdrant_client.get_collections()
        return {"success": True, "collections": [col.name for col in collections.collections]}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/collection-info")
def collection_info(collection_name: str = COLLECTION_NAME):
    """Get collection information."""
    try:
        if qdrant_client is None:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")

        info = qdrant_client.get_collection(collection_name)
        return {
            "success": True,
            "collection_name": collection_name,
            "vector_size": info.config.params.vectors["dense"].size if isinstance(info.config.params.vectors, dict) and "dense" in info.config.params.vectors else (info.config.params.vectors.size if not isinstance(info.config.params.vectors, dict) else 1024),
            "point_count": info.points_count,
            "status": str(info.status),
            "vectors_config": str(info.config.params.vectors),
            "sparse_vectors_config": str(info.config.params.sparse_vectors) 
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/collection/{collection_name}")
def delete_collection(collection_name: str):
    """Delete a collection."""
    try:
        if qdrant_client is None:
            raise HTTPException(status_code=500, detail="Qdrant client not initialized")
        qdrant_client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
        return {"success": True, "message": f"Collection '{collection_name}' deleted"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global qdrant_client
    logger.info("Starting Embedding & Vector Store service (HYBRID)...")

    if not init_embedding_model():
        raise RuntimeError("Embedding model initialization failed")

    qdrant_client = init_qdrant_client()
    logger.info("Service initialized successfully")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)