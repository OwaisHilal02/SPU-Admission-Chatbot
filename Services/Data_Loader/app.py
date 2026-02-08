from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import requests
from datetime import datetime

app = FastAPI(title="Data Loader Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to Markdown data inside Docker container
DATA_PATH = Path("/app/data")

class DocumentModel(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}
    chunk_id: Optional[int] = None

class LoadRequest(BaseModel):
    collection_name: str = "arabic_university_docs"

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extract faculty and document category from filename."""
    metadata = {}
    
    # Faculty mapping (Arabic to English)
    faculty_map = {
        "الصيدلة": "Pharmacy",
        "الطب البشري": "Medicine",
        "طب الأسنان": "Dentistry",
        "العلوم الادارية": "Business",
        "هندسة الذكاء الاصطناعي": "AI Engineering",
        "هندسة البترول": "Petroleum Engineering",
        "هندسة تكنولوجيا البناء و التشييد": "Construction Engineering",
    }
    
    # Document category mapping
    category_map = {
        "الخطة الدراسية": "curriculum",
        "توصيف المقررات": "courses_descriptions", 
        "الرسوم": "fees",
        "معدلات": "admission",
        "رؤية": "faculty_info",
        "كلية": "faculty_info",
        "القرار": "regulation",
        "معلومات التواصل": "uni_info",      
        "معلومات الجامعة": "uni_info",      
        "متطلبات الجامعة": "req_courses"    
    }
    
    # Extract faculty
    for ar_name, en_name in faculty_map.items():
        if ar_name in filename:
            metadata["faculty"] = en_name
            metadata["faculty_ar"] = ar_name
            break
    
    # Extract category
    for ar_keyword, category in category_map.items():
        if ar_keyword in filename:
            metadata["doc_category"] = category
            break
    
    # Default to general if no faculty found
    if "faculty" not in metadata:
        metadata["doc_category"] = metadata.get("doc_category", "general")
    
    return metadata


def load_md_file(file_path: Path) -> str:
    """Load and parse a Markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Successfully loaded Markdown file: {file_path.name}")
        return content
    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        raise

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Data Loader service is running!",
        "data_path": str(DATA_PATH),
        "data_path_exists": DATA_PATH.exists()
    }

@app.get("/scan")
def scan_data_folder():
    """List all Markdown files in the Data folder."""
    if not DATA_PATH.exists() or not DATA_PATH.is_dir():
        return {
            "success": False,
            "message": f"Data folder not found at {DATA_PATH}!"
        }
    
    files = []
    for file in sorted(DATA_PATH.iterdir()):
        if file.is_file() and file.suffix.lower() == '.md':
            # This line MUST be inside the loop:
            metadata = extract_metadata_from_filename(file.name)
            
            files.append({
                "name": file.name,
                "size_bytes": file.stat().st_size,
                "type": "markdown",
                "metadata": metadata 
            })
    
    return {
        "success": True,
        "total_files": len(files),
        "files": files
    }

@app.get("/load")
def load_documents():
    """Load all Markdown files and return summary statistics."""
    if not DATA_PATH.exists() or not DATA_PATH.is_dir():
        return {
            "success": False,
            "message": f"Data folder not found at {DATA_PATH}!"
        }
    
    all_docs = []
    for file_path in sorted(DATA_PATH.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() != '.md':
            continue
        
        try:
            # Load Markdown content
            md_content = load_md_file(file_path)
            
            # Create document with enriched metadata
            extracted_metadata = extract_metadata_from_filename(file_path.name)
            doc = {
                "page_content": md_content,
                "metadata": {
                    "source": file_path.name,
                    "document_type": "university_data",
                    "format": "markdown",
                    "file_size": file_path.stat().st_size,
                    **extracted_metadata  # Add faculty and doc_category
                }
            }
            all_docs.append(doc)
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            continue
    
    if not all_docs:
        return {
            "success": False,
            "message": "No Markdown documents found or loaded"
        }
    
    preview = all_docs[0]["page_content"][:300] if all_docs else "No data found"
    
    return {
        "success": True,
        "total_documents": len(all_docs),
        "preview_first_document": preview,
        "loaded_files": [
            {
                "file": doc["metadata"]["source"], 
                "assigned_metadata": {
                    "faculty": doc["metadata"].get("faculty"),
                    "category": doc["metadata"].get("doc_category")
                }
            } 
            for doc in all_docs
        ],
        "stats": {
            "avg_char_length": sum(len(doc["page_content"]) for doc in all_docs) / len(all_docs),
            "total_size_bytes": sum(doc["metadata"]["file_size"] for doc in all_docs)
        }
    }

@app.get("/get-all-documents")
def get_all_documents():
    """Return all loaded documents for the splitting service."""
    if not DATA_PATH.exists() or not DATA_PATH.is_dir():
        return {
            "success": False,
            "message": f"Data folder not found at {DATA_PATH}!"
        }
    
    all_docs = []
    for file_path in sorted(DATA_PATH.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() != '.md':
            continue
        
        try:
            # Load Markdown content
            md_content = load_md_file(file_path)
            
            # Add enriched metadata
            extracted_metadata = extract_metadata_from_filename(file_path.name)
            all_docs.append({
                "page_content": md_content,
                "metadata": {
                    "source": file_path.name,
                    "document_type": "university_data",
                    "format": "markdown",
                    "file_size": file_path.stat().st_size,
                    **extracted_metadata  # Add faculty and doc_category
                }
            })
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            continue
    
    return {
        "success": True,
        "documents": all_docs,
        "total_documents": len(all_docs)
    }

@app.get("/auto-pipeline")
def auto_pipeline():
    """Complete automated pipeline: Load → Split → Embed → Store."""
    try:
        logger.info("Step 1: Loading Markdown documents...")
        all_docs_response = get_all_documents()
        
        if not all_docs_response["success"]:
            return {"success": False, "message": "Failed to load documents"}
        
        documents = all_docs_response["documents"]
        logger.info(f"Loaded {len(documents)} Markdown documents")
        
        # Step 2: Split documents
        logger.info("Step 2: Sending to splitting service...")
        split_response = requests.post(
            "http://data-splitting:5002/split",
            json=documents,
            timeout=120
        )
        
        if split_response.status_code != 200:
            return {
                "success": False,
                "error": f"Splitting failed: {split_response.text}"
            }
        
        split_result = split_response.json()
        if not split_result.get("success"):
            return {
                "success": False,
                "error": f"Splitting service error: {split_result}"
            }
        
        split_docs = split_result["chunks"]
        logger.info(f"Split into {len(split_docs)} chunks")
        
        # Step 3: Embed and store
        logger.info("Step 3: Sending to embedding service...")
        embed_response = requests.post(
            "http://embedding-store:5003/embed-and-store",
            json={"documents": split_docs},
            timeout=1200
        )
        
        if embed_response.status_code != 200:
            return {
                "success": False,
                "error": f"Embedding failed: {embed_response.text}"
            }
        
        embed_result = embed_response.json()
        if not embed_result.get("success"):
            return {
                "success": False,
                "error": f"Embedding service error: {embed_result}"
            }
        
        logger.info("Pipeline completed successfully!")
        return {
            "success": True,
            "message": "Complete pipeline executed successfully",
            "original_documents": len(documents),
            "split_chunks": len(split_docs),
            "stored_in_vector_db": embed_result.get("stored_count", len(split_docs)),
            "steps_completed": ["loading_markdown", "splitting", "embedding", "storage"]
        }
    
    except requests.exceptions.Timeout as e:
        return {"success": False, "error": f"Request timeout: {str(e)}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request error: {str(e)}"}
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/stats")
async def get_loading_stats():
    """Get system statistics."""
    try:
        # Get local file stats
        files = [f for f in DATA_PATH.glob("*.md") if f.is_file()]
        total_documents = len(files)
        
        # Get dynamic last update time
        last_update = datetime.now()
        if files:
            timestamps = []
            for f_path in files:
                try:
                    timestamps.append(f_path.stat().st_mtime)
                except Exception:
                    pass # Ignore files that can't be stat-ed
            if timestamps:
                last_update = datetime.fromtimestamp(max(timestamps))
        
        # Format date
        formatted_date = last_update.strftime("%Y-%m-%d %H:%M")

        # Get Vector DB Stats from Embedding Store
        vector_db_docs = 0
        debug_error = None
        
        try:
            # Short timeout to avoid hanging if service is down
            response = requests.get(
                "http://embedding-store:5003/collection-info",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    vector_db_docs = data.get("point_count", 0)
                else:
                    debug_error = f"API Error: {data.get('error')}"
            else:
                debug_error = f"HTTP {response.status_code}"
        except Exception as e:
            debug_error = f"Connection Failed: {str(e)}"
            logger.error(f"Failed to connect to embedding service: {e}")

        # Hardcoded from Data_Splitting service config
        AVG_CHUNK_SIZE = 1200

        # Total chunks is same as vector db docs (1 vector per chunk)
        total_chunks = vector_db_docs

        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "vector_db_docs": vector_db_docs,
            "last_update": formatted_date,
            "avg_chunk_size": AVG_CHUNK_SIZE,
            "debug_error": debug_error
        }
    except Exception as e:
        logger.error(f"Error generating stats: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "vector_db_docs": 0,
            "last_update": "Error",
            "avg_chunk_size": 1200,
            "error": str(e)
        }

@app.on_event("startup")
def startup_event():
    """Startup event handler."""
    logger.info("Data Loader Service starting up...")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Data path exists: {DATA_PATH.exists()}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)