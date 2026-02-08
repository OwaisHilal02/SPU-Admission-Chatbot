from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


app = FastAPI(title="Data Splitting Service")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Splitting Configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


# Rules for MarkdownHeaderTextSplitter
headers_to_split_on = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
]


# Initialize splitters
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False # Keep headers in the content for context
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)


class DocumentModel(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}
    chunk_id: Optional[int] = None

CHUNKS_STORAGE: List[DocumentModel] = []


def smart_split_markdown(content: str, source_metadata: Dict[str, Any], start_id: int):
    """
    Splits markdown by headers and detects faculty-specific sections 
    to enrich metadata for multi-faculty documents (like Fees/Admission).
    """
    # 1. Faculty mapping (Arabic to English)
    faculty_map = {
        "الصيدلة": "Pharmacy",
        "الطب البشري": "Medicine",
        "طب الأسنان": "Dentistry",
        "العلوم الادارية": "Business",
        "العلوم الإدارية": "Business", # Handling variant spelling
        "هندسة الذكاء الاصطناعي": "AI Engineering",
        "هندسة البترول": "Petroleum Engineering",
        "هندسة تكنولوجيا البناء و التشييد": "Construction Engineering",
    }
    # 2. Split text based on Markdown headers (#, ##, etc.)
    header_docs = header_splitter.split_text(content)
    final_chunks = []
    chunk_counter = start_id 
    
    for doc in header_docs:
        # 3. Create the initial metadata by merging file-level and header-level info
        current_metadata = {**source_metadata, **doc.metadata}
        
        # 4. HEADER FACULTY DETECTION
        # We look through the detected headers (h1, h2, h3, h4) for faculty names
        for i in range(1, 5):
            header_key = f"header_{i}"
            header_val = current_metadata.get(header_key, "")
            
            if header_val:
                for ar_name, en_name in faculty_map.items():
                    if ar_name in header_val:
                        # Success! We found a faculty header. 
                        # This chunk is now specifically tagged for this faculty.
                        current_metadata["faculty"] = en_name
                        current_metadata["faculty_ar"] = ar_name
                        break
        
        # 5. Perform final text splitting (RecursiveCharacterSplitter)
        sub_chunks = text_splitter.split_text(doc.page_content)
        
        for sub_chunk in sub_chunks:
            final_chunks.append(DocumentModel(
                page_content=sub_chunk,
                metadata=current_metadata,
                chunk_id=chunk_counter
            ))
            chunk_counter += 1
                    
    # Return both the chunks and the updated counter
    return final_chunks, chunk_counter


@app.post("/split")
def split_documents(documents: List[DocumentModel]):
    """Main splitting endpoint with global ID tracking."""
    try:
        logger.info(f"Received {len(documents)} documents for splitting")
        all_chunks = []
        
        # Track ID globally across all documents in this batch
        current_id = 0 
        
        for doc in documents:
            chunks, next_id = smart_split_markdown(doc.page_content, doc.metadata, current_id)
            all_chunks.extend(chunks)
            current_id = next_id # Update the counter for the next document
            
        # Update global storage for the /all-chunks endpoint
        CHUNKS_STORAGE.extend(all_chunks)
        
        logger.info(f"Successfully generated {len(all_chunks)} unique chunks")
        return {
            "success": True, 
            "chunks": all_chunks,
            "total_chunks": len(all_chunks)
        }
    except Exception as e:
        logger.error(f"Error during splitting: {e}")
        return {"success": False, "error": str(e)}


@app.get("/all-chunks", response_model=Dict[str, Any])
def get_all_chunks():
    """Retrieve all chunks stored in memory since the service started."""
    return {
        "total_count": len(CHUNKS_STORAGE),
        "chunks": CHUNKS_STORAGE
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Data Splitting"}


@app.get("/config")
def get_config():
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "headers_tracked": [h[1] for h in headers_to_split_on],
        "splitter_type": "MarkdownHeaderAwareHybrid"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)