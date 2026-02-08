import os
import uuid
import re
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import requests
import uvicorn
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="University Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:5004")
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
MAX_TOKENS = 1536
TEMPERATURE = 0.2
MIN_RELEVANCE_SCORE = 0.3

# Conversation memory (in-memory storage)
conversation_history = {}

# HuggingFace Client
client = None
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

if hf_token:
    try:
        client = InferenceClient(api_key=hf_token)
        logger.info(f"HuggingFace client initialized with model: {LLM_MODEL}")
    except Exception as e:
        logger.error(f"Error initializing HF client: {e}")
else:
    logger.warning("HF_TOKEN not set - chatbot will not work!")


def detect_language(text: str) -> str:
    """Simple language detection for metadata only."""
    if re.search(r'[\u0600-\u06FF]', text):
        return "arabic"
    return "english"


def get_conversation_history(conversation_id: str) -> str:
    """Get formatted conversation history."""
    if conversation_id in conversation_history:
        history = conversation_history[conversation_id][-3:]  # Last 3 exchanges
        formatted = []
        for exchange in history:
            formatted.append(f"User: {exchange['query']}\nAssistant: {exchange['answer']}")
        return "\n\n".join(formatted)
    return ""


def update_conversation_history(conversation_id: str, query: str, answer: str):
    """Update conversation history."""
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []
    
    conversation_history[conversation_id].append({"query": query, "answer": answer})
    
    # Keep only last 5 exchanges
    if len(conversation_history[conversation_id]) > 5:
        conversation_history[conversation_id] = conversation_history[conversation_id][-5:]


def expand_query_with_context(query: str, conversation_history_str: str) -> str:
    """
    Expand the current query using conversation history to resolve pronouns.
    Uses a lightweight LLM call to rewrite queries like 'this faculty' into specific terms.
    """
    if not conversation_history_str or not client:
        return query
    
    # Only expand if query contains pronouns/references
    reference_words = ['this', 'that', 'it', 'its', 'them', 'هذا', 'هذه', 'ذلك', 'تلك', 'ها', 'نفس']
    if not any(word in query.lower() for word in reference_words):
        return query
    
    try:
        expansion_prompt = {
            "role": "system",
            "content": (
                "You are a query rewriter. Given a conversation history and a user's question, "
                "rewrite the question to be self-contained by replacing pronouns (this, that, it, هذا, هذه, etc.) "
                "with the actual entity (faculty, subjest, document category) from the conversation.\n\n"
                "Examples:\n"
                "History: 'User asked about AI Engineering faculty'\n"
                "Query: 'what are the fees for this faculty?'\n"
                "Rewritten: 'what are the fees for AI Engineering faculty?'\n\n"
                "History: 'المستخدم سأل عن كلية الطب'\n"
                "Query: 'كم رسوم هذه الكلية؟'\n"
                "Rewritten: 'كم رسوم كلية الطب؟'\n\n"
                "CRITICAL: If the query is in English but asks about data likely stored in Arabic (like 'admission', 'fees', 'plans', 'courses'), "
                "APPEND the Arabic translation of the keywords to the query.\n"
                "Example: 'Admission requirements' -> 'Admission requirements (معدلات القبول)'\n"
                "Example: 'Tuition fees' -> 'Tuition fees (الرسوم الدراسية)'\n"
                "Example: 'Pharmacy plan' -> 'Pharmacy plan (الخطة الدراسية كلية الصيدلة)'\n\n"
                "Only output the rewritten query, nothing else."
            )
        }
        
        messages = [
            expansion_prompt,
            {
                "role": "user",
                "content": f"Conversation History:\n{conversation_history_str}\n\nCurrent Query: {query}\n\nRewritten Query:"
            }
        ]
        
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=200,
            temperature=0.1
        )
        
        expanded_query = completion.choices[0].message.content.strip()
        logger.info(f"🔍 Query expanded: '{query}' → '{expanded_query}'")
        return expanded_query
        
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return query


def extract_query_filters(query: str) -> Dict[str, Optional[str]]:
    """
    Extract metadata filters from a user query using a specialized university prompt.
    Categorized into curriculum, faculty_info, fees, admission, regulation, uni-info, and req-courses.
    """
    if not client:
        return {}
    
    try:
        filter_prompt = {
            "role": "system",
            "content": (
                "You are an expert metadata extractor for a Syrian University RAG system.\n"
                "Given a user query, output a JSON object with zero or more of these keys: "
                "faculty, doc_category, year, semester.\n\n"

                "### DOCUMENT CATEGORIES (doc_category):\n"
                "Map the query to EXACTLY ONE of the following seven categories:\n\n"

                "1) 'curriculum': Specific to each faculty. Includes study plans (الخطط الدراسية), "
                "courses per year, and semester breakdown.\n"
                "2) 'faculty_info': Specific to each faculty. Includes faculty vision, mission, "
                "goals, total credit hours, departments (الأقسام), and faculty-specific internal rules.\n"
                "3) 'fees': Centralized data for all faculties regarding tuition fees and payment costs (الرسوم).\n"
                "4) 'admission': Centralized data for all faculties regarding minimum admission scores (المعدلات) "
                "and high school requirements.\n"
                "5) 'regulation': General university regulations (قرار 21, قرار 41) that apply to ALL students "
                "and faculties across the university.\n"
                "6) 'uni_info': General university-level info including university vision/goals, contact info, "
                "address, location, and list of all available faculties.\n"
                "7) 'req_courses': Information about common required courses (مواد المتطلبات) that students from all faculties must take.\n\n "
                "8) 'courses_descriptions' : a description abput each course in the faculty"

                "### ENTITY KEYS:\n"
                "- 'faculty': Medicine, Pharmacy, Dentistry, AI Engineering, Petroleum Engineering, Construction Engineering, Business.\n"
                "- 'year': 'السنة الأولى', 'السنة الثانية', 'السنة الثالثة', 'السنة الرابعة', 'السنة الخامسة'.\n"
                "- 'semester': 'الفصل الأول', 'الفصل الثاني'.\n\n"

                "### CRITICAL RULES:\n"
                "1) For 'fees', 'admission', 'regulation', 'uni-info', and 'req-courses': "
                "   DO NOT include 'year' or 'semester' as these are not year-specific.\n"
                "2) For 'faculty_info' and 'curriculum': You MUST include 'faculty' if mentioned.\n"
                "3) 'Departments' (الأقسام) belong to 'faculty_info'.\n"
                "4) 'General contact info' (phone, address) belongs to 'uni-info'.\n"
                "5) Return ONLY a valid JSON object. Omit keys if they are not clearly detected."
            )
        }
        
        messages = [
            filter_prompt,
            {"role": "user", "content": f"Query: {query}\nJSON:"}
        ]
        
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=100,
            temperature=0.1
        )
        
        result_text = completion.choices[0].message.content.strip()
        
        # Robust parsing for JSON response
        filters = {}
        try:
            filters = json.loads(result_text)
        except json.JSONDecodeError:
            # Fallback regex match for JSON blocks
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                filters = json.loads(match.group(0))
        
        if filters:
            logger.info(f"🎯 Extracted Filters: {filters}")
            
        return filters
        
    except Exception as e:
        logger.error(f"Filter extraction failed for query '{query}': {e}")
        return {}

def retrieve_documents(
    query: str, 
    k: int = 8, 
    min_score: float = 0.3,
    faculty: Optional[str] = None,
    doc_category: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """Retrieve relevant documents from RAG service with optional filters."""
    try:
        rag_payload = {
            "query": query, 
            "k": k, 
            "min_score": min_score
        }
        
        # Add filters if provided
        if faculty:
            rag_payload["faculty"] = faculty
        if doc_category:
            rag_payload["doc_category"] = doc_category
        if kwargs.get("year"):
            rag_payload["year"] = kwargs.get("year")
        if kwargs.get("semester"):
            rag_payload["semester"] = kwargs.get("semester")
            
        response = requests.post(f"{RAG_SERVICE_URL}/retrieve", json=rag_payload, timeout=30)
        response.raise_for_status()
        
        docs = response.json().get("results", [])
        logger.info(f"📥 Retrieved {len(docs)} documents")
        return docs
        
    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return []


def generate_answer(query: str, documents: List[Dict], conversation_history_str: str) -> str:
    """Generate answer using HuggingFace InferenceClient with multilingual prompt."""
    if not client:
        return "Error: HuggingFace API not configured. Please set HF_TOKEN environment variable."
    
    try:
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "").strip()[:1500] 
            metadata = doc.get("metadata", {})
            
            # Extract key metadata for clarity
            faculty = metadata.get("faculty", "")
            category = metadata.get("doc_category", "")
            
            # Create informative header
            header = f"[Source {i}"
            if faculty:
                header += f" - Faculty: {faculty}"
            if category:
                header += f" - Category: {category}"
            header += "]"
            
            context_parts.append(f"{header}\n{content}")
        
        doc_context = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━\n\n".join(context_parts) if context_parts else "No relevant information found in database."
        
        # Detect user's language explicitly
        user_language = "English" if detect_language(query) == "english" else "Arabic"
        logger.info(f"🌐 User language: {user_language}")
        
        system_prompt = {
          "role": "system",
          "content": (
             "You are a university information assistant.\n\n"
        
             "### CORE RULES:\n"
             "1. Only use information from the sources below. Never guess or fabricate data.\n\n"
        
             "2. Match user's language:**\n"
             "   - English question → English answer\n"
             "   - Arabic question → Arabic answer\n\n"
        
             "3. **Present information clearly:**\n"
             "   - Use **bold for numbers, dates, and key facts\n"
             "   - Use markdown tables for comparisons (only if you have COMPLETE data for ALL items)\n"
             "   - State facts directly - NEVER say 'according to', 'based on', 'من المصدر'\n\n"
        
             "4. For calculations: Show the math clearly\n"
             "   - Example: $30/hour × 180 hours = $5,400 total**\n\n"
        
             "5. **Resolve references: Use conversation history to understand 'this', 'it', 'هذا', 'هذه'\n\n"
        
             "6. **IMPORTANT DISTINCTIONS:**\n"
             "   - Admission requirements = High school grades to ENTER (e.g., 1430/2200, 65%)\n"
             "   - Graduation requirements = Credit hours to COMPLETE (e.g., 180 hours)\n"
             "   - These are different - never mix them\n\n"
        
             "7. **If data is missing:**\n"
             "   - Arabic: عذراً، المعلومة غير متوفرة. للاستفسار: 📞 00963116990200\n"
             "   - English: Information unavailable. Contact: 📞 00963116990200\n"
            )
        }
        
        # Build messages array with explicit language instruction
        messages = [
            system_prompt,
            {
                "role": "user",
                "content": f"""RESPOND IN: {user_language.upper()}

                AVAILABLE SOURCES:
                 {doc_context}

                {f"PREVIOUS CONVERSATION:{chr(10)}{conversation_history_str}{chr(10)}" if conversation_history_str else ""}
                USER'S QUESTION:
                 {query}

                YOUR ANSWER (in {user_language}):"""
            }
        ]
        
        # HuggingFace API call
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.9
        )
        
        answer = completion.choices[0].message.content.strip()
        
        # Clean up common prefixes
        answer = re.sub(r'^(Answer:|الإجابة:|Response:)\s*', '', answer, flags=re.IGNORECASE).strip()
        
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Sorry, an error occurred while processing your question. / عذراً، حدث خطأ أثناء معالجة سؤالك."


def generate_answer_stream(query: str, documents: List[Dict], conversation_history_str: str):
    """
    Stream answer generation using HuggingFace InferenceClient.
    Yields Server-Sent Events (SSE) formatted chunks.
    """
    if not client:
        yield f"data: {json.dumps({'type': 'error', 'content': 'HuggingFace API not configured'})}\n\n"
        return
    
    try:
        # Build context from documents (same as generate_answer)
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "").strip()[:1500] 
            metadata = doc.get("metadata", {})
            
            faculty = metadata.get("faculty", "")
            category = metadata.get("doc_category", "")
            
            header = f"[Source {i}"
            if faculty:
                header += f" - Faculty: {faculty}"
            if category:
                header += f" - Category: {category}"
            header += "]"
            
            context_parts.append(f"{header}\n{content}")
        
        doc_context = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━\n\n".join(context_parts) if context_parts else "No relevant information found in database."
        
        user_language = "English" if detect_language(query) == "english" else "Arabic"
        
        system_prompt = {
          "role": "system",
          "content": (
             "You are a university information assistant.\n\n"
        
             "### CORE RULES:\n"
             "1. Only use information from the sources below. Never guess or fabricate data.\n\n"
        
             "2. Match user's language:**\n"
             "   - English question → English answer\n"
             "   - Arabic question → Arabic answer\n\n"
        
             "3. **Present information clearly:**\n"
             "   - Use **bold for numbers, dates, and key facts\n"
             "   - Use markdown tables for comparisons (only if you have COMPLETE data for ALL items)\n"
             "   - State facts directly - NEVER say 'according to', 'based on', 'من المصدر'\n\n"
        
             "4. For calculations: Show the math clearly\n"
             "   - Example: $30/hour × 180 hours = $5,400 total**\n\n"
        
             "5. **Resolve references: Use conversation history to understand 'this', 'it', 'هذا', 'هذه'\n\n"
        
             "6. **IMPORTANT DISTINCTIONS:**\n"
             "   - Admission requirements = High school grades to ENTER (e.g., 1430/2200, 65%)\n"
             "   - Graduation requirements = Credit hours to COMPLETE (e.g., 180 hours)\n"
             "   - These are different - never mix them\n\n"
        
             "7. **If data is missing:**\n"
             "   - Arabic: عذراً، المعلومة غير متوفرة. للاستفسار: 📞 00963116990200\n"
             "   - English: Information unavailable. Contact: 📞 00963116990200\n"

             "8. **Greetings:**\n"
             "   - Only greet the user if they greet you first, and if he does once, dont keep greeting him.\n"
            )
        }
        
        messages = [
            system_prompt,
            {
                "role": "user",
                "content": f"""RESPOND IN: {user_language.upper()}

                AVAILABLE SOURCES:
                 {doc_context}

                {f"PREVIOUS CONVERSATION:{chr(10)}{conversation_history_str}{chr(10)}" if conversation_history_str else ""}
                USER'S QUESTION:
                 {query}

                YOUR ANSWER (in {user_language}):"""
            }
        ]
        
        # HuggingFace API call with streaming enabled
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.9,
            stream=True  # Enable streaming
        )
        
        # Yield each chunk as SSE event
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        
        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in streaming answer: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


def calculate_confidence(documents: List[Dict]) -> float:
    """Calculate confidence based on document relevance."""
    if not documents:
        return 0.0
    
    scores = [doc.get("score", 0) for doc in documents]
    avg_score = sum(scores) / len(scores)
    return min(avg_score * 1.2, 1.0)


# Request/Response Models
class ChatRequest(BaseModel):
    query: str
    k: int = 8
    min_relevance_score: float = 0.3
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    answer: str
    conversation_id: str
    sources: List[Dict]
    confidence: float
    language: str
    metadata: Dict


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "hf_client": client is not None,
        "model": LLM_MODEL,
        "temperature": TEMPERATURE
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        conversation_id = request.conversation_id or str(uuid.uuid4())
        language = detect_language(query)
        
        logger.info(f"💬 Chat request: {query[:50]}... (language: {language})")
        
        # Get conversation history
        history_str = get_conversation_history(conversation_id)
        
        # Extract metadata filters from query
        filters = extract_query_filters(query)
        
        # Expand query with context BEFORE retrieval
        expanded_query = expand_query_with_context(query, history_str)
        
        # Retrieve documents with filters
        documents = retrieve_documents(
            expanded_query,
            k=request.k,
            min_score=request.min_relevance_score,
            **filters
        )
        
        # Generate answer
        answer = generate_answer(query, documents, history_str)
        
        # Update conversation history
        update_conversation_history(conversation_id, query, answer)
        
        # Calculate confidence
        confidence = calculate_confidence(documents)
        
        # Prepare metadata
        metadata = {
            "model": LLM_MODEL,
            "documents_retrieved": len(documents),
            "conversation_length": len(conversation_history.get(conversation_id, [])),
            "temperature": TEMPERATURE,
            "expanded_query": expanded_query if expanded_query != query else None
        }
        
        return ChatResponse(
            success=True,
            answer=answer,
            conversation_id=conversation_id,
            sources=documents,
            confidence=confidence,
            language=language,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        conversation_id = request.conversation_id or str(uuid.uuid4())
        return ChatResponse(
            success=False,
            answer=f"Error: {str(e)}",
            conversation_id=conversation_id,
            sources=[],
            confidence=0.0,
            language="english",
            metadata={}
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Returns the LLM response token-by-token for real-time display.
    """
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        conversation_id = request.conversation_id or str(uuid.uuid4())
        language = detect_language(query)
        
        logger.info(f"💬 Stream request: {query[:50]}... (language: {language})")
        
        # Get conversation history
        history_str = get_conversation_history(conversation_id)
        
        # Extract metadata filters from query
        filters = extract_query_filters(query)
        
        # Expand query with context BEFORE retrieval
        expanded_query = expand_query_with_context(query, history_str)
        
        # Retrieve documents with filters
        documents = retrieve_documents(
            expanded_query,
            k=request.k,
            min_score=request.min_relevance_score,
            **filters
        )
        
        # Calculate confidence (to send at end)
        confidence = calculate_confidence(documents)
        
        # Create a generator that wraps the stream and handles metadata
        def stream_with_metadata():
            answer_parts = []
            
            # Stream the answer
            for chunk in generate_answer_stream(query, documents, history_str):
                # Parse the SSE data to capture the full answer
                if 'type": "token"' in chunk:
                    try:
                        data = json.loads(chunk.replace("data: ", "").strip())
                        if data.get("type") == "token":
                            answer_parts.append(data.get("content", ""))
                    except:
                        pass
                yield chunk
            
            # Reconstruct full answer for history
            full_answer = "".join(answer_parts)
            
            # Update conversation history with the full answer
            update_conversation_history(conversation_id, query, full_answer)
            
            # Send metadata as final event
            metadata_event = {
                "type": "metadata",
                "conversation_id": conversation_id,
                "sources": documents,
                "confidence": confidence,
                "language": language,
                "documents_retrieved": len(documents),
                "expanded_query": expanded_query if expanded_query != query else None
            }
            yield f"data: {json.dumps(metadata_event)}\n\n"
        
        return StreamingResponse(
            stream_with_metadata(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream error: {e}")
        
        def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )


@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history for a specific conversation_id."""
    if conversation_id in conversation_history:
        del conversation_history[conversation_id]
        return {"success": True, "message": f"Conversation {conversation_id} cleared"}
    return {"success": False, "message": "Conversation not found"}


@app.get("/conversations")
async def list_conversations():
    """List all active conversations."""
    return {
        "success": True,
        "conversations": [
            {
                "id": conv_id,
                "length": len(exchanges)
            }
            for conv_id, exchanges in conversation_history.items()
        ]
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    uvicorn.run(app, host="0.0.0.0", port=port)