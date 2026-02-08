# 🎓 SPU Admission Chatbot

A bilingual (English/Arabic) AI-powered chatbot for the Syrian Private University (SPU) admission inquiries. Built with a Retrieval-Augmented Generation (RAG) architecture to provide accurate, source-grounded answers about faculties, fees, rules, and admission requirements.



## ✨ Features

- **RAG Architecture**: Retrieves precise information from university documents before answering.
- **Bilingual Support**: Fully supports **Arabic** and **English** queries and interface.
- **Real-time Streaming**: Responses are streamed token-by-token for a natural conversational experience.
- **Source Citations**: Displays specific document chunks used to generate each answer with relevance scores.
- **Microservices Architecture**: Modular design using Docker containers.

## 🛠️ Tech Stack

**Frontend:**
- **React** (Vite + TypeScript)
- **Tailwind CSS** & **Shadcn UI**
- **i18n** for internationalization

**Backend & AI:**
- **Python** (FastAPI)
- **Llama 3.3 70B** (via HuggingFace Inference API) for reasoning
- **BGE-M3** for multilingual embeddings
- **Qdrant** for vector storage and retrieval

**DevOps:**
- **Docker** & **Docker Compose** for orchestration

## 🚀 Getting Started

### Prerequisites
- Docker Desktop installed
- HuggingFace API Token (with access to Llama 3 models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/OwaisHilal02/SPU-Admission-Chatbot.git
   cd SPU-Admission-Chatbot
   ```

2. **Set up Environment Variables**
   Create a `.env` file in the root directory (or set them in `docker-compose.yml`):
   ```bash
   HF_TOKEN=your_huggingface_token_here
   HUGGING_FACE_HUB_TOKEN=your_huggingface_token_here
   ```

3. **Run with Docker Compose**
   ```bash
   docker-compose up --build -d
   ```

4. **Access the Application**
   - **Frontend (Chat Interface):** [http://localhost:5173](http://localhost:5173) (or the port shown in terminal)
   - **Qdrant Dashboard:** [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

## 📂 Project Structure

```
├── Data/                   # Source documents (PDFs, etc.)
├── Services/
│   ├── Data_Loader/        # Service to ingest documents into Qdrant
│   ├── Data_Splitting/     # Text chunking logic
│   ├── Embedding_Store/    # Embedding generation service
│   ├── QA_Chatting/        # Main RAG & LLM interaction service
│   ├── RAG_Service/        # Retrieval logic
│   ├── spu-ai-connect-main/ # React Frontend Application
│   └── Streamlit_Interface/ # (Deprecated) Old prototype interface
├── docker-compose.yml      # Container orchestration
└── ...
```

## 🔄 Pipeline Overview

1. **Ingestion**: Documents in `Data/` are processed, chunked, and embedded using BGE-M3.
2. **Storage**: Embeddings are stored in Qdrant Vector DB.
3. **Retrieval**: User queries are embedded and matched against stored chunks (Hybrid Search).
4. **Generation**: Top chunks are sent to Llama 3.3, which generates a grounded response.
5. **Streaming**: Response is streamed to the React frontend via Server-Sent Events (SSE).
