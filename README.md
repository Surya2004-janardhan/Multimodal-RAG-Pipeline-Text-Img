# Multimodal RAG System

An advanced Retrieval-Augmented Generation system for reasoned synthesis of text, tables, and images from complex documents.

## Features
- **Multimodal Retrieval**: Search over text, tables, and images using a single text query.
- **Layout Awareness**: Extracts structured tables and handles multi-column documents using `Unstructured`.
- **Visual Grounding**: Generates answers that cite specific visual elements using `LLaVA` (via Ollama).
- **FastAPI Backend**: Ready-to-use REST API for ingestion and querying.

## Setup

### 1. Prerequisites
- Python 3.10+ (Recommended: 3.12.1 as set up)
- [Ollama](https://ollama.com/) installed and running.
- Pull the LLaVA model: `ollama pull llava`

### 2. Environment Setup
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies (using uv for performance)
uv pip install -r requirements.txt
```

### 3. Configuration
Rename `.env.example` to `.env` and adjust paths if necessary. Ensure `OLLAMA_BASE_URL` points to your active Ollama instance.

## Usage

### 1. Start the API
```bash
uvicorn src.api.main:app --reload
```

### 2. Ingest Documents
Place your PDFs and images in the `sample_documents/` folder, then call the ingest endpoint:
```bash
curl -X POST http://localhost:8000/ingest
```

### 3. Query the System
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the Q3 sales targets mentioned in the report?"}'
```

## Project Structure
- `src/ingestion`: PDF and Image parsing logic.
- `src/embeddings`: CLIP model for multimodal vectors.
- `src/vector_store`: ChromaDB persistence.
- `src/retrieval`: Cross-modal search strategy.
- `src/generation`: Vision-LLM integration.
- `src/api`: FastAPI endpoints.
