# Multimodal RAG System

An advanced Retrieval-Augmented Generation system for reasoned synthesis of text, tables, and images from complex documents.

## Features
- **Research-Focused RAG**: Optimized for analyzing seminal ML/AI research papers.
- **Multimodal Retrieval**: Search over text, tables, and diagrams using a shared CLIP embedding space.
- **Lightning Fast Inference**: Powered by **Groq** and **Llama-4-Scout** for sub-second technical reasoning.
- **OCR-Enriched Context**: Images are searchable by their textual content, enabling deep diagram analysis.
- **FastAPI Backend**: Ready-to-use REST API for ingestion and querying.

## Setup

### 1. Prerequisites
- Python 3.10+
- **Groq API Key**: Obtain from [Groq Console](https://console.groq.com/).

### 2. Environment Setup
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies (using uv for performance)
uv pip install -r requirements.txt
```

### 3. Configuration
Rename `.env.example` to `.env` and add your `GROQ_API_KEY`. The system default model is set to `meta-llama/llama-4-scout-17b-16e-instruct`.

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
$headers = @{"Content-Type" = "application/json"}
$body = '{"query": "Explain the Transformer architecture key components.", "n_results": 5}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query -Headers $headers -Body $body
```

### 4. Run Evaluation
Measure retrieval performance (Hit Rate and MRR) using the provided notebook:
```bash
# Open and run all cells in evaluation.ipynb
```

## Project Structure
- `src/ingestion`: PDF and Image parsing logic.
- `src/embeddings`: CLIP model for multimodal vectors.
- `src/vector_store`: ChromaDB persistence.
- `src/retrieval`: Cross-modal search strategy.
- `src/generation`: Vision-LLM integration.
- `src/api`: FastAPI endpoints.
