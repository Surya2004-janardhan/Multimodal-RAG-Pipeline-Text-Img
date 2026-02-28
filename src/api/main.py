import os
import glob
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

from src.ingestion.document_parser import PDFParser
from src.ingestion.image_processor import ImageProcessor
from src.embeddings.model_loader import MultimodalEmbedder, LangChainCLIPEmbeddings
from src.vector_store.chroma_manager import ChromaManager
from src.retrieval.retriever import MultimodalRetriever
from src.generation.generator import MultimodalGenerator

# Load environment variables
load_dotenv()

app = FastAPI(title="Multimodal RAG API (LangChain + Gemini)", version="1.1.0")

# --- Initialize Project Components ---
# Using LangChain-compatible embedding wrapper
clip_lc = LangChainCLIPEmbeddings()
embedder = clip_lc.embedder # Keep access to the raw embedder for internal tasks

# Initialize Vector Store with LangChain wrapper
vector_store = ChromaManager(embedding_function=clip_lc)

retriever = MultimodalRetriever(embedder, vector_store)
generator = MultimodalGenerator()
pdf_parser = PDFParser()
image_processor = ImageProcessor()

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5

class Source(BaseModel):
    document_id: str
    page_number: int
    content_type: str
    snippet: Optional[str] = None
    image_path: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

# --- Helper Functions ---
def process_single_file(file_path: str):
    """
    Orchestrates the ingestion, embedding, and indexing of a single file.
    """
    ext = os.path.splitext(file_path)[1].lower()
    chunks = []
    
    if ext == ".pdf":
        chunks = pdf_parser.extract_content(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        processed = image_processor.process_image(file_path)
        if processed:
            chunks = [processed]
    elif ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                chunks = [{
                    "doc_id": os.path.basename(file_path),
                    "page": 1,
                    "type": "text",
                    "content": content,
                    "metadata": {
                        "source": file_path,
                        "page_number": 1,
                        "content_type": "text"
                    }
                }]
        except Exception as e:
            print(f"[!] Error reading txt file {file_path}: {e}")
    
    if not chunks:
        return

    # Collect lists for batch addition
    all_ids = []
    all_embeddings = []
    all_metadatas = []
    all_documents = []

    print(f"[*] Encoding {len(chunks)} chunks for {os.path.basename(file_path)}...", flush=True)
    
    # Generate embeddings
    for i, chunk in enumerate(chunks):
        # Stable ID using file hash and index
        chunk_id = f"{chunk['doc_id']}_{i}_{chunk['type']}_{chunk['page']}"
        
        # Determine content for embedding
        if chunk["type"] == "image":
            embedding = embedder.encode_image(chunk["content"]).tolist()
            doc_text = chunk.get("ocr_text", f"Image from {chunk['doc_id']} page {chunk['page']}")
        else:
            embedding = embedder.encode_text(chunk["content"]).tolist()
            doc_text = chunk["content"]

        all_ids.append(chunk_id)
        all_embeddings.append(embedding)
        all_metadatas.append(chunk["metadata"])
        all_documents.append(doc_text)
        
        if (i + 1) % 50 == 0:
            print(f"  - Encoded {i + 1}/{len(chunks)} chunks...", flush=True)

    # Final batch push to Chroma in smaller chunks of 50 to avoid timeouts/OOM
    if all_ids:
        batch_size = 50
        print(f"[*] Pushing {len(all_ids)} items to ChromaDB for {os.path.basename(file_path)}...", flush=True)
        for j in range(0, len(all_ids), batch_size):
            end = min(j + batch_size, len(all_ids))
            vector_store.add_embeddings(
                ids=all_ids[j:end],
                embeddings=all_embeddings[j:end],
                metadatas=all_metadatas[j:end],
                documents=all_documents[j:end]
            )
        print(f"[+] Finished indexing {os.path.basename(file_path)}", flush=True)
    else:
        print(f"[!] No content found to index for {os.path.basename(file_path)}", flush=True)

# --- Endpoints ---

@app.get("/status")
def get_status():
    return {
        "status": "Ready",
        "document_count": vector_store.get_count(),
        "collection_name": "multimodal_rag"
    }

@app.get("/")
def read_root():
    return {"message": "Multimodal RAG System is running.", "status": "Ready"}

@app.post("/ingest")
async def ingest_documents(background_tasks: BackgroundTasks):
    """
    Triggers ingestion of all documents in the sample_documents folder.
    This runs as a background task.
    """
    raw_path = os.getenv("RAW_DATA_PATH", "./sample_documents")
    files = glob.glob(os.path.join(raw_path, "*.*"))
    
    valid_files = [f for f in files if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.txt'))]
    
    if not valid_files:
        return {"status": "error", "message": f"No valid documents found in {raw_path}"}

    for file in valid_files:
        background_tasks.add_task(process_single_file, file)
        
    return {
        "status": "success", 
        "message": f"Ingestion started for {len(valid_files)} files.",
        "files": [os.path.basename(f) for f in valid_files]
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Standard RAG endpoint: Retrieval -> Context Formatting -> Generation.
    """
    # 1. Retrieval
    relevant_items = retriever.retrieve(request.query, n_results=request.n_results)
    
    if not relevant_items:
        return QueryResponse(
            answer="No relevant context found in the database. Please ingest documents first.",
            sources=[]
        )
        
    # 2. Generation
    result = generator.generate_answer(request.query, relevant_items)
    
    # 3. Format Sources
    formatted_sources = []
    for meta in result["sources"]:
        formatted_sources.append(Source(
            document_id=meta["source"],
            page_number=meta["page_number"],
            content_type=meta["content_type"],
            image_path=meta.get("image_path")
        ))
        
    return QueryResponse(
        answer=result["answer"],
        sources=formatted_sources
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
