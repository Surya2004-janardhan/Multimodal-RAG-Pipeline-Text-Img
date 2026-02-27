import os
import glob
from dotenv import load_dotenv
from src.ingestion.document_parser import PDFParser
from src.ingestion.image_processor import ImageProcessor
from src.embeddings.model_loader import MultimodalEmbedder
from src.vector_store.chroma_manager import ChromaManager

# Load environment variables
load_dotenv()

def debug_ingest():
    print("--- STARTING FOREGROUND DEBUG INGESTION ---", flush=True)
    
    # 1. Initialize Components
    print("[*] Initializing components...", flush=True)
    embedder = MultimodalEmbedder()
    vector_store = ChromaManager()
    pdf_parser = PDFParser()
    image_processor = ImageProcessor()
    
    # 2. Find Files
    raw_path = os.getenv("RAW_DATA_PATH", "./sample_documents")
    files = glob.glob(os.path.join(raw_path, "*.*"))
    valid_files = [f for f in files if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.txt'))]
    
    if not valid_files:
        print(f"[!] No valid documents found in {raw_path}", flush=True)
        return

    print(f"[*] Found {len(valid_files)} files to ingest.", flush=True)

    # 3. Process Each File
    for file_path in valid_files:
        filename = os.path.basename(file_path)
        print(f"\n>>> PROCESSING: {filename}", flush=True)
        
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
                        "doc_id": filename,
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
                print(f"[!] Error reading txt file {file_path}: {e}", flush=True)
        
        if not chunks:
            print(f"[!] No chunks extracted for {filename}", flush=True)
            continue

        print(f"[*] Extracted {len(chunks)} chunks. Starting encoding...", flush=True)

        all_ids = []
        all_embeddings = []
        all_metadatas = []
        all_documents = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk['doc_id']}_{i}_{chunk['type']}_{chunk['page']}"
            
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
                print(f"  - Encoded {i + 1}/{len(chunks)}...", flush=True)

        # Push to Chroma
        if all_ids:
            batch_size = 50
            for j in range(0, len(all_ids), batch_size):
                end = min(j + batch_size, len(all_ids))
                vector_store.add_embeddings(
                    ids=all_ids[j:end],
                    embeddings=all_embeddings[j:end],
                    metadatas=all_metadatas[j:end],
                    documents=all_documents[j:end]
                )
            print(f"[+] Finished indexing {filename}", flush=True)

    print("\n--- DEBUG INGESTION COMPLETE ---", flush=True)
    print(f"Final total document count: {vector_store.get_count()}", flush=True)

if __name__ == "__main__":
    debug_ingest()
