# Project Plan: Multimodal RAG System

This document outlines the detailed requirements, technology stack, and workflow for the Multimodal RAG System.

## 1. Real Requirements Analysis
The objective is to build a system that can reason over text, tables, and images extracted from complex documents (PDFs, PNG, JPEG, TXT). 

**Key Capabilities:**
- **Layout Awareness**: Handle multi-column PDFs and slides.
- **Table Extraction**: Detect and preserve table structures.
- **Visual Grounding**: The Vision-Language Model (VLM) must "see" the raw images, not just OCR text, to answer questions (e.g., "What is the trend in this chart?").
- **Cross-Modal Retrieval**: Use common embeddings so text queries can find relevant images.
- **API Access**: A RESTful interface for querying and source attribution.

## 2. Technology Stack (Free & Open Source)

To fulfill the "Free & Open Source" requirement with "Max Achievable Quality", we will use the following:

- **Language**: Python 3.10+
- **OCR Engine**: **EasyOCR** or **PyTesseract**. (EasyOCR is preferred for ease of use and good support for different languages/fonts).
- **Document Parsing**: 
    - **PyMuPDF (fitz)**: For high-speed text and image extraction from PDFs.
    - **Unstructured**: For advanced layout analysis and table partitioning.
- **Multimodal Embeddings**: **CLIP (Contrastive Language-Image Pre-training)** via `sentence-transformers`.
    - Model: `clip-ViT-B-32`.
    - This allows text and images to be mapped to the same 512-dimensional vector space.
- **Vector Database**: **ChromaDB**.
    - Purpose: Store vector embeddings and rich metadata (page numbers, doc IDs, content types).
- **Vision-Language Model (VLM)**: **LLaVA v1.5** via **Ollama**.
    - LLaVA is state-of-the-art open source for multimodal reasoning.
    - Ollama provides a simple local API to interact with it.
- **Web Framework**: **FastAPI**.
    - Purpose: High-performance REST API.

## 3. Workflow & System Architecture

### Phase 1: Ingestion Pipeline
1.  **Iterative Extraction**: Scan `sample_documents/` for PDF/Images.
2.  **PDF Parsing**: 
    - Extract text blocks.
    - Identify and extract images as separate files.
    - Use `unstructured` to detect tables and convert them to text-based summaries/JSON for indexing.
3.  **OCR**: Run OCR on extracted images and standalone image files to get "backup" text context.
4.  **Metadata Tagging**: Every chunk is tagged:
    ```json
    {
      "doc_id": "report.pdf",
      "page": 4,
      "type": "image",
      "image_path": "data/extracted/img_1.png",
      "text_content": "[OCR result or description]"
    }
    ```

### Phase 2: Indexing
1.  **Embedding Generation**:
    - **Text/Tables**: Encode using CLIP text encoder.
    - **Images**: Encode using CLIP image encoder.
2.  **Vector Storage**: Push embeddings + metadata to ChromaDB.

### Phase 3: Retrieval
1.  **Query Encoding**: Encode user text query using CLIP text encoder.
2.  **Similarity Search**: Find top-k most similar entries in ChromaDB (could be text, tables, or images).
3.  **Context Construction**: Gather the raw text, table data, and *file paths* to images.

### Phase 4: Generation
1.  **Multimodal Prompting**: Pass the text context and the retrieved images (as base64) to the LLaVA model.
2.  **Visual Grounding**: Instruct the model to reference source material by document and page number.
3.  **Response**: Output the generated text and the list of sources used.

## 4. Proposed Directory Structure
```text
.
├── src/
│   ├── api/             # FastAPI App
│   ├── ingestion/       # PDF/Image parsing logic
│   ├── embeddings/      # CLIP model management
│   ├── retrieval/       # Vector search & ranking
│   ├── generation/      # LLaVA interaction via Ollama
│   └── vector_store/    # ChromaDB management
├── data/
│   ├── raw/             # Initial documents
│   ├── processed/       # Extracted images/tables
│   └── chroma/          # Vector DB storage
├── sample_documents/    # 10 test files
├── plan.md              # This file
├── requirements.txt     # Python dependencies
└── ... (other docs)
```

## 5. Implementation Phases (Total 8)

### Phase 1: Environment & Project Scaffolding
- Initialize Python environment and `requirements.txt`.
- Set up directory structure for source code, data, and tests.
- Configure environment variables (`.env`).

### Phase 2: PDF Parsing & Asset Extraction
- Implement logic to iterate through PDFs.
- Extract high-quality text, tables, and embedded images.
- Use `unstructured` for layout-aware partitioning.

### Phase 3: OCR & Standalone Image Ingestion
- Implement OCR pipeline using `EasyOCR`.
- Process standalone PNG/JPEG images.
- Map extracted visual data back to source document and page metadata.

### Phase 4: Multimodal Embedding Strategy
- Load CLIP (`contrastive-language-image-pretraining`) models.
- Implement specialized encoding for text, table summaries, and raw images.
- Ensure all modalities share the same semantic vector space.

### Phase 5: Vector Storage & Persistence
- Initialize **ChromaDB** with persistence.
- Design metadata schema for traceability (doc_id, page, content_type, file_path).
- Implement indexing logic for efficient bulk ingestion.

### Phase 6: Cross-Modal Retrieval & Ranking Engine
- Build the core retriever to handle text-to-multimodal queries.
- Implement results fusion (combining text and image scores).
- Ensure references to physical image files are preserved for the generation stage.

### Phase 7: Local VLM Integration & Visual Grounding
- Integrate **LLaVA v1.5** via Ollama API.
- Design visual-grounding prompts that require the model to use retrieved images.
- Implement response formatting that cites specific page numbers and visual elements.

### Phase 8: REST API, Evaluation & Documentation
- Develop **FastAPI** endpoints (`/query`, `/ingest`).
- Create an evaluation notebook to measure retrieval performance (Hit Rate, MRR).
- Produce `ARCHITECTURE.md` and complete `README.md`.
