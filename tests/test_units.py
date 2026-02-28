import pytest
import os
from src.embeddings.model_loader import MultimodalEmbedder
from src.ingestion.document_parser import PDFParser

def test_embedder_initialization():
    """Unit test: Check if CLIP embedder loads."""
    embedder = MultimodalEmbedder()
    assert embedder.model is not None
    print("[+] Embedder unit test passed.")

def test_text_embedding_shape():
    """Unit test: Check text embedding dimensions (should be 512)."""
    embedder = MultimodalEmbedder()
    emb = embedder.encode_text("Test research paper")
    assert len(emb) == 512
    print("[+] Text embedding shape test passed.")

def test_pdf_parser_exists():
    """Unit test: Verify PDF parser can be initialized."""
    parser = PDFParser()
    assert parser is not None
    print("[+] Parser unit test passed.")
