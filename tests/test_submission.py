import pytest
import requests
import time

API_URL = "http://localhost:8000"

def test_api_health():
    """Requirement check: Verify API is running and responsive."""
    try:
        response = requests.get(f"{API_URL}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Ready"
        assert "document_count" in data
    except requests.exceptions.ConnectionError:
        pytest.fail("API server not detected. Ensure 'uvicorn src.api.main:app' is running.")

def test_e2e_ingestion_and_query():
    """Requirement check: End-to-end multimodal pipeline (Ingestion -> Retrieval -> Generation)."""
    # 1. Trigger Ingestion (if not already done)
    # Check current count
    status = requests.get(f"{API_URL}/status").json()
    if status["document_count"] == 0:
        print("[*] DB is empty, triggering ingestion...")
        ingest_resp = requests.post(f"{API_URL}/ingest")
        assert ingest_resp.status_code == 200
        
        # Poll for completion (Max 60 seconds)
        for _ in range(30):
            time.sleep(2)
            current_status = requests.get(f"{API_URL}/status").json()
            if current_status["document_count"] > 0:
                print(f"[+] Ingestion started. Current count: {current_status['document_count']}")
                break
        else:
            pytest.fail("Ingestion failed to populate DB within timeout.")

    # 2. Query
    payload = {
        "query": "What is the key contribution of the Transformer paper?",
        "n_results": 3
    }
    response = requests.post(f"{API_URL}/query", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    
    # 3. Source Metadata Check (Step 7 Requirement)
    source = data["sources"][0]
    assert "document_id" in source
    assert "page_number" in source
    assert "content_type" in source
    print("[+] End-to-end query test passed.")
