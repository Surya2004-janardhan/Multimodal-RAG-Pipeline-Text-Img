import pytest
import requests
import json

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
        pytest.fail("API server not detected at http://localhost:8000. Ensure it's running for tests.")

def test_e2e_query_processing():
    """Requirement check: End-to-end multimodal query processing."""
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
    
    # Check for presence of source metadata as required by Step 7
    source = data["sources"][0]
    assert "document_id" in source
    assert "page_number" in source
    assert "content_type" in source
