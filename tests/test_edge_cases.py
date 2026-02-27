import pytest
import requests
import json

BASE_URL = "http://localhost:8000"

def test_empty_query():
    """Test how the API handles an empty query string."""
    payload = {"query": "", "n_results": 5}
    response = requests.post(f"{BASE_URL}/query", json=payload)
    assert response.status_code == 200
    # Depending on implementation, it might return a message or generic results
    assert "answer" in response.json()

def test_large_n_results():
    """Test requesting a very large number of results."""
    payload = {"query": "TCS", "n_results": 1000}
    response = requests.post(f"{BASE_URL}/query", json=payload)
    assert response.status_code == 200
    # Should automatically cap to maximum available or a reasonable limit
    assert len(response.json()["sources"]) <= 1000

def test_missing_fields():
    """Test the API with missing required fields in the JSON payload."""
    payload = {"n_results": 5} # Missing 'query'
    response = requests.post(f"{BASE_URL}/query", json=payload)
    assert response.status_code == 422 # FastAPI validation error

def test_non_existent_route():
    """Test hitting a non-existent endpoint."""
    response = requests.get(f"{BASE_URL}/invalid_route")
    assert response.status_code == 404

def test_status_endpoint():
    """Test the verification status endpoint."""
    response = requests.get(f"{BASE_URL}/status")
    assert response.status_code == 200
    data = response.json()
    assert "document_count" in data
    assert data["status"] == "Ready"

def test_query_no_results():
    """Test with a query that is unlikely to have any context match."""
    payload = {"query": "zxywvutsrqponmlkjihgfedcba", "n_results": 1}
    response = requests.post(f"{BASE_URL}/query", json=payload)
    assert response.status_code == 200
    assert "answer" in response.json()
