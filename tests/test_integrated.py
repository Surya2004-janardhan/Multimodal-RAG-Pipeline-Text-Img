import pytest
import requests
import json
import os

API_URL = "http://localhost:8000"

def test_health_check():
    """Verify API is alive."""
    try:
        response = requests.get(f"{API_URL}/")
        assert response.status_code == 200
        assert "Ready" in response.json()["status"]
    except requests.exceptions.ConnectionError:
        pytest.fail("FastAPI server must be running (uvicorn src.api.main:app) for integration tests.")

def test_ingestion_endpoint():
    """Verify ingestion can be triggered."""
    response = requests.post(f"{API_URL}/ingest")
    assert response.status_code == 200
    assert "success" in response.json()["status"]

def test_query_multimodal_transformer():
    """End-to-end: query a specific research paper known to be in DB."""
    payload = {
        "query": "Explain the Transformer architecture.",
        "n_results": 3
    }
    response = requests.post(f"{API_URL}/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    # Check if Transformer paper is in sources
    found = any("Attention" in s["document_id"] for s in data["sources"])
    assert found, "Transformer paper not found in retrieved sources."

def test_off_topic_guardrail():
    """Verify guardrails work for unrelated queries."""
    payload = {
        "query": "Give me a recipe for chocolate cake.",
        "n_results": 1
    }
    response = requests.post(f"{API_URL}/query", json=payload)
    assert response.status_code == 200
    assert "expertise is currently limited" in response.json()["answer"]
