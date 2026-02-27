import pytest
import requests
import time
import subprocess
import os

def test_api_running():
    # Attempt to query the root endpoint
    try:
        response = requests.get("http://localhost:8000/")
        assert response.status_code == 200
        assert "Multimodal RAG System" in response.json()["message"]
    except requests.exceptions.ConnectionError:
        pytest.fail("API is not running. Start it with uvicorn before running tests.")

def test_query_endpoint():
    # Basic check for query format
    payload = {"query": "Test query", "n_results": 1}
    try:
        response = requests.post("http://localhost:8000/query", json=payload)
        assert response.status_code == 200
        json_resp = response.json()
        assert "answer" in json_resp
        assert "sources" in json_resp
    except requests.exceptions.ConnectionError:
        pytest.fail("API is not running.")
