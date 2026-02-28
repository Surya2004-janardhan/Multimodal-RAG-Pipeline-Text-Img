import requests
import json

BASE_URL = "http://localhost:8000"

def run_test(name, payload):
    print(f"\n--- Running Test: {name} ---")
    try:
        response = requests.post(f"{BASE_URL}/query", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"QUESTION: {payload.get('query')}")
            print(f"ANSWER: {data['answer']}")
            print(f"SOURCES FOUND: {len(data['sources'])}")
            for i, src in enumerate(data['sources'][:3]):
                print(f"  [{i+1}] {src['document_id']} (Page {src['page_number']}) - {src['content_type']}")
        else:
            print(f"FAILED: Status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"CONNECTION ERROR: {e}")

if __name__ == "__main__":
    # Test cases for ML/AI Papers
    tests = [
        ("Transformer Logic", {"query": "Explain the architecture of the Transformer model as described in 'Attention Is All You Need'. What are the key components?", "n_results": 5}),
        ("Optimization details", {"query": "How is the Adam optimizer different from standard SGD? Reference the Kingma and Ba paper.", "n_results": 3}),
        ("Multimodal/Vision", {"query": "What are the main results or charts mentioned in the ImageNet 2014 paper?", "n_results": 5}),
        ("Dropout/Overfitting", {"query": "Explain how dropout prevents overfitting according to Srivastava et al. (2014).", "n_results": 3}),
        ("Ambiguous/Edge Case", {"query": "!!!", "n_results": 1}),
    ]

    print("=== STARTING ML PAPER VERIFICATION ===\n")
    for name, payload in tests:
        run_test(name, payload)
    print("\n=== VERIFICATION COMPLETE ===")
