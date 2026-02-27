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
    # Test cases
    tests = [
        ("Financial Trends", {"query": "What are the revenue trends for TCS in 2025?", "n_results": 3}),
        ("Sustainability", {"query": "What is the TCS carbon footprint strategy?", "n_results": 3}),
        ("Image Analysis", {"query": "Describe the tables or charts found in the Fact Sheets.", "n_results": 3}),
        ("Ambiguous/Edge Case", {"query": "!!!", "n_results": 1}),
        ("Empty Query", {"query": "", "n_results": 1}),
    ]

    print("=== STARTING MANUAL VERIFICATION ===\n")
    for name, payload in tests:
        run_test(name, payload)
    print("\n=== VERIFICATION COMPLETE ===")
