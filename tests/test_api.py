import json
import sys

import requests

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

url = "http://localhost:8001/api/v1/search"
payload = {
    "query": "Hệ thống điều hòa HVAC được điều khiển qua đâu?"
}

print(f"Sending POST request to {url}...")
try:
    response = requests.post(url, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    print("\nResponse JSON:")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
except requests.exceptions.RequestException as e:
    print(f"Failed to connect to API server: {e}")
    print("Please make sure the server is running by executing:")
    print("  uv run uvicorn main:app --app-dir src --host 0.0.0.0 --port 8001 --reload")
