import os
import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("API_KEY", "")

headers = {}
if API_KEY:
    headers["x-api-key"] = API_KEY

# Ready - with retry
import time
max_attempts = 15
for attempt in range(max_attempts):
    try:
        r = requests.get(f"{API_URL}/ready", timeout=5)
        r.raise_for_status()
        print("ready:", r.json())
        break
    except Exception as e:
        if attempt < max_attempts - 1:
            print(f"Retry {attempt + 1}/{max_attempts}: {e}")
            time.sleep(2)
        else:
            raise

# Metrics
r = requests.get(f"{API_URL}/metrics", timeout=5)
r.raise_for_status()
print("metrics: ok")

# Ask
r = requests.post(f"{API_URL}/ask", json={"query": "test"}, headers=headers, timeout=10)
r.raise_for_status()
print("ask:", r.json())
