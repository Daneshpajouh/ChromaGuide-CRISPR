import requests
import urllib3
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def debug_scan():
    org = "google"
    url = "https://huggingface.co/api/models"
    params = {
        "author": org,
        "sort": "likes",
        "direction": "-1",
        "limit": 5
    }

    print(f"Scanning {org}...")
    try:
        resp = requests.get(url, params=params, verify=False, timeout=10)
        print(f"Status: {resp.status_code}")
        print(f"Headers: {dict(resp.headers)}")
        try:
            data = resp.json()
            print(f"Data Type: {type(data)}")
            print(f"Count: {len(data)}")
            print("First item sample:")
            if data:
                print(json.dumps(data[0], indent=2))
            else:
                print("Data is empty list []")
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            print(f"Raw Text: {resp.text[:500]}")

    except Exception as e:
        print(f"Request Error: {e}")

if __name__ == "__main__":
    debug_scan()
