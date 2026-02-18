import requests
import os

API_URL = "http://localhost:8000/api"

def test_file_upload():
    print("⚡ [TEST] Testing File Upload Endpoint...")

    # Create a dummy test file
    test_filename = "test_doc.txt"
    with open(test_filename, "w") as f:
        f.write("This is a test document for Edison v4.0 local ingestion.")

    try:
        # Upload
        files = {'file': open(test_filename, 'rb')}
        response = requests.post(f"{API_URL}/upload", files=files)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Upload Success: {data}")
            assert data['filename'] == test_filename
            assert os.path.exists(data['path'])
        else:
            print(f"❌ Upload Failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")
    finally:
        # Cleanup
        if os.path.exists(test_filename):
            os.remove(test_filename)

if __name__ == "__main__":
    test_file_upload()
