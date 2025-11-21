import sys
import os
sys.path.append(os.getcwd())
from fastapi.testclient import TestClient
from demos.app import app
import os

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "Omnilingual ASR" in response.text

def test_get_config():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "languages" in data
    assert len(data["models"]) > 0

def test_static_files():
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    
    response = client.get("/static/js/app.js")
    assert response.status_code == 200

if __name__ == "__main__":
    # Run tests manually
    try:
        test_read_main()
        print("✅ Main page test passed")
        test_get_config()
        print("✅ Config API test passed")
        test_static_files()
        print("✅ Static files test passed")
    except Exception as e:
        print(f"❌ Test failed: {e}")
