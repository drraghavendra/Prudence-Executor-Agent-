import requests
import json

def test_deployment():
    base_url = "http://localhost:8000"
    
    # Test backend health
    try:
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        print("✅ Backend health check passed")
    except:
        print("❌ Backend health check failed")
        return False
    
    # Test database connection
    try:
        response = requests.get(f"{base_url}/api/users/test")
        assert response.status_code in [200, 404]
        print("✅ Database connection test passed")
    except:
        print("❌ Database connection test failed")
        return False
    
    # Test Cardano connection
    try:
        response = requests.get(f"{base_url}/api/cardano/network")
        assert response.status_code == 200
        print("✅ Cardano network connection test passed")
    except:
        print("❌ Cardano network connection test failed")
        return False
    
    print("🎉 All deployment tests passed!")
    return True

if __name__ == "__main__":
    test_deployment()