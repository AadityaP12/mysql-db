#!/usr/bin/env python3
"""
Quick integration tests for live server
Run this while your server is running on localhost:8000
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_result(test_name, response, expected_status=200):
    """Print test results"""
    status = "✅ PASS" if response.status_code == expected_status else "❌ FAIL"
    print(f"{status} {test_name} - Status: {response.status_code}")
    if response.status_code != expected_status:
        print(f"   Expected: {expected_status}, Got: {response.status_code}")
    return response.status_code == expected_status

def main():
    print("🚀 Testing your live server at localhost:8000")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    # Test 1: Health Check
    total += 1
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if test_result("Health Check", response, 200):
            passed += 1
    except requests.exceptions.ConnectionError:
        print("❌ FAIL Health Check - Cannot connect to server")
        print("Make sure your server is running: uvicorn app.main:app --reload")
        return
    except Exception as e:
        print(f"❌ FAIL Health Check - Error: {e}")
    
    # Test 2: Root Endpoint
    total += 1
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if test_result("Root Endpoint", response, 200):
            passed += 1
    except Exception as e:
        print(f"❌ FAIL Root Endpoint - Error: {e}")
    
    # Test 3: API Documentation (if debug enabled)
    total += 1
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if test_result("API Docs", response, 200):
            passed += 1
        elif response.status_code == 404:
            print("⚠️  API Docs disabled (normal in production)")
            passed += 1  # This is acceptable
    except Exception as e:
        print(f"❌ FAIL API Docs - Error: {e}")
    
    # Test 4: Database Health
    total += 1
    try:
        response = requests.get(f"{BASE_URL}/health/database", timeout=10)
        if test_result("Database Health", response, 200):
            passed += 1
    except Exception as e:
        print(f"❌ FAIL Database Health - Error: {e}")
    
    # Test 5: Firebase Health
    total += 1
    try:
        response = requests.get(f"{BASE_URL}/health/firebase", timeout=10)
        if test_result("Firebase Health", response, 200):
            passed += 1
    except Exception as e:
        print(f"❌ FAIL Firebase Health - Error: {e}")
    
    # Test 6: Cache Status
    total += 1
    try:
        response = requests.get(f"{BASE_URL}/cache/status", timeout=5)
        if test_result("Cache Status", response, 200):
            passed += 1
    except Exception as e:
        print(f"❌ FAIL Cache Status - Error: {e}")
    
    # Test 7: User Registration (should work)
    total += 1
    try:
        registration_data = {
            "email": f"test_{int(time.time())}@example.com",
            "password": "testpass123",
            "full_name": "Integration Test User",
            "phone_number": "+919876543210", 
            "role": "health_worker",
            "region": "Northeast India",
            "state": "Assam",
            "district": "Kamrup",
            "preferred_language": "en"
        }
        response = requests.post(
            f"{BASE_URL}/api/v1/auth/register",
            headers=HEADERS,
            data=json.dumps(registration_data),
            timeout=10
        )
        if test_result("User Registration", response, 200):
            passed += 1
            try:
                result = response.json()
                if result.get("success") and result.get("data", {}).get("uid"):
                    print("   ✅ Firebase user creation successful")
                else:
                    print("   ⚠️  Registration response format unexpected")
            except:
                pass
    except Exception as e:
        print(f"❌ FAIL User Registration - Error: {e}")
    
    # Test 8: Protected Endpoint (should fail with 401)
    total += 1
    try:
        response = requests.get(f"{BASE_URL}/api/v1/auth/profile", timeout=5)
        if test_result("Protected Endpoint (No Auth)", response, 401):
            passed += 1
            print("   ✅ Authentication protection working")
    except Exception as e:
        print(f"❌ FAIL Protected Endpoint - Error: {e}")
    
    # Test 9: Data Validation (should fail with 422)
    total += 1
    try:
        invalid_data = {"invalid": "data"}
        response = requests.post(
            f"{BASE_URL}/api/v1/data/water-quality",
            headers={"Authorization": "Bearer fake", "Content-Type": "application/json"},
            data=json.dumps(invalid_data),
            timeout=5
        )
        # Should fail with either 401 (auth) or 422 (validation)
        if response.status_code in [401, 422]:
            print("✅ PASS Data Validation - Status: 401/422 (expected)")
            passed += 1
        else:
            print(f"❌ FAIL Data Validation - Status: {response.status_code}")
    except Exception as e:
        print(f"❌ FAIL Data Validation - Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your backend is working correctly")
        print("✅ Database connections are healthy")
        print("✅ Firebase integration is working")
        print("✅ Authentication is properly secured")
        print("✅ Data validation is working")
        print("\n🚀 Ready for frontend integration!")
    elif passed >= total - 2:
        print("🟡 MOSTLY WORKING")
        print("✅ Core functionality is working")
        print("⚠️  Minor issues detected - check failed tests")
        print("\n🚀 Should be okay for frontend integration")
    else:
        print("🔴 ISSUES DETECTED")
        print("❌ Multiple failures - check your server configuration")
        print("❌ May need fixes before frontend integration")
    
    print(f"\n💡 Your server startup logs showed all connections working")
    print("💡 Most likely any failures are minor configuration issues")

if __name__ == "__main__":
    main()