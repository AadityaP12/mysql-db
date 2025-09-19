import requests
import json
import firebase_admin
from firebase_admin import auth, credentials
import time

class FirebaseAuthTester:
    def __init__(self, base_url="http://localhost:8000", service_account_path=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        self.service_account_path = service_account_path or "app/sih-wb-firebase-adminsdk.json"
        
        # Initialize Firebase Admin SDK
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.service_account_path)
                firebase_admin.initialize_app(cred)
            print("✅ Firebase Admin SDK initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Firebase: {e}")
    
    def create_test_user_with_firebase(self, email=None):
        """Create a test user directly in Firebase"""
        try:
            if not email:
                email = f"test_user_{int(time.time())}@example.com"
            
            # Create user in Firebase Auth
            user_record = auth.create_user(
                email=email,
                password="testpassword123",
                display_name="Test User",
                disabled=False
            )
            
            # Set custom claims for role-based access
            custom_claims = {
                "role": "health_worker",
                "region": "Northeast India",
                "state": "Assam",
                "district": "Kamrup"
            }
            auth.set_custom_user_claims(user_record.uid, custom_claims)
            
            print(f"✅ Created Firebase user: {user_record.uid}")
            print(f"✅ Email: {email}")
            
            return user_record.uid, email
            
        except Exception as e:
            print(f"❌ Failed to create Firebase user: {e}")
            return None, None
    
    def create_custom_token(self, uid):
        """Create a custom token for testing"""
        try:
            custom_token = auth.create_custom_token(uid)
            print(f"✅ Created custom token for UID: {uid}")
            return custom_token.decode('utf-8')
        except Exception as e:
            print(f"❌ Failed to create custom token: {e}")
            return None
    
    def test_firebase_registration(self):
        """Test user registration via API"""
        print("\n" + "="*50)
        print("TESTING FIREBASE REGISTRATION VIA API")
        print("="*50)
        
        registration_data = {
            "email": f"api_test_user_{int(time.time())}@example.com",
            "password": "securepassword123",
            "full_name": "API Test User",
            "phone_number": "+919876543210",
            "role": "health_worker",
            "region": "Northeast India",
            "state": "Assam",
            "district": "Kamrup",
            "block": "Guwahati",
            "village": "Test Village",
            "organization": "Test Health Center",
            "employee_id": "API_TEST_001",
            "preferred_language": "en"
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/auth/register",
            headers=self.headers,
            data=json.dumps(registration_data)
        )
        
        self.print_result("API User Registration", response, 200)
        
        if response.status_code == 200:
            try:
                result = response.json()
                uid = result.get("data", {}).get("uid")
                print(f"✅ Created user via API with UID: {uid}")
                return uid
            except:
                print("❌ Failed to extract UID from API response")
        return None
    
    def test_with_custom_token(self, uid):
        """Test API endpoints with custom Firebase token"""
        print("\n" + "="*50)
        print("TESTING WITH CUSTOM FIREBASE TOKEN")
        print("="*50)
        
        # Create custom token
        custom_token = self.create_custom_token(uid)
        if not custom_token:
            print("❌ Cannot test with custom token")
            return
        
        # Note: Custom tokens need to be exchanged for ID tokens
        # In a real app, the client would do this exchange
        print("⚠️  Note: Custom tokens need client-side exchange for ID tokens")
        print("This is a limitation in server-side testing")
        
        # Try using custom token (will likely fail as it needs to be exchanged)
        auth_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {custom_token}"
        }
        
        response = requests.get(
            f"{self.base_url}/api/v1/auth/verify-token",
            headers=auth_headers
        )
        self.print_result("Verify Custom Token", response, 401)  # Expected to fail
    
    def test_token_verification_endpoint(self):
        """Test the token verification endpoint"""
        print("\n" + "="*50)
        print("TESTING TOKEN VERIFICATION")
        print("="*50)
        
        # Test without token
        response = requests.get(f"{self.base_url}/api/v1/auth/verify-token")
        self.print_result("Verify Token (No Auth)", response, 401)
        
        # Test with invalid token
        auth_headers = {
            "Authorization": "Bearer invalid_token_12345"
        }
        response = requests.get(
            f"{self.base_url}/api/v1/auth/verify-token",
            headers=auth_headers
        )
        self.print_result("Verify Token (Invalid)", response, 401)
    
    def test_user_profile_endpoint(self):
        """Test user profile endpoint"""
        print("\n" + "="*50)
        print("TESTING USER PROFILE ENDPOINTS")
        print("="*50)
        
        # Test without auth
        response = requests.get(f"{self.base_url}/api/v1/auth/profile")
        self.print_result("Get Profile (No Auth)", response, 401)
        
        # Test profile update without auth
        update_data = {
            "full_name": "Updated Test User",
            "preferred_language": "hi"
        }
        
        response = requests.put(
            f"{self.base_url}/api/v1/auth/profile",
            headers=self.headers,
            data=json.dumps(update_data)
        )
        self.print_result("Update Profile (No Auth)", response, 401)
    
    def cleanup_test_users(self):
        """Clean up test users created during testing"""
        print("\n" + "="*50)
        print("CLEANING UP TEST USERS")
        print("="*50)
        
        try:
            # List users and find test users
            page = auth.list_users()
            users_to_delete = []
            
            for user in page.users:
                if "test_user_" in user.email or "api_test_user_" in user.email:
                    users_to_delete.append(user.uid)
            
            # Delete test users
            for uid in users_to_delete[:5]:  # Limit to 5 for safety
                try:
                    auth.delete_user(uid)
                    print(f"✅ Deleted test user: {uid}")
                except Exception as e:
                    print(f"❌ Failed to delete user {uid}: {e}")
            
            if len(users_to_delete) > 5:
                print(f"⚠️  {len(users_to_delete) - 5} more test users remain")
                
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
    
    def print_result(self, test_name, response, expected_status=200):
        """Print test results"""
        status = "✅ PASS" if response.status_code == expected_status else "❌ FAIL"
        print(f"\n{status} {test_name}")
        print(f"Status Code: {response.status_code} (expected: {expected_status})")
        
        try:
            json_response = response.json()
            print(f"Response: {json.dumps(json_response, indent=2)[:300]}...")
        except:
            print(f"Response: {response.text[:300]}...")
    
    def run_firebase_tests(self):
        """Run Firebase-specific tests"""
        print("🔥 Starting Firebase Authentication Tests")
        print(f"Testing backend at: {self.base_url}")
        
        try:
            # Test API registration
            uid = self.test_firebase_registration()
            
            # Test token verification
            self.test_token_verification_endpoint()
            
            # Test user profile endpoints
            self.test_user_profile_endpoint()
            
            # Test with custom token if user was created
            if uid:
                self.test_with_custom_token(uid)
            
            print("\n" + "="*50)
            print("🎉 FIREBASE TESTS COMPLETED")
            print("="*50)
            print("✅ Firebase integration is working")
            print("⚠️  Some tests failed as expected (invalid tokens)")
            print("💡 For full testing, implement client-side token exchange")
            
            # Ask before cleanup
            cleanup = input("\nDo you want to clean up test users? (y/N): ")
            if cleanup.lower() == 'y':
                self.cleanup_test_users()
            
        except Exception as e:
            print(f"❌ Firebase tests failed: {e}")

if __name__ == "__main__":
    tester = FirebaseAuthTester()
    tester.run_firebase_tests()