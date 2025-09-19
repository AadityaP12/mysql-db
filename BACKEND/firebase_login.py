import requests
import json

# Firebase project details
API_KEY = "AIzaSyC6pNVgbv6KLkeKg19W_9co8FPztkbemrc"
FIREBASE_AUTH_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"

# Replace with your Firebase test user credentials
EMAIL = "test@example.com"
PASSWORD = "your_password"

def firebase_login(email: str, password: str):
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(FIREBASE_AUTH_URL, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ Firebase login successful!")
        print(f"User: {data['email']}")
        print(f"ID Token (use this in Authorization header):\n\n{data['idToken']}\n")
        print("Example curl command:\n")
        print(f"""curl -X GET http://localhost:8000/api/v1/auth/profile \\
  -H "Authorization: Bearer {data['idToken']}" """)
    else:
        print("\n❌ Login failed!")
        print(response.json())

if __name__ == "__main__":
    firebase_login(EMAIL, PASSWORD)
