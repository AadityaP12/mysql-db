import os
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get service account path from .env
service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")

# Initialize Firebase Admin SDK
cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred)

# The UID of the user from Firebase Console
USER_UID = "6WzUaC8CSKQ5r9waizDBFn54GQh1"

# Set a new password
new_password = "Test@12345"  # <-- choose your own strong password

user = auth.update_user(
    USER_UID,
    password=new_password
)

print(f"✅ Password reset successfully for UID {user.uid}")
