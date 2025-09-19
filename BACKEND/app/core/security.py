import firebase_admin
from firebase_admin import auth
from fastapi import HTTPException
from typing import Dict, Any
import jwt
from datetime import datetime, timedelta
from app.config import settings


async def verify_firebase_token(token: str) -> Dict[str, Any]:
    """
    Verify Firebase ID token and return user information
    """
    try:
        # Verify the ID token
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        email = decoded_token.get('email', '')
        
        # Get additional user data from Firestore if needed
        # You can expand this to fetch user role and other details
        
        user_info = {
            'uid': uid,
            'email': email,
            'role': decoded_token.get('role', 'user'),  # Default role
            'region': decoded_token.get('region', ''),
            'verified': decoded_token.get('email_verified', False)
        }
        
        return user_info
        
    except firebase_admin.auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except firebase_admin.auth.RevokedIdTokenError:
        raise HTTPException(status_code=401, detail="Token has been revoked")
    except firebase_admin.auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")


def create_access_token(data: Dict[str, Any], expires_delta: timedelta = None) -> str:
    """
    Create JWT access token (for internal use if needed)
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_access_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT access token (for internal use if needed)
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Role definitions
USER_ROLES = {
    "user": "Community Member",
    "health_worker": "Community Health Worker",
    "asha": "ASHA Worker", 
    "local_authority": "Local Government Authority",
    "state_health_authority": "State Health Authority",
    "admin": "System Administrator",
    "iot_sensor": "IoT Device"
}


def get_role_permissions(role: str) -> Dict[str, bool]:
    """
    Define permissions for each role
    """
    permissions = {
        "user": {
            "view_alerts": True,
            "report_symptoms": True,
            "view_water_quality": True
        },
        "health_worker": {
            "view_alerts": True,
            "report_symptoms": True,
            "upload_health_data": True,
            "view_water_quality": True,
            "send_community_alerts": True
        },
        "asha": {
            "view_alerts": True,
            "report_symptoms": True,
            "upload_health_data": True,
            "view_water_quality": True,
            "send_community_alerts": True,
            "access_household_data": True
        },
        "local_authority": {
            "view_alerts": True,
            "upload_health_data": True,
            "view_water_quality": True,
            "send_community_alerts": True,
            "access_regional_data": True,
            "manage_resources": True
        },
        "state_health_authority": {
            "view_alerts": True,
            "upload_health_data": True,
            "view_water_quality": True,
            "send_community_alerts": True,
            "access_regional_data": True,
            "access_state_data": True,
            "manage_resources": True,
            "generate_reports": True
        },
        "admin": {
            "full_access": True
        },
        "iot_sensor": {
            "upload_sensor_data": True
        }
    }
    
    return permissions.get(role, permissions["user"])