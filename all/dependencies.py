from fastapi import Depends, HTTPException, Header
from typing import Optional, Dict, Any
import firebase_admin
from firebase_admin import auth
from app.core.security import verify_firebase_token
from app.db.database import get_firestore_client


async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Verify Firebase token and return user information
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        # Extract token from "Bearer <token>"
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
            
        # Verify Firebase token
        user_info = await verify_firebase_token(token)
        return user_info
        
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Check if current user has admin privileges
    """
    user_role = current_user.get("role", "user")
    if user_role not in ["admin", "state_health_authority"]:
        raise HTTPException(
            status_code=403, 
            detail="Insufficient permissions. Admin access required."
        )
    return current_user


async def get_health_worker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Check if current user is a health worker (ASHA/CHW)
    """
    user_role = current_user.get("role", "user")
    allowed_roles = ["health_worker", "asha", "admin", "local_authority", "state_health_authority"]
    
    if user_role not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail="Access restricted to health workers and authorities"
        )
    return current_user


async def get_data_collector(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Check if user can collect/upload data
    """
    user_role = current_user.get("role", "user")
    allowed_roles = ["health_worker", "asha", "iot_sensor", "admin", "local_authority"]
    
    if user_role not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions for data collection"
        )
    return current_user


async def get_firestore_db():
    """
    Get Firestore database client
    """
    return get_firestore_client()


# Role-based access control decorator
def require_roles(allowed_roles: list):
    """
    Create a dependency that requires specific roles
    """
    async def check_role(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role", "user")
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access restricted. Required roles: {', '.join(allowed_roles)}"
            )
        return current_user
    
    return check_role