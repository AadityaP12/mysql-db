from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    USER = "user"
    HEALTH_WORKER = "health_worker"
    ASHA = "asha"
    LOCAL_AUTHORITY = "local_authority"
    STATE_HEALTH_AUTHORITY = "state_health_authority"
    ADMIN = "admin"
    IOT_SENSOR = "iot_sensor"


class UserRegistration(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2, max_length=100)
    phone_number: str = Field(..., pattern=r"^[+]?[1-9]?[0-9]{7,15}$")
    role: UserRole = UserRole.USER
    region: str = Field(..., min_length=2, max_length=100)
    state: str = Field(..., min_length=2, max_length=50)
    district: str = Field(..., min_length=2, max_length=50)
    block: Optional[str] = None
    village: Optional[str] = None
    organization: Optional[str] = None
    employee_id: Optional[str] = None
    preferred_language: str = "en"
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "asha.worker@health.gov.in",
                "password": "securepass123",
                "full_name": "Priya Sharma",
                "phone_number": "+919876543210",
                "role": "asha",
                "region": "Northeast India",
                "state": "Assam",
                "district": "Kamrup",
                "block": "Guwahati",
                "village": "Jalukbari",
                "organization": "State Health Mission",
                "employee_id": "ASHA001",
                "preferred_language": "as"
            }
        }


class UserLogin(BaseModel):
    email: EmailStr
    password: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "password123"
            }
        }


class FirebaseTokenRequest(BaseModel):
    firebase_token: str = Field(..., description="Firebase ID token")
    
    class Config:
        json_schema_extra = {
            "example": {
                "firebase_token": "eyJhbGciOiJSUzI1NiIsImtpZCI6IjU..."
            }
        }


class UserProfile(BaseModel):
    uid: str
    email: str
    full_name: str
    phone_number: str
    role: UserRole
    region: str
    state: str
    district: str
    block: Optional[str] = None
    village: Optional[str] = None
    organization: Optional[str] = None
    employee_id: Optional[str] = None
    preferred_language: str = "en"
    verified: bool = False
    active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    profile_completed: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "uid": "firebase_user_id_123",
                "email": "user@example.com",
                "full_name": "John Doe",
                "phone_number": "+919876543210",
                "role": "health_worker",
                "region": "Northeast India",
                "state": "Assam",
                "district": "Kamrup",
                "verified": True,
                "active": True,
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    phone_number: Optional[str] = Field(None, pattern=r"^[+]?[1-9]?[0-9]{7,15}$")
    region: Optional[str] = Field(None, min_length=2, max_length=100)
    state: Optional[str] = Field(None, min_length=2, max_length=50)
    district: Optional[str] = Field(None, min_length=2, max_length=50)
    block: Optional[str] = None
    village: Optional[str] = None
    organization: Optional[str] = None
    employee_id: Optional[str] = None
    preferred_language: Optional[str] = "en"
    
    class Config:
        json_schema_extra = {
            "example": {
                "full_name": "Updated Name",
                "phone_number": "+919876543210",
                "preferred_language": "hi"
            }
        }


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: UserProfile


class AuthResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    user: Optional[UserProfile] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Authentication successful",
                "data": {
                    "access_token": "jwt_token_here"
                },
                "user": {
                    "uid": "user_id",
                    "email": "user@example.com",
                    "role": "health_worker"
                }
            }
        }


class PasswordReset(BaseModel):
    email: EmailStr
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com"
            }
        }


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)
    confirm_password: str
    
    def passwords_match(self) -> bool:
        return self.new_password == self.confirm_password
    
    class Config:
        json_schema_extra = {
            "example": {
                "current_password": "oldpassword",
                "new_password": "newpassword123",
                "confirm_password": "newpassword123"
            }
        }


class RoleChangeRequest(BaseModel):
    user_uid: str
    new_role: UserRole
    reason: str = Field(..., min_length=10)
    requested_by: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_uid": "firebase_user_id",
                "new_role": "health_worker",
                "reason": "User has been appointed as community health worker",
                "requested_by": "admin_user_id"
            }
        }
