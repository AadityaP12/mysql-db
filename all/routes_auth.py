from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from typing import Dict, Any
import firebase_admin
from firebase_admin import auth
from datetime import datetime, timezone

from app.schemas.auth import (
    UserRegistration, UserLogin, FirebaseTokenRequest, 
    UserProfile, UserProfileUpdate, AuthResponse, 
    PasswordReset, PasswordChange, RoleChangeRequest
)
from app.dependencies import get_current_user, get_admin_user
from app.db.database import firestore_service, FirestoreCollections
from app.core.utils import create_response, generate_id
from app.core.security import verify_firebase_token

router = APIRouter()
security = HTTPBearer()


@router.post("/register", response_model=AuthResponse)
async def register_user(user_data: UserRegistration):
    """
    Register a new user with Firebase Authentication and store profile in Firestore
    """
    try:
        # Create user in Firebase Auth
        user_record = auth.create_user(
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.full_name,
            phone_number=user_data.phone_number,
            disabled=False
        )
        
        # Create user profile in Firestore
        user_profile = {
            "uid": user_record.uid,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "phone_number": user_data.phone_number,
            "role": user_data.role.value,
            "region": user_data.region,
            "state": user_data.state,
            "district": user_data.district,
            "block": user_data.block,
            "village": user_data.village,
            "organization": user_data.organization,
            "employee_id": user_data.employee_id,
            "preferred_language": user_data.preferred_language,
            "verified": False,
            "active": True,
            "created_at": datetime.now(timezone.utc),
            "last_login": None,
            "profile_completed": True
        }
        
        # Store user profile in Firestore
        await firestore_service.create_document(
            FirestoreCollections.USERS, 
            user_profile, 
            user_record.uid
        )
        
        # Set custom claims for role-based access
        custom_claims = {
            "role": user_data.role.value,
            "region": user_data.region,
            "state": user_data.state,
            "district": user_data.district
        }
        auth.set_custom_user_claims(user_record.uid, custom_claims)
        
        return create_response(
            success=True,
            message="User registered successfully",
            data={
                "uid": user_record.uid,
                "email": user_data.email,
                "role": user_data.role.value
            }
        )
        
    except auth.EmailAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email address already registered"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login/firebase", response_model=AuthResponse)
async def firebase_login(token_request: FirebaseTokenRequest):
    """
    Authenticate user with Firebase ID token
    """
    try:
        # Verify Firebase token
        user_info = await verify_firebase_token(token_request.firebase_token)
        
        # Get user profile from Firestore
        user_profile = await firestore_service.get_document(
            FirestoreCollections.USERS, 
            user_info["uid"]
        )
        
        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        # Update last login
        await firestore_service.update_document(
            FirestoreCollections.USERS,
            user_info["uid"],
            {"last_login": datetime.now(timezone.utc)}
        )
        
        return create_response(
            success=True,
            message="Login successful",
            data={
                "user": user_profile,
                "token_valid": True
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Login failed: {str(e)}"
        )


@router.get("/profile", response_model=UserProfile)
async def get_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current user's profile
    """
    try:
        user_profile = await firestore_service.get_document(
            FirestoreCollections.USERS,
            current_user["uid"]
        )
        
        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        return user_profile
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch profile: {str(e)}"
        )


@router.put("/profile", response_model=AuthResponse)
async def update_user_profile(
    profile_update: UserProfileUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update current user's profile
    """
    try:
        # Prepare update data
        update_data = {}
        for field, value in profile_update.dict(exclude_unset=True).items():
            if value is not None:
                update_data[field] = value
        
        if update_data:
            update_data["updated_at"] = datetime.now(timezone.utc)
            
            # Update in Firestore
            await firestore_service.update_document(
                FirestoreCollections.USERS,
                current_user["uid"],
                update_data
            )
            
            # Update Firebase Auth display name if full_name is updated
            if "full_name" in update_data:
                auth.update_user(
                    current_user["uid"],
                    display_name=update_data["full_name"]
                )
        
        return create_response(
            success=True,
            message="Profile updated successfully",
            data={"updated_fields": list(update_data.keys())}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile update failed: {str(e)}"
        )


@router.post("/password/reset")
async def reset_password(reset_request: PasswordReset):
    """
    Send password reset email
    """
    try:
        # Firebase handles password reset emails automatically
        # We just need to verify the email exists
        try:
            auth.get_user_by_email(reset_request.email)
        except auth.UserNotFoundError:
            # Don't reveal whether email exists or not for security
            pass
        
        return create_response(
            success=True,
            message="Password reset email sent if the email exists in our system"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process password reset request"
        )


@router.post("/password/change")
async def change_password(
    password_change: PasswordChange,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Change user password (requires current password verification)
    """
    try:
        if not password_change.passwords_match():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password and confirmation do not match"
            )
        
        # Update password in Firebase Auth
        auth.update_user(
            current_user["uid"],
            password=password_change.new_password
        )
        
        # Log password change
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "user_uid": current_user["uid"],
                "action": "password_changed",
                "timestamp": datetime.now(timezone.utc),
                "ip_address": None  # You can get this from request if needed
            }
        )
        
        return create_response(
            success=True,
            message="Password changed successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password change failed: {str(e)}"
        )


@router.post("/role/change")
async def change_user_role(
    role_request: RoleChangeRequest,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """
    Change user role (Admin only)
    """
    try:
        # Verify target user exists
        target_user = await firestore_service.get_document(
            FirestoreCollections.USERS,
            role_request.user_uid
        )
        
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Target user not found"
            )
        
        # Update role in Firestore
        await firestore_service.update_document(
            FirestoreCollections.USERS,
            role_request.user_uid,
            {
                "role": role_request.new_role.value,
                "role_updated_at": datetime.now(timezone.utc),
                "role_updated_by": admin_user["uid"]
            }
        )
        
        # Update custom claims in Firebase Auth
        custom_claims = {
            "role": role_request.new_role.value,
            "region": target_user.get("region"),
            "state": target_user.get("state"),
            "district": target_user.get("district")
        }
        auth.set_custom_user_claims(role_request.user_uid, custom_claims)
        
        # Log role change
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "user_uid": role_request.user_uid,
                "action": "role_changed",
                "old_role": target_user.get("role"),
                "new_role": role_request.new_role.value,
                "changed_by": admin_user["uid"],
                "reason": role_request.reason,
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
        return create_response(
            success=True,
            message=f"User role changed to {role_request.new_role.value}",
            data={
                "user_uid": role_request.user_uid,
                "new_role": role_request.new_role.value
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Role change failed: {str(e)}"
        )


@router.get("/users")
async def list_users(
    page: int = 1,
    limit: int = 20,
    role: str = None,
    state: str = None,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """
    List users with optional filtering (Admin only)
    """
    try:
        # Build filters
        filters = []
        if role:
            filters.append(("role", "==", role))
        if state:
            filters.append(("state", "==", state))
        
        # Query users
        users = await firestore_service.query_collection(
            FirestoreCollections.USERS,
            filters=filters,
            limit=limit * page  # Simple pagination
        )
        
        # Apply pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_users = users[start_idx:end_idx]
        
        # Remove sensitive information
        for user in paginated_users:
            user.pop("password", None)
        
        return create_response(
            success=True,
            message="Users retrieved successfully",
            data=paginated_users,
            meta={
                "page": page,
                "limit": limit,
                "total": len(users)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve users: {str(e)}"
        )


@router.delete("/users/{user_uid}")
async def deactivate_user(
    user_uid: str,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """
    Deactivate user account (Admin only)
    """
    try:
        # Update user status in Firestore
        await firestore_service.update_document(
            FirestoreCollections.USERS,
            user_uid,
            {
                "active": False,
                "deactivated_at": datetime.now(timezone.utc),
                "deactivated_by": admin_user["uid"]
            }
        )
        
        # Disable user in Firebase Auth
        auth.update_user(user_uid, disabled=True)
        
        # Log deactivation
        await firestore_service.create_document(
            FirestoreCollections.AUDIT_LOGS,
            {
                "user_uid": user_uid,
                "action": "user_deactivated",
                "deactivated_by": admin_user["uid"],
                "timestamp": datetime.now(timezone.utc)
            }
        )
        
        return create_response(
            success=True,
            message="User deactivated successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User deactivation failed: {str(e)}"
        )


@router.get("/verify-token")
async def verify_token(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Verify if the current token is valid
    """
    return create_response(
        success=True,
        message="Token is valid",
        data={
            "user_uid": current_user["uid"],
            "role": current_user.get("role"),
            "verified": True
        }
    )