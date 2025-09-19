from typing import Any, Dict, Optional, List, Union, Tuple
from datetime import datetime, timezone, timedelta
import hashlib
import uuid
import json
import re
import asyncio
from functools import wraps, lru_cache
import time
import csv
from io import StringIO, BytesIO
import base64
from decimal import Decimal, InvalidOperation

from app.config import settings
from app.core.monitoring import structured_logger, metrics_collector

logger = structured_logger


def create_response(
    success: bool = True,
    message: str = "",
    data: Any = None,
    status_code: int = 200,
    meta: Optional[Dict] = None,
    errors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create standardized API response format with enhanced features
    """
    response = {
        "success": success,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status_code": status_code,
        "api_version": settings.API_VERSION
    }
    
    if data is not None:
        response["data"] = data
    
    if meta:
        response["meta"] = meta
    
    if errors:
        response["errors"] = errors
    
    # Add request tracking ID for debugging
    response["request_id"] = generate_id("REQ_")
    
    return response


def generate_id(prefix: str = "", length: int = 32) -> str:
    """
    Generate unique ID with optional prefix and custom length
    """
    if length < 8:
        length = 8
    elif length > 64:
        length = 64
    
    # Use timestamp + random for better uniqueness
    timestamp = str(int(time.time() * 1000))  # milliseconds
    unique_part = str(uuid.uuid4()).replace('-', '')
    
    # Combine and truncate to desired length
    combined = timestamp + unique_part
    unique_id = combined[:length]
    
    return f"{prefix}{unique_id}" if prefix else unique_id


def hash_string(text: str, algorithm: str = "sha256") -> str:
    """
    Generate hash of string with support for different algorithms
    """
    if algorithm == "md5":
        return hashlib.md5(text.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(text.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(text.encode()).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(text.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def validate_coordinates(latitude: float, longitude: float, region: str = "northeast_india") -> bool:
    """
    Validate if coordinates are within specified region bounds
    """
    if region == "northeast_india":
        # Northeast India bounds
        lat_min, lat_max = 21.0, 30.0
        lon_min, lon_max = 87.0, 98.0
    elif region == "india":
        # All India bounds
        lat_min, lat_max = 6.0, 37.0
        lon_min, lon_max = 68.0, 98.0
    elif region == "global":
        # Global bounds
        lat_min, lat_max = -90.0, 90.0
        lon_min, lon_max = -180.0, 180.0
    else:
        # Custom bounds from settings
        bounds = settings.GEO_BOUNDS
        lat_min, lat_max = bounds.get("lat_min", -90), bounds.get("lat_max", 90)
        lon_min, lon_max = bounds.get("lon_min", -180), bounds.get("lon_max", 180)
    
    return lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float, unit: str = "km") -> float:
    """
    Calculate distance between two coordinates using Haversine formula
    Supports different units: km, miles, meters, nautical_miles
    """
    from math import radians, sin, cos, sqrt, atan2
    
    # Earth's radius in kilometers
    R = 6371.0
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    distance_km = R * c
    
    # Convert to requested unit
    conversions = {
        "km": 1.0,
        "meters": 1000.0,
        "miles": 0.621371,
        "nautical_miles": 0.539957
    }
    
    return distance_km * conversions.get(unit, 1.0)


@lru_cache(maxsize=128)
def get_risk_level(risk_score: float, custom_thresholds: Optional[Tuple[float, float]] = None) -> Dict[str, str]:
    """
    Convert risk score to categorical risk level with caching
    """
    if custom_thresholds:
        low_threshold, high_threshold = custom_thresholds
    else:
        low_threshold = settings.RISK_THRESHOLD_LOW
        high_threshold = settings.RISK_THRESHOLD_HIGH
    
    if risk_score < low_threshold:
        return {
            "level": "LOW",
            "color": "#28a745",  # Green
            "color_name": "GREEN",
            "description": "Low risk of outbreak",
            "urgency": 1,
            "action_required": False
        }
    elif risk_score < high_threshold:
        return {
            "level": "MEDIUM", 
            "color": "#ffc107",  # Yellow
            "color_name": "YELLOW",
            "description": "Moderate risk - increased monitoring required",
            "urgency": 2,
            "action_required": True
        }
    else:
        return {
            "level": "HIGH",
            "color": "#dc3545",  # Red
            "color_name": "RED", 
            "description": "High risk - immediate action required",
            "urgency": 3,
            "action_required": True
        }


def format_phone_number(phone: str, country_code: str = "+91", validate: bool = True) -> str:
    """
    Format phone number with validation for different countries
    """
    # Remove any spaces, dashes, or special characters except +
    cleaned = re.sub(r'[^\d+]', '', phone)
    
    # Handle different country formats
    if country_code == "+91":  # India
        # Remove country code if present
        if cleaned.startswith("+91"):
            cleaned = cleaned[3:]
        elif cleaned.startswith("91") and len(cleaned) == 12:
            cleaned = cleaned[2:]
        elif cleaned.startswith("0") and len(cleaned) == 11:
            cleaned = cleaned[1:]
        
        # Validate 10-digit number
        if validate and (len(cleaned) != 10 or not cleaned.isdigit()):
            raise ValueError("Invalid Indian phone number format")
        
        return f"+91{cleaned}"
    
    elif country_code == "+1":  # US/Canada
        if cleaned.startswith("+1"):
            cleaned = cleaned[2:]
        elif cleaned.startswith("1") and len(cleaned) == 11:
            cleaned = cleaned[1:]
        
        if validate and (len(cleaned) != 10 or not cleaned.isdigit()):
            raise ValueError("Invalid US/Canada phone number format")
        
        return f"+1{cleaned}"
    
    else:
        # Generic international format
        if not cleaned.startswith("+"):
            cleaned = country_code + cleaned
        return cleaned


def get_language_text(text_dict: Dict[str, str], language: str = None) -> str:
    """
    Get text in specified language with fallback support
    """
    if not language:
        language = settings.DEFAULT_LANGUAGE
    
    # Try requested language
    if language in text_dict:
        return text_dict[language]
    
    # Try English as fallback
    if "en" in text_dict:
        return text_dict["en"]
    
    # Try default language from settings
    if settings.DEFAULT_LANGUAGE in text_dict:
        return text_dict[settings.DEFAULT_LANGUAGE]
    
    # Return first available
    if text_dict:
        return next(iter(text_dict.values()))
    
    return ""


def paginate_results(data: List, page: int = 1, limit: int = 20) -> Dict[str, Any]:
    """
    Paginate a list of results with enhanced metadata
    """
    if page < 1:
        page = 1
    if limit < 1:
        limit = 1
    elif limit > 1000:
        limit = 1000  # Safety limit
    
    total = len(data)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    paginated_data = data[start_idx:end_idx]
    total_pages = (total + limit - 1) // limit
    
    return {
        "data": paginated_data,
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
            "total_items": total,
            "items_per_page": limit,
            "items_on_page": len(paginated_data),
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "next_page": page + 1 if page < total_pages else None,
            "prev_page": page - 1 if page > 1 else None,
            "start_index": start_idx + 1 if paginated_data else 0,
            "end_index": start_idx + len(paginated_data)
        }
    }


def sanitize_input(text: str, allow_html: bool = False, max_length: int = None) -> str:
    """
    Advanced input sanitization with configurable options
    """
    if not isinstance(text, str):
        text = str(text)
    
    if not allow_html:
        # Remove HTML tags
        text = re.sub(r'<[^>]*>', '', text)
        
        # Remove potentially harmful characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '\\', '/', '`']
        for char in dangerous_chars:
            text = text.replace(char, '')
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Apply length limit
    if max_length and len(text) > max_length:
        text = text[:max_length].strip()
    
    return text


def validate_email(email: str) -> bool:
    """
    Validate email address format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_password_strength(password: str) -> Dict[str, Any]:
    """
    Validate password strength and return detailed analysis
    """
    analysis = {
        "is_valid": False,
        "score": 0,
        "requirements": {
            "length": len(password) >= 8,
            "uppercase": bool(re.search(r'[A-Z]', password)),
            "lowercase": bool(re.search(r'[a-z]', password)),
            "digits": bool(re.search(r'\d', password)),
            "special_chars": bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        },
        "suggestions": []
    }
    
    # Calculate score
    score = 0
    for req, met in analysis["requirements"].items():
        if met:
            score += 1
    
    # Additional scoring
    if len(password) >= 12:
        score += 1
    if len(password) >= 16:
        score += 1
    
    analysis["score"] = min(score, 5)  # Max score of 5
    
    # Generate suggestions
    if not analysis["requirements"]["length"]:
        analysis["suggestions"].append("Use at least 8 characters")
    if not analysis["requirements"]["uppercase"]:
        analysis["suggestions"].append("Include uppercase letters")
    if not analysis["requirements"]["lowercase"]:
        analysis["suggestions"].append("Include lowercase letters")
    if not analysis["requirements"]["digits"]:
        analysis["suggestions"].append("Include numbers")
    if not analysis["requirements"]["special_chars"]:
        analysis["suggestions"].append("Include special characters")
    
    # Check for common patterns
    if re.search(r'(.)\1{2,}', password):
        analysis["suggestions"].append("Avoid repeating characters")
        score -= 1
    
    if re.search(r'(123|abc|qwe|password)', password.lower()):
        analysis["suggestions"].append("Avoid common patterns")
        score -= 1
    
    analysis["score"] = max(score, 0)
    analysis["is_valid"] = analysis["score"] >= 3 and all([
        analysis["requirements"]["length"],
        analysis["requirements"]["uppercase"] or analysis["requirements"]["lowercase"],
        analysis["requirements"]["digits"] or analysis["requirements"]["special_chars"]
    ])
    
    return analysis


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable format
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    elif hasattr(obj, 'dict'):  # Pydantic models
        return obj.dict()
    elif hasattr(obj, '__dict__'):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj


def export_to_csv(data: List[Dict], filename: str = None) -> StringIO:
    """
    Export data to CSV format
    """
    if not data:
        raise ValueError("No data to export")
    
    output = StringIO()
    
    # Get all possible field names
    fieldnames = set()
    for record in data:
        fieldnames.update(record.keys())
    
    fieldnames = sorted(list(fieldnames))
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for record in data:
        # Convert complex objects to strings
        clean_record = {}
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                clean_record[key] = json.dumps(value)
            elif isinstance(value, datetime):
                clean_record[key] = value.isoformat()
            else:
                clean_record[key] = str(value) if value is not None else ""
        
        writer.writerow(clean_record)
    
    output.seek(0)
    return output


def export_to_excel(data: List[Dict], sheet_name: str = "Data") -> BytesIO:
    """
    Export data to Excel format (requires openpyxl)
    """
    try:
        import openpyxl
        from openpyxl.utils.dataframe import dataframe_to_rows
        import pandas as pd
    except ImportError:
        raise ImportError("openpyxl and pandas are required for Excel export")
    
    if not data:
        raise ValueError("No data to export")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Convert datetime columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to identify datetime columns
            sample_values = df[col].dropna().head()
            if len(sample_values) > 0:
                try:
                    pd.to_datetime(sample_values.iloc[0])
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
    
    # Create Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets[sheet_name]
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    
    output.seek(0)
    return output


def parse_duration_string(duration_str: str) -> int:
    """
    Parse duration string (e.g., '1h', '30m', '45s') to seconds
    """
    pattern = r'(\d+)([smhd])'
    matches = re.findall(pattern, duration_str.lower())
    
    total_seconds = 0
    multipliers = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400
    }
    
    for value, unit in matches:
        total_seconds += int(value) * multipliers.get(unit, 1)
    
    return total_seconds


def format_duration(seconds: int) -> str:
    """
    Format seconds into human-readable duration
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds}s" if remaining_seconds else f"{minutes}m"
    elif seconds < 86400:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{hours}h {remaining_minutes}m" if remaining_minutes else f"{hours}h"
    else:
        days = seconds // 86400
        remaining_hours = (seconds % 86400) // 3600
        return f"{days}d {remaining_hours}h" if remaining_hours else f"{days}d"


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Decorator for retrying functions with exponential backoff
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            "Function failed after max retries",
                            function=func.__name__,
                            attempts=attempt + 1,
                            error=str(e)
                        )
                        raise
                    
                    delay = base_delay * (backoff_factor ** attempt)
                    logger.warning(
                        "Function failed, retrying",
                        function=func.__name__,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            "Function failed after max retries",
                            function=func.__name__,
                            attempts=attempt + 1,
                            error=str(e)
                        )
                        raise
                    
                    delay = base_delay * (backoff_factor ** attempt)
                    logger.warning(
                        "Function failed, retrying",
                        function=func.__name__,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def performance_monitor(threshold: float = 1.0):
    """
    Decorator to monitor function performance
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration > threshold:
                    logger.warning(
                        "Slow function execution",
                        function=func.__name__,
                        duration=duration,
                        threshold=threshold,
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys())
                    )
                else:
                    logger.debug(
                        "Function executed",
                        function=func.__name__,
                        duration=duration
                    )
                
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    duration=duration,
                    error=str(e)
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration > threshold:
                    logger.warning(
                        "Slow function execution",
                        function=func.__name__,
                        duration=duration,
                        threshold=threshold,
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys())
                    )
                else:
                    logger.debug(
                        "Function executed",
                        function=func.__name__,
                        duration=duration
                    )
                
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    duration=duration,
                    error=str(e)
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data against JSON schema
    """
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=schema)
        return {"valid": True, "errors": []}
    except ImportError:
        logger.warning("jsonschema not available for validation")
        return {"valid": True, "errors": ["Validation skipped - jsonschema not installed"]}
    except jsonschema.ValidationError as e:
        return {"valid": False, "errors": [str(e)]}
    except Exception as e:
        return {"valid": False, "errors": [f"Validation error: {str(e)}"]}


def generate_qr_code(data: str, size: int = 10) -> Optional[str]:
    """
    Generate QR code as base64-encoded PNG
    """
    try:
        import qrcode
        from PIL import Image
        import io
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=size,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except ImportError:
        logger.warning("qrcode library not available")
        return None
    except Exception as e:
        logger.error("QR code generation failed", error=str(e))
        return None


class DataValidator:
    """Advanced data validation utilities"""
    
    @staticmethod
    def validate_water_quality_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate water quality data"""
        errors = []
        warnings = []
        
        # pH validation
        ph = data.get('ph_level')
        if ph is not None:
            if not isinstance(ph, (int, float)) or ph < 0 or ph > 14:
                errors.append("pH level must be between 0 and 14")
            elif ph < 6.5 or ph > 8.5:
                warnings.append("pH level outside WHO recommended range (6.5-8.5)")
        
        # Turbidity validation
        turbidity = data.get('turbidity')
        if turbidity is not None:
            if not isinstance(turbidity, (int, float)) or turbidity < 0:
                errors.append("Turbidity must be a non-negative number")
            elif turbidity > 5:
                warnings.append("High turbidity detected (>5 NTU)")
        
        # Temperature validation
        temp = data.get('temperature')
        if temp is not None:
            if not isinstance(temp, (int, float)) or temp < -10 or temp > 60:
                errors.append("Temperature must be between -10°C and 60°C")
        
        # Location validation
        location = data.get('location', {})
        if 'latitude' in location and 'longitude' in location:
            if not validate_coordinates(location['latitude'], location['longitude']):
                errors.append("Coordinates outside valid region")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    @staticmethod
    def validate_health_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate health data"""
        errors = []
        warnings = []
        
        # Age validation
        age = data.get('age')
        if age is not None:
            if not isinstance(age, int) or age < 0 or age > 120:
                errors.append("Age must be between 0 and 120")
        
        # Fever temperature validation
        fever_temp = data.get('fever_temperature')
        if fever_temp is not None:
            if not isinstance(fever_temp, (int, float)) or fever_temp < 35 or fever_temp > 45:
                errors.append("Fever temperature must be between 35°C and 45°C")
            elif fever_temp > 39:
                warnings.append("High fever detected (>39°C)")
        
        # Symptoms validation
        symptoms = data.get('symptoms', [])
        if not isinstance(symptoms, list) or len(symptoms) == 0:
            errors.append("At least one symptom must be specified")
        
        # High-risk symptom detection
        high_risk_symptoms = [
            'severe_diarrhea', 'bloody_stool', 'severe_dehydration',
            'high_fever', 'persistent_vomiting'
        ]
        if any(symptom in symptoms for symptom in high_risk_symptoms):
            warnings.append("High-risk symptoms detected - immediate attention required")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


# Regional language mappings for alerts (enhanced with more languages)
ALERT_MESSAGES = {
    "en": {
        "water_contamination": "Water contamination detected in your area. Please boil water before use.",
        "disease_outbreak": "Disease outbreak alert. Please maintain hygiene and contact nearest health center.",
        "high_risk": "High risk area identified. Take immediate precautions.",
        "maintenance": "Water system maintenance scheduled. Alternative water sources available.",
        "general": "Health advisory: Please follow recommended safety guidelines."
    },
    "hi": {
        "water_contamination": "आपके क्षेत्र में जल प्रदूषण का पता चला है। कृपया उपयोग से पहले पानी उबालें।",
        "disease_outbreak": "बीमारी के प्रकोप की चेतावनी। कृपया स्वच्छता बनाए रखें और निकटतम स्वास्थ्य केंद्र से संपर्क करें।",
        "high_risk": "उच्च जोखिम वाला क्षेत्र चिह्नित। तत्काल सावधानी बरतें।",
        "maintenance": "जल प्रणाली रखरखाव निर्धारित। वैकल्पिक जल स्रोत उपलब्ध।",
        "general": "स्वास्थ्य सलाह: कृपया अनुशंसित सुरक्षा दिशानिर्देशों का पालन करें।"
    },
    "as": {
        "water_contamination": "আপোনাৰ এলেকাত পানী প্ৰদূষণ ধৰা পৰিছে। ব্যৱহাৰৰ আগতে পানী উতলাই লওক।",
        "disease_outbreak": "ৰোগৰ প্ৰাদুৰ্ভাৱৰ সতৰ্কতা। অনুগ্ৰহ কৰি স্বাস্থ্যবিধি বজাই ৰাখক আৰু নিকটতম স্বাস্থ্য কেন্দ্ৰৰ সৈতে যোগাযোগ কৰক।",
        "high_risk": "উচ্চ বিপদৰ এলেকা চিনাক্ত। তৎক্ষণাৎ সাৱধানতা অৱলম্বন কৰক।",
        "maintenance": "পানী প্ৰণালী ৰক্ষণাবেক্ষণ নিৰ্ধাৰিত। বিকল্প পানী উৎস উপলব্ধ।",
        "general": "স্বাস্থ্য পৰামৰ্শ: অনুগ্ৰহ কৰি পৰামৰ্শিত সুৰক্ষা নিৰ্দেশাৱলী অনুসৰণ কৰক।"
    },
    "bn": {
        "water_contamination": "আপনার এলাকায় পানি দূষণ শনাক্ত হয়েছে। ব্যবহারের আগে পানি ফুটিয়ে নিন।",
        "disease_outbreak": "রোগ প্রাদুর্ভাবের সতর্কতা। স্বাস্থ্যবিধি মেনে চলুন এবং নিকটতম স্বাস্থ্য কেন্দ্রে যোগাযোগ করুন।",
        "high_risk": "উচ্চ ঝুঁকিপূর্ণ এলাকা চিহ্নিত। অবিলম্বে সতর্কতা অবলম্বন করুন।",
        "maintenance": "পানি ব্যবস্থা রক্ষণাবেক্ষণ নির্ধারিত। বিকল্প পানির উৎস উপলব্ধ।",
        "general": "স্বাস্থ্য পরামর্শ: প্রস্তাবিত নিরাপত্তা নির্দেশিকা অনুসরণ করুন।"
    }
}


# Utility functions for common calculations
def calculate_bmi(weight_kg: float, height_m: float) -> Dict[str, Any]:
    """Calculate BMI and return category"""
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("Weight and height must be positive numbers")
    
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
        color = "#3498db"  # Blue
    elif bmi < 25:
        category = "Normal"
        color = "#2ecc71"  # Green
    elif bmi < 30:
        category = "Overweight" 
        color = "#f39c12"  # Orange
    else:
        category = "Obese"
        color = "#e74c3c"  # Red
    
    return {
        "bmi": round(bmi, 1),
        "category": category,
        "color": color,
        "healthy_range": "18.5 - 24.9"
    }


def calculate_water_quality_index(parameters: Dict[str, float]) -> Dict[str, Any]:
    """Calculate water quality index from multiple parameters"""
    # Simplified WQI calculation (actual implementation would be more complex)
    scores = {}
    weights = {
        'ph_level': 0.25,
        'turbidity': 0.25,
        'dissolved_oxygen': 0.20,
        'bacterial_count': 0.30
    }
    
    # pH scoring (ideal: 7.0)
    ph = parameters.get('ph_level', 7.0)
    if 6.5 <= ph <= 8.5:
        scores['ph_level'] = 100 - abs(ph - 7.0) * 10
    else:
        scores['ph_level'] = max(0, 50 - abs(ph - 7.0) * 20)
    
    # Turbidity scoring (lower is better)
    turbidity = parameters.get('turbidity', 0)
    scores['turbidity'] = max(0, 100 - turbidity * 20)
    
    # Dissolved oxygen scoring (higher is better, up to saturation)
    do = parameters.get('dissolved_oxygen', 8.0)
    scores['dissolved_oxygen'] = min(100, do * 12.5)
    
    # Bacterial count scoring (lower is better)
    bacteria = parameters.get('bacterial_count', 0)
    scores['bacterial_count'] = max(0, 100 - bacteria * 10)
    
    # Calculate weighted average
    wqi = sum(scores[param] * weights.get(param, 0) for param in scores)
    
    # Determine quality level
    if wqi >= 90:
        quality = "Excellent"
        color = "#2ecc71"
    elif wqi >= 70:
        quality = "Good"
        color = "#3498db"
    elif wqi >= 50:
        quality = "Fair"
        color = "#f39c12"
    elif wqi >= 25:
        quality = "Poor"
        color = "#e67e22"
    else:
        quality = "Very Poor"
        color = "#e74c3c"
    
    return {
        "wqi": round(wqi, 1),
        "quality": quality,
        "color": color,
        "component_scores": scores,
        "parameters_used": list(scores.keys())
    }


# Initialize data validator
data_validator = DataValidator()