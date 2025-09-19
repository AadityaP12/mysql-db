import redis.asyncio as redis
import json
import pickle
from typing import Any, Optional, Dict, List
import structlog
from functools import wraps
import hashlib
import asyncio

from app.config import settings

logger = structlog.get_logger(__name__)

# Global Redis client
redis_client: Optional[redis.Redis] = None


class CacheManager:
    """Redis-based cache manager with advanced features"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = settings.CACHE_DEFAULT_TTL
        self.enabled = settings.CACHE_ENABLED

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        if not self.enabled:
            return default

        try:
            value = await self.redis.get(self._make_key(key))
            if value is None:
                return default

            # Try to deserialize JSON first, then pickle as fallback
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                try:
                    return pickle.loads(value)
                except Exception:
                    return value.decode("utf-8") if isinstance(value, bytes) else value

        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set value in cache"""
        if not self.enabled:
            return False

        try:
            # Serialize value
            if isinstance(value, (dict, list, str, int, float, bool)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = pickle.dumps(value)

            cache_key = self._make_key(key)
            ttl = ttl or self.default_ttl

            return await self.redis.set(
                cache_key, serialized_value, ex=ttl, nx=nx or None, xx=xx or None
            )

        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        if not self.enabled:
            return False
        try:
            return await self.redis.delete(self._make_key(key)) > 0
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        if not self.enabled:
            return False
        try:
            return await self.redis.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error("Cache exists check failed", key=key, error=str(e))
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        if not self.enabled:
            return False
        try:
            return await self.redis.expire(self._make_key(key), ttl)
        except Exception as e:
            logger.error("Cache expire failed", key=key, error=str(e))
            return False

    async def ttl(self, key: str) -> int:
        if not self.enabled:
            return -1
        try:
            return await self.redis.ttl(self._make_key(key))
        except Exception as e:
            logger.error("Cache TTL check failed", key=key, error=str(e))
            return -1

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        if not self.enabled:
            return None
        try:
            return await self.redis.incrby(self._make_key(key), amount)
        except Exception as e:
            logger.error("Cache increment failed", key=key, error=str(e))
            return None

    async def mget(self, keys: List[str]) -> List[Any]:
        if not self.enabled:
            return [None] * len(keys)

        try:
            cache_keys = [self._make_key(key) for key in keys]
            values = await self.redis.mget(cache_keys)

            result = []
            for value in values:
                if value is None:
                    result.append(None)
                    continue

                try:
                    result.append(json.loads(value))
                except Exception:
                    try:
                        result.append(pickle.loads(value))
                    except Exception:
                        result.append(
                            value.decode("utf-8") if isinstance(value, bytes) else value
                        )
            return result

        except Exception as e:
            logger.error("Cache mget failed", keys=keys, error=str(e))
            return [None] * len(keys)

    async def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        if not self.enabled or not mapping:
            return False
        try:
            pipe = self.redis.pipeline()
            ttl = ttl or self.default_ttl

            for key, value in mapping.items():
                if isinstance(value, (dict, list, str, int, float, bool)):
                    serialized_value = json.dumps(value)
                else:
                    serialized_value = pickle.dumps(value)

                cache_key = self._make_key(key)
                pipe.set(cache_key, serialized_value, ex=ttl)

            await pipe.execute()
            return True

        except Exception as e:
            logger.error("Cache mset failed", error=str(e))
            return False

    async def clear_pattern(self, pattern: str) -> int:
        if not self.enabled:
            return 0
        try:
            keys = await self.redis.keys(self._make_key(pattern))
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error("Cache clear pattern failed", pattern=pattern, error=str(e))
            return 0

    async def get_info(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        try:
            info = await self.redis.info()
            return {
                "enabled": True,
                "connected": True,
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error("Failed to get cache info", error=str(e))
            return {"enabled": True, "connected": False, "error": str(e)}

    def _make_key(self, key: str) -> str:
        return f"health_monitor:{settings.VERSION}:{key}"


class CacheDecorator:
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    def cached(
        self,
        key_prefix: str = None,
        ttl: int = None,
        exclude_args: List[str] = None,
        vary_on_user: bool = False,
    ):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.cache.enabled:
                    return await func(*args, **kwargs)

                cache_key = self._generate_cache_key(
                    func, key_prefix, args, kwargs, exclude_args, vary_on_user
                )

                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    logger.debug("Cache hit", function=func.__name__, key=cache_key)
                    return cached_result

                logger.debug("Cache miss", function=func.__name__, key=cache_key)
                result = await func(*args, **kwargs)

                if result is not None:
                    await self.cache.set(cache_key, result, ttl)

                return result

            return wrapper

        return decorator

    def _generate_cache_key(
        self,
        func,
        key_prefix: str,
        args: tuple,
        kwargs: dict,
        exclude_args: List[str],
        vary_on_user: bool,
    ) -> str:
        parts = [key_prefix or func.__name__]

        if vary_on_user and "current_user" in kwargs:
            user = kwargs.get("current_user", {})
            user_id = user.get("uid", "anonymous")
            parts.append(f"user:{user_id}")

        exclude_args = exclude_args or []
        start_idx = 1 if args and hasattr(args[0], func.__name__) else 0
        for i, arg in enumerate(args[start_idx:]):
            if isinstance(arg, (str, int, float, bool)):
                parts.append(f"arg{i}:{arg}")
            else:
                parts.append(f"arg{i}:{hash(str(arg))}")

        for key, value in sorted(kwargs.items()):
            if key not in exclude_args:
                if isinstance(value, (str, int, float, bool)):
                    parts.append(f"{key}:{value}")
                elif isinstance(value, dict):
                    dict_hash = hashlib.md5(
                        json.dumps(value, sort_keys=True).encode()
                    ).hexdigest()[:8]
                    parts.append(f"{key}:{dict_hash}")
                else:
                    parts.append(f"{key}:{hash(str(value))}")

        return ":".join(parts)


# Redis connection init
async def init_redis():
    global redis_client

    if not settings.CACHE_ENABLED:
        logger.info("Cache is disabled")
        return None

    try:
        # Use redis-py asyncio client correctly
        redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True
        )

        # Test connection
        await redis_client.ping()
        logger.info("Redis connection established successfully")
        return redis_client
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        redis_client = None
        return None



def get_cache() -> CacheManager:
    global redis_client
    if redis_client is None:
        class DisabledCacheManager:
            enabled = False
            async def get(self, *a, **k): return None
            async def set(self, *a, **k): return False
            async def delete(self, *a, **k): return False
            async def exists(self, *a, **k): return False
            async def expire(self, *a, **k): return False
            async def ttl(self, *a, **k): return -1
            async def increment(self, *a, **k): return None
            async def mget(self, keys): return [None]*len(keys)
            async def mset(self, *a, **k): return False
            async def clear_pattern(self, *a, **k): return 0
            async def get_info(self): return {"enabled": False}
        return DisabledCacheManager()
    return CacheManager(redis_client)


def get_cache_decorator() -> CacheDecorator:
    return CacheDecorator(get_cache())


# Example cache helpers (unchanged from your original)
class ModelCache:
    def __init__(self):
        self.cache = get_cache()
        self.ttl = settings.CACHE_USER_PROFILE_TTL
    async def get_user_profile(self, uid: str) -> Optional[Dict]:
        return await self.cache.get(f"user_profile:{uid}")
    async def set_user_profile(self, uid: str, profile: Dict) -> bool:
        return await self.cache.set(f"user_profile:{uid}", profile, self.ttl)
    async def invalidate_user_profile(self, uid: str) -> bool:
        return await self.cache.delete(f"user_profile:{uid}")


class DataCache:
    def __init__(self):
        self.cache = get_cache()
    async def get_water_quality_data(self, location_key: str) -> Optional[List]:
        return await self.cache.get(f"water_quality:{location_key}")
    async def set_water_quality_data(self, location_key: str, data: List) -> bool:
        return await self.cache.set(
            f"water_quality:{location_key}", data, settings.CACHE_WATER_QUALITY_TTL
        )
    async def get_health_data(self, location_key: str) -> Optional[List]:
        return await self.cache.get(f"health_data:{location_key}")
    async def set_health_data(self, location_key: str, data: List) -> bool:
        return await self.cache.set(
            f"health_data:{location_key}", data, settings.CACHE_DEFAULT_TTL
        )


class PredictionCache:
    def __init__(self):
        self.cache = get_cache()
        self.ttl = settings.CACHE_PREDICTIONS_TTL
    async def get_prediction(self, input_hash: str) -> Optional[Dict]:
        return await self.cache.get(f"prediction:{input_hash}")
    async def set_prediction(self, input_hash: str, prediction: Dict) -> bool:
        return await self.cache.set(f"prediction:{input_hash}", prediction, self.ttl)
    def generate_input_hash(self, prediction_input: Dict) -> str:
        input_str = json.dumps(prediction_input, sort_keys=True)
        return hashlib.md5(input_str.encode()).hexdigest()


# Initialize Redis connection on import
try:
    loop = asyncio.get_running_loop()
    loop.create_task(init_redis())
except RuntimeError:
    pass
