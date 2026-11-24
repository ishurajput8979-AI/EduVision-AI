"""Cache management for performance optimization."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

try:
    import redis
except ImportError:
    redis = None


class CacheManager:
    """Manage caching of expensive operations using Redis or in-memory fallback.
    
    Features:
    - Automatic TTL (time-to-live) for cache entries
    - Pattern-based cache invalidation
    - Fallback to in-memory cache if Redis unavailable
    - Automatic serialization/deserialization
    
    Setup for Production:
    1. Install Redis: brew install redis (macOS) or apt-get install redis-server (Linux)
    2. Set environment variable:
       export CACHE_URL="redis://localhost:6379"
    """
    
    def __init__(self, redis_url: str | None = None):
        """Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL (uses env var CACHE_URL if not provided)
        """
        if redis_url is None:
            redis_url = os.environ.get('CACHE_URL', '')
        
        self.redis_client = None
        self.cache_enabled = False
        self.memory_cache = {}
        self.cache_timestamps = {}
        
        # Try to connect to Redis
        if redis and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                self.cache_enabled = 'redis'
                print("Redis cache initialized")
            except Exception as e:
                print(f"Redis connection failed: {e}")
                print("Falling back to in-memory cache")
                self.cache_enabled = 'memory'
        else:
            self.cache_enabled = 'memory'
            print("Redis not available, using in-memory cache")
    
    def get(self, key: str) -> Any | None:
        """Retrieve cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        try:
            if self.cache_enabled == 'redis':
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
                return None
            
            elif self.cache_enabled == 'memory':
                if key in self.memory_cache:
                    # Check if expired
                    timestamp = self.cache_timestamps.get(key, 0)
                    if timestamp > datetime.now().timestamp():
                        return self.memory_cache[key]
                    else:
                        # Expired, remove it
                        del self.memory_cache[key]
                        del self.cache_timestamps[key]
                return None
        
        except Exception as e:
            print(f"Cache GET error: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Cache a value with TTL.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl_seconds: Time-to-live in seconds (default: 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            serialized = json.dumps(value, default=str)
            
            if self.cache_enabled == 'redis':
                self.redis_client.setex(key, ttl_seconds, serialized)
                return True
            
            elif self.cache_enabled == 'memory':
                self.memory_cache[key] = value
                self.cache_timestamps[key] = (
                    datetime.now() + timedelta(seconds=ttl_seconds)
                ).timestamp()
                return True
        
        except Exception as e:
            print(f"Cache SET error: {e}")
        
        return False
    
    def invalidate(self, pattern: str = '*') -> int:
        """Clear cache keys matching pattern.
        
        Args:
            pattern: Pattern to match (e.g., 'attendance:*', 'embedding:student_*')
            
        Returns:
            Number of keys deleted
        """
        try:
            if self.cache_enabled == 'redis':
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
            
            elif self.cache_enabled == 'memory':
                keys_to_delete = [k for k in self.memory_cache.keys() if self._match_pattern(k, pattern)]
                for k in keys_to_delete:
                    del self.memory_cache[k]
                    if k in self.cache_timestamps:
                        del self.cache_timestamps[k]
                return len(keys_to_delete)
        
        except Exception as e:
            print(f"Cache INVALIDATE error: {e}")
        
        return 0
    
    def clear(self) -> bool:
        """Clear all cached values.
        
        Returns:
            True if successful
        """
        try:
            if self.cache_enabled == 'redis':
                self.redis_client.flushdb()
                return True
            elif self.cache_enabled == 'memory':
                self.memory_cache.clear()
                self.cache_timestamps.clear()
                return True
        except Exception as e:
            print(f"Cache CLEAR error: {e}")
        
        return False
    
    def set_if_not_exists(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value only if key doesn't exist (atomic operation).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
            
        Returns:
            True if set successfully (was not present), False if already existed
        """
        if self.get(key) is not None:
            return False
        
        return self.set(key, value, ttl_seconds)
    
    def increment(self, key: str, delta: int = 1, ttl_seconds: int = 3600) -> int:
        """Increment a counter value.
        
        Args:
            key: Cache key
            delta: Amount to increment by
            ttl_seconds: TTL for new keys
            
        Returns:
            New counter value
        """
        try:
            if self.cache_enabled == 'redis':
                return self.redis_client.incrby(key, delta)
            elif self.cache_enabled == 'memory':
                current = self.memory_cache.get(key, 0)
                new_value = current + delta
                self.set(key, new_value, ttl_seconds)
                return new_value
        except Exception as e:
            print(f"Cache INCREMENT error: {e}")
        
        return 0
    
    @staticmethod
    def _match_pattern(text: str, pattern: str) -> bool:
        """Simple glob pattern matching.
        
        Args:
            text: Text to match
            pattern: Pattern with * wildcards
            
        Returns:
            True if matches
        """
        import fnmatch
        return fnmatch.fnmatch(text, pattern)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        stats = {
            'enabled': self.cache_enabled,
            'memory_keys': len(self.memory_cache) if self.cache_enabled == 'memory' else 'N/A'
        }
        
        if self.cache_enabled == 'redis':
            try:
                redis_info = self.redis_client.info()
                stats['redis_keys'] = redis_info.get('db0', {}).get('keys', 0)
                stats['redis_memory'] = redis_info.get('used_memory_human', 'N/A')
            except Exception:
                pass
        
        return stats


# ============ USAGE EXAMPLES ============

"""
# Initialize cache
cache = CacheManager()

# Cache embedding index (expensive operation)
cache_key = f"embedding_index_{teacher_id}_{assignment_id}"
index = cache.get(cache_key)

if not index:
    # Expensive: build from database
    print("Building embedding index...")
    index = build_embedding_index(...)
    cache.set(cache_key, index, ttl_seconds=3600)  # Cache for 1 hour
else:
    print("Using cached embedding index")

# Invalidate cache when attendance is marked
cache.invalidate(f"embedding_index_{teacher_id}_*")

# Cache student attendance data
student_key = f"student:attendance:{erp_id}"
cache.set(student_key, attendance_data, ttl_seconds=300)  # Cache for 5 min

# Clear all cache on system update
cache.clear()

# Get cache statistics
stats = cache.get_stats()
print(stats)
"""


# Create default singleton instance
cache_manager = CacheManager()
