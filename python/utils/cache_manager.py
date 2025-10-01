#!/usr/bin/env python3
"""
Cache Manager Module for ML MCP System
Implements caching for improved performance
"""

import pickle
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from functools import wraps
import sys
import warnings
warnings.filterwarnings('ignore')


class CacheManager:
    """Manage caching for expensive operations"""

    def __init__(self, cache_dir: str = "temp/cache", ttl_seconds: int = 3600):
        """
        Initialize cache manager

        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'evictions': 0
        }

    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate unique cache key from function name and arguments"""
        # Create a string representation of arguments
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"

        # Hash to create fixed-length key
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key"""
        return self.cache_dir / f"{key}.cache"

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            self.stats['misses'] += 1
            return None

        try:
            # Check if cache is expired
            file_age = time.time() - cache_path.stat().st_mtime
            if file_age > self.ttl_seconds:
                cache_path.unlink()  # Delete expired cache
                self.stats['evictions'] += 1
                self.stats['misses'] += 1
                return None

            # Load cached data
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            self.stats['hits'] += 1
            print(f"Cache HIT: {key[:8]}...", file=sys.stderr)
            return data

        except Exception as e:
            print(f"Cache read error: {e}", file=sys.stderr)
            self.stats['misses'] += 1
            return None

    def set(self, key: str, value: Any):
        """
        Store value in cache

        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)

            self.stats['writes'] += 1
            print(f"Cache WRITE: {key[:8]}...", file=sys.stderr)

        except Exception as e:
            print(f"Cache write error: {e}", file=sys.stderr)

    def clear(self):
        """Clear all cache entries"""
        count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
            count += 1

        print(f"Cleared {count} cache entries", file=sys.stderr)

    def clear_expired(self):
        """Clear only expired cache entries"""
        count = 0
        current_time = time.time()

        for cache_file in self.cache_dir.glob("*.cache"):
            file_age = current_time - cache_file.stat().st_mtime
            if file_age > self.ttl_seconds:
                cache_file.unlink()
                count += 1

        self.stats['evictions'] += count
        print(f"Cleared {count} expired cache entries", file=sys.stderr)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0

        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'writes': self.stats['writes'],
            'evictions': self.stats['evictions'],
            'hit_rate_percent': round(hit_rate, 2),
            'total_entries': len(cache_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }


# Global cache instance
_global_cache = None


def get_cache(cache_dir: str = "temp/cache", ttl_seconds: int = 3600) -> CacheManager:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager(cache_dir, ttl_seconds)
    return _global_cache


def cached(ttl_seconds: int = 3600):
    """
    Decorator to cache function results

    Args:
        ttl_seconds: Time-to-live for cache entry

    Example:
        @cached(ttl_seconds=1800)
        def expensive_computation(data):
            # ... long computation
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache(ttl_seconds=ttl_seconds)

            # Generate cache key
            cache_key = cache._generate_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result)

            return result

        return wrapper
    return decorator


class ResultCache:
    """Simple in-memory cache for results"""

    def __init__(self, max_size: int = 100):
        """
        Initialize result cache

        Args:
            max_size: Maximum number of entries to keep
        """
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """Set value in cache"""
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = value
        self.access_times[key] = time.time()

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()

    def size(self) -> int:
        """Get number of cached entries"""
        return len(self.cache)


class ModelCache:
    """Cache for ML models"""

    def __init__(self, cache_dir: str = "temp/models"):
        """
        Initialize model cache

        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model_id: str, model: Any, metadata: Optional[Dict] = None):
        """
        Save model to cache

        Args:
            model_id: Unique identifier for the model
            model: Model object to cache
            metadata: Optional metadata about the model
        """
        model_path = self.cache_dir / f"{model_id}.pkl"
        metadata_path = self.cache_dir / f"{model_id}.json"

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save metadata
        if metadata:
            metadata['cached_at'] = time.time()
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"Model cached: {model_id}", file=sys.stderr)

    def load_model(self, model_id: str) -> Optional[Any]:
        """
        Load model from cache

        Args:
            model_id: Unique identifier for the model

        Returns:
            Cached model or None if not found
        """
        model_path = self.cache_dir / f"{model_id}.pkl"

        if not model_path.exists():
            return None

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            print(f"Model loaded from cache: {model_id}", file=sys.stderr)
            return model

        except Exception as e:
            print(f"Error loading cached model: {e}", file=sys.stderr)
            return None

    def get_metadata(self, model_id: str) -> Optional[Dict]:
        """Get model metadata"""
        metadata_path = self.cache_dir / f"{model_id}.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def list_models(self) -> list:
        """List all cached models"""
        models = []
        for model_file in self.cache_dir.glob("*.pkl"):
            model_id = model_file.stem
            metadata = self.get_metadata(model_id)
            models.append({
                'model_id': model_id,
                'size_mb': model_file.stat().st_size / (1024 * 1024),
                'modified': model_file.stat().st_mtime,
                'metadata': metadata
            })

        return models

    def clear_old_models(self, max_age_days: int = 7):
        """Clear models older than specified days"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        count = 0

        for model_file in self.cache_dir.glob("*.pkl"):
            file_age = current_time - model_file.stat().st_mtime
            if file_age > max_age_seconds:
                model_id = model_file.stem
                model_file.unlink()

                # Also delete metadata
                metadata_path = self.cache_dir / f"{model_id}.json"
                if metadata_path.exists():
                    metadata_path.unlink()

                count += 1

        print(f"Cleared {count} old models", file=sys.stderr)


def main():
    """CLI interface for cache management"""
    if len(sys.argv) < 2:
        print("Usage: python cache_manager.py <action>")
        print("Actions: stats, clear, clear_expired, list_models")
        sys.exit(1)

    action = sys.argv[1]

    try:
        if action == 'stats':
            cache = get_cache()
            result = cache.get_stats()
        elif action == 'clear':
            cache = get_cache()
            cache.clear()
            result = {'success': True, 'message': 'Cache cleared'}
        elif action == 'clear_expired':
            cache = get_cache()
            cache.clear_expired()
            result = {'success': True, 'message': 'Expired entries cleared'}
        elif action == 'list_models':
            model_cache = ModelCache()
            result = {'models': model_cache.list_models()}
        else:
            result = {'error': f'Unknown action: {action}'}

        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()