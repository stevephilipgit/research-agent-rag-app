import hashlib
import time

# in-memory cache
response_cache = {}
retrieval_cache = {}
cache_time = {}

TTL = 300  # 5 minutes


def normalize_query(query: str):
    return (query or "").strip().lower()


def get_cache_key(query: str):
    q = normalize_query(query)
    return hashlib.md5(q.encode()).hexdigest()


def _time_key(prefix: str, key: str) -> str:
    return f"{prefix}:{key}"


def is_valid(prefix: str, key: str) -> bool:
    tkey = _time_key(prefix, key)
    return tkey in cache_time and (time.time() - cache_time[tkey] < TTL)


def get_cached_response(query):
    key = get_cache_key(query)
    if key in response_cache and is_valid("response", key):
        print("CACHE HIT: response")
        return response_cache.get(key)
    print("CACHE MISS: response")
    return None


def set_cached_response(query, response):
    key = get_cache_key(query)
    response_cache[key] = response
    cache_time[_time_key("response", key)] = time.time()


def get_cached_retrieval(query):
    key = get_cache_key(query)
    if key in retrieval_cache and is_valid("retrieval", key):
        print("CACHE HIT: retrieval")
        return retrieval_cache.get(key)
    print("CACHE MISS: retrieval")
    return None


def set_cached_retrieval(query, docs):
    key = get_cache_key(query)
    retrieval_cache[key] = docs
    cache_time[_time_key("retrieval", key)] = time.time()
