import time
import logging
from config import ENABLE_RETRY

logger = logging.getLogger(__name__)

def retry_call(func, retries=2, delay=1):
    if not ENABLE_RETRY:
        return func()
        
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            logger.warning(f"Retry attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
    return None
