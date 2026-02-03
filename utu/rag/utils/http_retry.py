"""HTTP request retry utility for RAG services.

Provides a reusable HTTP POST request function with automatic retry logic
for handling transient failures (502, 503, timeouts, connection errors).
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def make_request_with_retry(
    url: str,
    json_data: dict,
    timeout: int = 60,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    logger_instance: Optional[logging.Logger] = None,
) -> dict:
    """Make HTTP POST request with retry logic for transient failures.

    This function automatically retries on common transient HTTP errors:
    - 502 Bad Gateway (service starting up)
    - 503 Service Unavailable
    - Request timeouts
    - Connection errors

    Other HTTP errors (4xx, 5xx except 502/503) are raised immediately without retry.

    Args:
        url: Request URL
        json_data: JSON payload to send
        timeout: Request timeout in seconds (default: 60)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 2.0)
        logger_instance: Optional logger instance (uses module logger if None)

    Returns:
        Response JSON data

    Raises:
        requests.exceptions.HTTPError: For non-retryable HTTP errors (e.g., 4xx)
        Exception: If all retry attempts failed

    Example:
        >>> response = make_request_with_retry(
        ...     url="http://localhost:8081/embed_query",
        ...     json_data={"query": "test"},
        ...     timeout=30,
        ...     max_retries=5,
        ... )
    """
    log = logger_instance or logger
    last_exception = None

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=json_data, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            last_exception = e

            if status_code == 502:
                log.warning(
                    f"Attempt {attempt + 1}/{max_retries}: 502 Bad Gateway. "
                    f"Service may be starting up. Retrying in {retry_delay}s..."
                )
            elif status_code == 503:
                log.warning(
                    f"Attempt {attempt + 1}/{max_retries}: 503 Service Unavailable. "
                    f"Retrying in {retry_delay}s..."
                )
            else:
                # Non-retryable HTTP errors (4xx, other 5xx)
                log.error(f"HTTP Error {status_code}: {str(e)}")
                raise

        except requests.exceptions.Timeout as e:
            last_exception = e
            log.warning(
                f"Attempt {attempt + 1}/{max_retries}: Request timeout. "
                f"Retrying in {retry_delay}s..."
            )

        except requests.exceptions.ConnectionError as e:
            last_exception = e
            log.warning(
                f"Attempt {attempt + 1}/{max_retries}: Connection error. "
                f"Retrying in {retry_delay}s..."
            )

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    log.error(f"All {max_retries} attempts failed")
    raise last_exception
