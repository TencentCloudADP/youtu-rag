"""Date and time utility functions for RAG system.

Provides functions for parsing and converting various date formats
to timestamp ranges, supporting:
- Standard formats: YYYY, YYYY-MM, YYYY-MM-DD
- Quarter formats: YYYY-Q1 to YYYY-Q4
- Half-year formats: YYYY-H1, YYYY-H2
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def date_to_time_range(date_str: str) -> Tuple[Optional[int], Optional[int]]:
    """Convert date string to time range (start and end timestamps).

    Supports multiple date formats:
    - YYYY (e.g., "2024") → Full year range
    - YYYY-MM (e.g., "2024-01") → Full month range
    - YYYY-MM-DD (e.g., "2024-01-15") → Single day range
    - YYYY-MM-DD HH:MM:SS (e.g., "2024-01-15 12:30:00") → Single day range (time part ignored)
    - YYYY-Q1 to YYYY-Q4 (e.g., "2024-Q3") → Quarter range
    - YYYY-H1, YYYY-H2 (e.g., "2024-H1") → Half-year range

    Args:
        date_str: Date string in one of the supported formats

    Returns:
        Tuple of (start_timestamp, end_timestamp) in Unix timestamp format.
        Returns (None, None) if the date string is invalid or cannot be parsed.

    Examples:
        >>> date_to_time_range("2024")
        (1704067200, 1735689599)  # 2024-01-01 00:00:00 to 2024-12-31 23:59:59

        >>> date_to_time_range("2024-Q3")
        (1719792000, 1727740799)  # 2024-07-01 00:00:00 to 2024-09-30 23:59:59

        >>> date_to_time_range("2024-H1")
        (1704067200, 1719791999)  # 2024-01-01 00:00:00 to 2024-06-30 23:59:59

        >>> date_to_time_range("2024-01-15 12:30:00")
        (1705276800, 1705363199)  # 2024-01-15 00:00:00 to 2024-01-15 23:59:59

        >>> date_to_time_range("invalid")
        (None, None)
    """
    if not date_str or date_str == "null":
        return None, None

    date_str = date_str.strip()

    # support：'2025-04-08 00:00:00' -> '2025-04-08'
    datetime_match = re.match(r'^(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}', date_str)
    if datetime_match:
        date_str = datetime_match.group(1)

    # Handle quarter format (YYYY-Q1 ~ YYYY-Q4)
    quarter_match = re.match(r"^(\d{4})-Q([1-4])$", date_str)
    if quarter_match:
        year = int(quarter_match.group(1))
        quarter = int(quarter_match.group(2))
        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3
        start_date = strf_to_timestamp(f"{year}-{start_month:02d}-01 00:00:00")
        if end_month == 12:
            end_date = strf_to_timestamp(f"{year}-12-31 23:59:59")
        else:
            next_month_first = datetime(year, end_month + 1, 1)
            last_day = (next_month_first - timedelta(days=1)).day
            end_date = strf_to_timestamp(f"{year}-{end_month:02d}-{last_day} 23:59:59")
        return start_date, end_date

    # Handle half-year format (YYYY-H1, YYYY-H2)
    half_match = re.match(r"^(\d{4})-H([1-2])$", date_str)
    if half_match:
        year = int(half_match.group(1))
        half = int(half_match.group(2))
        if half == 1:
            start_date = strf_to_timestamp(f"{year}-01-01 00:00:00")
            end_date = strf_to_timestamp(f"{year}-06-30 23:59:59")
        else:  # half == 2
            start_date = strf_to_timestamp(f"{year}-07-01 00:00:00")
            end_date = strf_to_timestamp(f"{year}-12-31 23:59:59")
        return start_date, end_date

    formats = [
        "%Y-%m-%d",  # 2024-01-15
        "%Y-%m",  # 2024-01
        "%Y",  # 2024
    ]

    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            if fmt == "%Y":
                start_date = strf_to_timestamp(date_obj.strftime("%Y-01-01 00:00:00"))
                end_date = strf_to_timestamp(date_obj.strftime("%Y-12-31 23:59:59"))
                return start_date, end_date
            elif fmt == "%Y-%m":
                start_date = strf_to_timestamp(date_obj.strftime("%Y-%m-01 00:00:00"))
                if date_obj.month == 12:
                    last_day = 31
                else:
                    next_month = date_obj.replace(month=date_obj.month + 1)
                    last_day = (next_month - timedelta(days=1)).day
                end_date = strf_to_timestamp(date_obj.strftime(f"%Y-%m-{last_day} 23:59:59"))
                return start_date, end_date
            else:  # %Y-%m-%d
                start_date = strf_to_timestamp(date_obj.strftime("%Y-%m-%d 00:00:00"))
                end_date = strf_to_timestamp(date_obj.strftime("%Y-%m-%d 23:59:59"))
                return start_date, end_date
        except ValueError:
            continue

    logger.warning(f"Could not parse date: {date_str}")
    return None, None


def strf_to_timestamp(date_str: str) -> int:
    """Convert date string (YYYY-MM-DD HH:MM:SS format) to Unix timestamp.

    Args:
        date_str: Date string in format "YYYY-MM-DD HH:MM:SS"

    Returns:
        Unix timestamp (seconds since epoch)

    Example:
        >>> strf_to_timestamp("2024-01-01 00:00:00")
        1704067200
    """
    return int(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp())
