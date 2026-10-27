import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)


def get_current_time(timezone_name: str = "UTC") -> str:
    """Read the system clock in the requested timezone, falling back to UTC.

    Args:
        timezone_name: IANA timezone selected by the calling session.

    Returns:
        JSON with the current date/time, UTC offset, timezone and weekday.
    """
    try:
        display_timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        logger.warning(
            "Invalid timezone %r; using UTC for the time tool", timezone_name
        )
        display_timezone = timezone.utc
        timezone_name = "UTC"
    now = datetime.now(display_timezone)
    return json.dumps(
        {
            "datetime": now.isoformat(sep=" ", timespec="seconds"),
            "timezone": timezone_name,
            "weekday": now.strftime("%A"),
        },
        ensure_ascii=False,
    )
