"""Shared model-facing chat history formatting."""

from copy import deepcopy
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def format_prompt_history(
    messages: list[dict], timezone_name: str | None
) -> list[dict]:
    """Format chat record copies consistently for judgment and generation.

    Args:
        messages: Stored chat records with original identities and timestamps.
        timezone_name: AstrBot session timezone; missing or invalid values use UTC.

    Returns:
        Independent records with readable timestamps and compact display IDs.
    """
    try:
        display_timezone = ZoneInfo(timezone_name or "UTC")
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        display_timezone = timezone.utc
    records = []
    for message in messages:
        record = {}
        for key, value in message.items():
            if key in {"platform_message_id", "is_new", "sent_to_agent"}:
                continue
            record[key] = deepcopy(value)
        if "timestamp" in record:
            try:
                timestamp = datetime.fromtimestamp(
                    record["timestamp"], tz=display_timezone
                )
                record["timestamp"] = timestamp.isoformat(sep=" ", timespec="seconds")
            except (TypeError, ValueError, OverflowError, OSError):
                record["timestamp"] = None
        if record.get("sender_id") is not None:
            record["sender_id"] = str(record["sender_id"])[-8:]
        records.append(record)
    return records
