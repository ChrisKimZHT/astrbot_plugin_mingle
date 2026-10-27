import json

from astrbot.api.event import AstrMessageEvent

from ..history import ChatHistory


async def chat_history(
    event: AstrMessageEvent,
    history: ChatHistory,
    *,
    available: bool,
    limit: int = 30,
    before_id: int = 0,
    after_id: int = 0,
    keyword: str = "",
) -> str:
    """Query chronological records scoped to a managed execution's session.

    Args:
        event: Event defining the session and managed execution status.
        history: Plugin-owned history store.
        available: Whether the plugin's storage is ready for use.
        limit: Maximum number of returned records, from 1 to 100.
        before_id: Exclusive upper record ID, or zero for no upper bound.
        after_id: Exclusive lower record ID, or zero for no lower bound.
        keyword: Optional literal text filter.

    Returns:
        JSON records and cursor boundaries, or an explicit validation error.
    """
    if not available or not event.get_extra("mingle_managed", False):
        return json.dumps({"error": "Mingle history is unavailable in this execution"})
    if limit < 1 or limit > 100 or before_id < 0 or after_id < 0:
        return json.dumps({"error": "Invalid limit or cursor"})
    records = await history.get(
        event.unified_msg_origin, limit, before_id, after_id, keyword
    )
    return json.dumps(
        {
            "messages": records,
            "oldest_id": records[0]["id"] if records else None,
            "newest_id": records[-1]["id"] if records else None,
        },
        ensure_ascii=False,
    )
