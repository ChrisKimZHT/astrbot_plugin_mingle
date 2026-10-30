"""Prepare Agent history and prompts without owning the framework pipeline."""

import json
import re

from astrbot.api.event import AstrMessageEvent
from astrbot.core.db.po import Conversation

from .history import ChatHistory, boundary_group
from .settings import Settings
from .utils.history_formatting import format_prompt_history


async def load_agent_history(
    event: AstrMessageEvent,
    conversation: Conversation,
    batch_ids: list[int],
    *,
    history: ChatHistory,
    settings: Settings,
    snapshot_end: int | None = None,
) -> list[dict]:
    """Load initial context or records following the latest saved Mingle cursor.

    Args:
        event: Managed event defining the full session and group boundary.
        conversation: Bound conversation with framework-saved Agent history.
        batch_ids: Real chat record IDs claimed by the current turn.
        history: Independent Mingle message store.
        settings: Current initial-context limit, resolved by the caller.
        snapshot_end: Optional upper cursor for a proactive turn without input.

    Returns:
        Chronological records respecting the group boundary. A wake batch may
        exceed the initial-context limit so all claimed input remains visible.
    """
    saved = json.loads(conversation.history or "[]")
    if snapshot_end is None:
        snapshot_end = max(batch_ids)
    cursor = 0
    for message in saved:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            texts = [content]
        elif isinstance(content, list):
            texts = [
                part.get("text", "") for part in content if part.get("type") == "text"
            ]
        else:
            texts = []
        for text in texts:
            match = re.search(
                r"(?:^|\n\n)New batch record IDs: \[(?:\d+(?:, \d+)*)?\]\. Snapshot ends at record (\d+)\.\n",
                text,
            )
            if match:
                cursor = max(cursor, int(match.group(1)))
    if not saved or (not cursor and not batch_ids):
        initial = await history.get(
            event.unified_msg_origin,
            limit=settings.initial_context_messages,
            before_id=snapshot_end + 1,
            boundary=boundary_group(event),
        )
        if not batch_ids or not event.get_extra("mingle_woke_from_sleep", False):
            return initial
        # Include the oldest claimed wake input even outside the initial window.
        first_id = min(batch_ids)
        if initial:
            first_id = min(first_id, initial[0]["id"])
        cursor = first_id - 1
    # Existing native conversations have no Mingle cursor; start at this batch.
    if saved and not cursor and batch_ids:
        cursor = min(batch_ids) - 1
    records = []
    while True:
        page = await history.get(
            event.unified_msg_origin,
            limit=100,
            after_id=cursor,
            before_id=snapshot_end + 1,
            boundary=boundary_group(event),
        )
        records.extend(page)
        if len(page) < 100:
            return records
        cursor = page[-1]["id"]


def build_user_prompt(
    records: list[dict],
    batch_ids: list[int],
    settings: Settings,
    *,
    timezone_name: str | None,
    is_bubble: bool,
    snapshot_end: int | None,
    is_recovery: bool,
    recovery_has_new_messages: bool,
    resume_after_rerun: bool,
) -> str:
    """Build the exact user input for ordinary, proactive, rerun and recovery turns.

    Args:
        records: Stored chat rows to format on independent copies.
        batch_ids: Real record IDs owned by the current turn.
        settings: Current proactive prompt configuration.
        timezone_name: Current AstrBot session timezone, with UTC fallback.
        is_bubble: Whether a fresh internal proactive event owns this turn.
        snapshot_end: Explicit upper cursor used by proactive turns.
        is_recovery: Whether this is the single completion-protocol recovery.
        recovery_has_new_messages: Whether recovery claimed additional input.
        resume_after_rerun: Whether to prepend the fixed continuation notice.

    Returns:
        User prompt preserving the persisted cursor syntax and configured text.
    """
    prompt_history = format_prompt_history(records, timezone_name)
    if is_bubble:
        end_id = snapshot_end
    else:
        end_id = max(batch_ids)
    prompt = (
        f"New batch record IDs: {batch_ids}. Snapshot ends at record {end_id}.\n"
        + json.dumps(prompt_history, ensure_ascii=False)
    )
    if is_bubble:
        prompt = settings.random_bubble_prompt + "\n\n" + prompt
    if is_recovery:
        notice = (
            "The previous turn did not follow the completion protocol: "
            "end with exactly <finished>. Content in the final response "
            "is not sent to the user. Review messages already sent "
            "successfully and use the send_message_to_user tool to send "
            "any remaining replies. Avoid duplicate messages and do not mention this "
            "recovery run."
        )
        if recovery_has_new_messages:
            notice += (
                "\nMeanwhile, the following new messages have arrived. "
                "You may reply to them separately or together:"
            )
        prompt = notice + "\n\n" + prompt
    elif resume_after_rerun:
        prompt = (
            "The previous Agent turn was stopped because new messages corrected "
            "or redirected the task. Continue from its saved history, apply the "
            "new messages below, and reuse completed tool results when relevant.\n\n"
            + prompt
        )
    return prompt


def clean_history_images(messages: list[dict]) -> tuple[list[dict], int]:
    """Replace historical images while preserving other content and checkpoints.

    Args:
        messages: Request contexts or a decoded conversation history snapshot.

    Returns:
        Cleaned message copies and the number of image parts replaced. Original
        messages and image bytes are never modified.
    """
    cleaned = []
    removed = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            cleaned.append(message)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                parts.append(
                    {
                        "type": "text",
                        "text": "[Previous-turn image omitted. Use "
                        "mingle_view_image to reopen it if cached.]",
                    }
                )
                removed += 1
            else:
                parts.append(part)
        cleaned.append({**message, "content": parts})
    return cleaned, removed
