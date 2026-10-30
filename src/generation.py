"""Prepare Agent history and prompts without owning the framework pipeline."""

import json
import re

from astrbot.api.event import AstrMessageEvent
from astrbot.core.agent.message import get_checkpoint_id
from astrbot.core.db.po import Conversation

from .history import ChatHistory, boundary_group
from .settings import Settings
from .utils.history_formatting import format_prompt_history

AGENT_BATCH_PATTERN = re.compile(
    r"(?:^|\n\n)New batch record IDs: (?P<batch_ids>\[(?:\d+(?:, \d+)*)?\])"
    r"\. Snapshot ends at record (?P<cursor>\d+)\.\n"
)
TURN_CHECKPOINT_PREFIX = "mingle:turn:"
CURSOR_CHECKPOINT_PREFIX = "mingle:cursor:"


def agent_history_cursor(messages: list[dict]) -> int:
    """Read the maximum saved snapshot, including a post-deletion watermark.

    Args:
        messages: Framework-saved history, without modifying its messages.

    Returns:
        Latest Mingle snapshot cursor, or zero when no cursor is present.
    """
    cursor = 0
    for message in messages:
        checkpoint = get_checkpoint_id(message)
        if checkpoint and checkpoint.startswith(CURSOR_CHECKPOINT_PREFIX):
            suffix = checkpoint[len(CURSOR_CHECKPOINT_PREFIX) :]
            if suffix.isdecimal():
                cursor = max(cursor, int(suffix))
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
            continue
        for text in texts:
            match = AGENT_BATCH_PATTERN.search(text)
            if match:
                cursor = max(cursor, int(match["cursor"]))
    return cursor


def remove_message_turns(messages: list[dict], record: dict) -> tuple[list[dict], int]:
    """Remove complete new checkpoint-linked requests for one stored message.

    Args:
        messages: Current native conversation history with retained checkpoints.
        record: Session-authorized Mingle row, including its original source key.

    Returns:
        Edited history and number of removed requests. Only new Mingle
        checkpoints with a retained request header are eligible; legacy and
        summarized history is never guessed. Requests sharing a managed run ID
        are removed together, including recovery. Inputs are not mutated.

    Raises:
        ValueError: If a linked request has lost its identifiable input boundary.
    """
    turns = []
    selected_runs = set()
    segment_start = 0
    for end, message in enumerate(messages):
        checkpoint = get_checkpoint_id(message)
        if not checkpoint:
            continue
        if not checkpoint.startswith(TURN_CHECKPOINT_PREFIX):
            segment_start = end + 1
            continue
        run_id, separator, turn_id = checkpoint[
            len(TURN_CHECKPOINT_PREFIX) :
        ].partition(":")
        if not separator or not run_id or not turn_id:
            segment_start = end + 1
            continue
        request_start = None
        request_match = None
        request_text = ""
        for index in range(segment_start, end):
            candidate = messages[index]
            if candidate.get("role") != "user":
                continue
            content = candidate.get("content")
            if isinstance(content, str):
                texts = [content]
            elif isinstance(content, list):
                texts = [
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                ]
            else:
                continue
            for text in texts:
                match = AGENT_BATCH_PATTERN.search(text)
                if match:
                    request_start = index
                    request_match = match
                    request_text = text
        segment_start = end + 1
        turns.append((request_start, end + 1, run_id))
        if request_start is None:
            continue
        if record["role"] == "bot":
            matches = record["source_key"].startswith(f"bot:{turn_id}:sent:")
        else:
            batch_ids = json.loads(request_match["batch_ids"])
            matches = record["id"] in batch_ids
            if not matches:
                try:
                    records, _ = json.JSONDecoder().raw_decode(
                        request_text[request_match.end() :]
                    )
                except ValueError:
                    records = []
                if isinstance(records, list):
                    matches = any(
                        isinstance(item, dict) and item.get("id") == record["id"]
                        for item in records
                    )
        if matches:
            selected_runs.add(run_id)
    ranges = []
    for start, end, run_id in turns:
        if run_id not in selected_runs:
            continue
        if start is None:
            raise ValueError("A linked Agent request has no retained input boundary")
        ranges.append((start, end))
    if not ranges:
        return messages, 0
    edited = []
    previous_end = 0
    for start, end in ranges:
        edited.extend(messages[previous_end:start])
        previous_end = end
    edited.extend(messages[previous_end:])
    # A leading checkpoint is internal metadata: the native runner ignores it
    # when binding history. The next saved real request carries the cursor again.
    cursor = agent_history_cursor(messages)
    while edited:
        checkpoint = get_checkpoint_id(edited[0])
        if not checkpoint or not checkpoint.startswith(CURSOR_CHECKPOINT_PREFIX):
            break
        edited.pop(0)
    edited.insert(
        0,
        {
            "role": "_checkpoint",
            "content": {"id": f"{CURSOR_CHECKPOINT_PREFIX}{cursor}"},
        },
    )
    return edited, len(ranges)


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
    cursor = agent_history_cursor(saved)
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
