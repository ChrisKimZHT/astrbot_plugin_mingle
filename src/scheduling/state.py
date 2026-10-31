"""Tickets and runtime state owned by a session coordinator."""

import asyncio
from dataclasses import dataclass, field

from astrbot.api.event import AstrMessageEvent

from . import timing


class CompletionSignal(asyncio.Event):
    """Wake coordination when a handler has finished saving framework history."""

    def __init__(self, changed: asyncio.Event) -> None:
        super().__init__()
        self.changed = changed

    def set(self) -> None:
        super().set()
        self.changed.set()


@dataclass
class PendingMessage:
    """Keep an original handler alive through admission and pipeline completion.

    Only the newest ticket in an admitted batch receives record IDs via ready;
    the others receive None. Internal proactive tickets use record_id=0 without
    creating a stored row. Their ready result contains only real backlog IDs.
    """

    event: AstrMessageEvent
    record_id: int
    forced: bool
    ready: asyncio.Future[list[int] | None]
    received_at: float
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    backlog_sleep_revision: int | None = None

    def release(self) -> None:
        """Release a waiting handler without admitting another Agent turn."""
        if not self.ready.done():
            self.ready.set_result(None)


def waiting_messages(messages: list[PendingMessage]) -> list[PendingMessage]:
    """Release stopped events and retain unresolved admission tickets.

    Args:
        messages: Pending or approved tickets; the input list is not changed.

    Returns:
        Tickets whose events remain eligible to await admission.
    """
    waiting = []
    for item in messages:
        stopped = item.event.is_stopped() or item.event.get_extra(
            "agent_stop_requested", False
        )
        if stopped:
            item.release()
        if not item.ready.done():
            waiting.append(item)
    return waiting


@dataclass
class SessionState:
    """Live queues and clocks managed by one coordinator per full session UMO.

    Background judgment and dispatch tasks return results without changing this
    state. Synchronous entry operations run atomic coordinator transactions on
    the same event loop, then notify the coordinator to recompute its deadline.
    No queue transition contains an await.
    """

    pending: list[PendingMessage] = field(default_factory=list)
    judging: list[PendingMessage] = field(default_factory=list)
    approved: list[PendingMessage] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    coordinator_task: asyncio.Task | None = None
    judgment_task: asyncio.Task | None = None
    dispatch_task: asyncio.Task | None = None
    active: PendingMessage | None = None
    active_ids: list[int] = field(default_factory=list)
    finished: dict[int, str] = field(default_factory=dict)
    last_agent_finished_at: float = float("-inf")
    last_agent_reply_at: float = field(default_factory=lambda: timing.monotonic())
    last_woken_at: float = float("-inf")
    sleep_enabled: bool = True
    sleeping: bool = False
    waking: bool = False
    sleep_revision: int = 0
    next_sleep_backlog_at: float = float("inf")
    bubbles_enabled: bool = False
    next_bubble_at: float = float("inf")
    bubble_revision: int = 0
    resume_after_rerun: bool = False
    history_editing: bool = False
    deleted_through_id: int | None = None
