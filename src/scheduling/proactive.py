"""Proactive deadlines and admission, driven by the existing session loop."""

import asyncio
from typing import TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent
from astrbot.core.cron.events import CronMessageEvent
from astrbot.core.platform.astrbot_message import MessageMember
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.platform_metadata import PlatformMetadata

from . import timing
from .state import CompletionSignal, PendingMessage, waiting_messages

if TYPE_CHECKING:
    from .session import SessionCoordinator


class RandomBubbleEvent(CronMessageEvent):
    """Route a fresh internal trigger through AstrBot's ordinary pipeline.

    Unlike a replayed incoming event, this owns no user attachments, permissions,
    or message identity. Sending uses the framework's proactive event support.
    """

    def __init__(
        self,
        context,
        session: MessageSession,
        platform_meta: PlatformMetadata,
        revision: int,
        self_id: str = "astrbot",
    ) -> None:
        """Create an internal trigger without borrowing a member's event.

        Args:
            context: AstrBot context used for proactive sending.
            session: Shared group session, never supplied by a model.
            platform_meta: Current adapter metadata for normal pipeline routing.
            revision: Sampled clock generation checked again at admission.
            self_id: Last observed bot identity, or the framework cron default.
        """
        super().__init__(
            context=context,
            session=session,
            message="Mingle proactive group-chat turn",
            sender_id=self_id,
            message_type=session.message_type,
            extras={
                "mingle_bubble_revision": revision,
                "_api_key_allow_admin_role": False,
            },
        )
        self.platform_meta = platform_meta
        self.platform = platform_meta
        self.message_obj.group_id = session.session_id
        self.message_obj.sender = MessageMember("mingle_scheduler", "Mingle scheduler")
        self.is_at_or_wake_command = False


class ProactiveSchedule:
    """Keep proactive policy together without owning a competing timer loop."""

    def __init__(self, owner: "SessionCoordinator") -> None:
        self.owner = owner
        self.waiting: (
            tuple[AstrMessageEvent, int, asyncio.Future[PendingMessage | None]] | None
        ) = None

    def watch(self) -> None:
        state = self.owner.state
        if state.bubbles_enabled:
            return
        state.bubbles_enabled = True
        self.reset_deadline()
        self.owner.notify()

    def stop(self) -> asyncio.Task | None:
        """Invalidate queued events and return any dispatch task to be joined."""
        state = self.owner.state
        state.bubbles_enabled = False
        state.bubble_revision += 1
        state.next_bubble_at = float("inf")
        task = state.dispatch_task
        state.dispatch_task = None
        if task is not None:
            task.cancel()
        self.owner.notify()
        return task

    def next_deadline(self) -> float:
        """Wait for future opportunities even while busy; never poll a blocked one."""
        state = self.owner.state
        settings = self.owner.scheduler.settings
        if not settings.enabled or not settings.random_bubble_enabled:
            return float("inf")
        busy = state.active is not None or state.approved or state.judging
        cooldown_at = state.last_agent_finished_at + settings.cooldown_seconds
        deadline = float("inf")
        if state.bubbles_enabled and state.dispatch_task is None:
            due_at = max(state.next_bubble_at, cooldown_at)
            if not busy or due_at > timing.monotonic():
                deadline = due_at
        if self.waiting is not None and not busy:
            deadline = min(deadline, cooldown_at)
        return deadline

    def reset_deadline(self, anchor_at: float | None = None) -> None:
        """Invalidate old triggers and sample from a real send or quiet-hours end.

        Args:
            anchor_at: Optional monotonic quiet-hours end instead of last send.
        """
        state = self.owner.state
        if anchor_at is None:
            anchor_at = state.last_agent_reply_at
        state.bubble_revision += 1
        state.next_bubble_at = timing.bubble_deadline(
            self.owner.scheduler.settings, anchor_at
        )
        state.changed.set()

    def advance(self) -> None:
        """Consume each due proactive opportunity once, dispatching outside the loop."""
        state = self.owner.state
        task = state.dispatch_task
        if task is not None:
            if not task.done():
                return
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                self.owner.scheduler.logger.exception(
                    "Failed to queue Mingle random bubble: %s", self.owner.key
                )
            state.dispatch_task = None
        settings = self.owner.scheduler.settings
        if not state.bubbles_enabled or not settings.enabled:
            return
        if not settings.random_bubble_enabled:
            return
        deadline = max(
            state.next_bubble_at,
            state.last_agent_finished_at + settings.cooldown_seconds,
        )
        if deadline > timing.monotonic():
            return
        quiet_delay = self.owner.scheduler._quiet_seconds_remaining(self.owner.key)
        if quiet_delay > 0:
            self.reset_deadline(timing.monotonic() + quiet_delay)
            return
        if state.active is not None or state.approved or state.judging:
            return
        revision = state.bubble_revision
        state.next_bubble_at = float("inf")
        state.dispatch_task = asyncio.create_task(
            self.owner.scheduler.queue_bubble(self.owner.key, revision)
        )
        state.dispatch_task.add_done_callback(self.owner.background_finished)

    def request(
        self, event: AstrMessageEvent, revision: int
    ) -> asyncio.Future[PendingMessage | None]:
        """Request the shared slot without introducing another admission worker.

        Args:
            event: Fresh internal event with no synthetic history row.
            revision: Random clock generation captured at dispatch.
        """
        result = asyncio.get_running_loop().create_future()
        if self.waiting is not None:
            result.set_result(None)
            return result
        self.waiting = (event, revision, result)
        self.admit()
        self.owner.notify()
        return result

    def admit(self) -> None:
        """Validate a queued trigger and atomically claim pending reference input."""
        if self.waiting is None:
            return
        state = self.owner.state
        if state.history_editing:
            return
        event, revision, result = self.waiting
        if result.done():
            self.waiting = None
            return
        settings = self.owner.scheduler.settings
        invalid = (
            self.owner.scheduler.closed
            or not settings.enabled
            or not settings.random_bubble_enabled
            or revision != state.bubble_revision
            or event.is_stopped()
        )
        if invalid:
            result.set_result(None)
            self.waiting = None
            return
        quiet_delay = self.owner.scheduler._quiet_seconds_remaining(self.owner.key)
        if quiet_delay > 0:
            self.reset_deadline(timing.monotonic() + quiet_delay)
            result.set_result(None)
            self.waiting = None
            return
        if state.active is not None or state.approved or state.judging:
            return
        cooldown_at = state.last_agent_finished_at + settings.cooldown_seconds
        if timing.monotonic() < cooldown_at:
            return
        self.owner.check_sleep()
        waiting = waiting_messages(state.pending)
        record_ids = [item.record_id for item in waiting]
        state.pending = []
        state.waking = False
        state.resume_after_rerun = False
        for item in waiting:
            item.release()
        ticket = PendingMessage(
            event,
            0,
            False,
            asyncio.get_running_loop().create_future(),
            timing.monotonic(),
            finished=CompletionSignal(state.changed),
            backlog_sleep_revision=state.sleep_revision,
        )
        state.active = ticket
        state.active_ids = record_ids
        event.set_extra("mingle_woke_from_sleep", True)
        ticket.ready.set_result(record_ids)
        result.set_result(ticket)
        self.waiting = None
        state.changed.set()
