import asyncio
import logging
from dataclasses import dataclass, field
from time import monotonic

from astrbot.api.event import AstrMessageEvent

from .decision import ReplyDecision, ReplyJudge
from .history import ChatHistory
from .settings import Settings


@dataclass
class PendingMessage:
    """Keep the original handler alive through admission and pipeline completion.

    Only the newest ticket in an admitted batch receives record IDs via ready;
    the others receive None. The admitted handler sets finished after its entire
    pipeline unwinds, including framework history saving.
    """

    event: AstrMessageEvent
    record_id: int
    forced: bool
    ready: asyncio.Future[list[int] | None]
    received_at: float
    finished: asyncio.Event = field(default_factory=asyncio.Event)

    def release(self) -> None:
        """Release a waiting handler without admitting another Agent turn."""
        if not self.ready.done():
            self.ready.set_result(None)


def _waiting_messages(messages: list[PendingMessage]) -> list[PendingMessage]:
    """Release stopped events and retain tickets still awaiting admission.

    Args:
        messages: Pending or approved tickets to inspect without changing the list.

    Returns:
        Unresolved tickets whose events have not been stopped.
    """
    waiting = []
    for item in messages:
        if item.event.is_stopped() or item.event.get_extra(
            "agent_stop_requested", False
        ):
            item.release()
        if not item.ready.done():
            waiting.append(item)
    return waiting


@dataclass
class SessionState:
    """Shared state for the session's judgment and execution workers.

    Tickets move through pending -> judging -> approved -> active. Judgment owns
    pending/judging and appends to approved; execution drains approved and owns
    active/active_ids. A stop decision may discard approved tickets. changed
    wakes judgment when arrivals or Agent transitions affect its deadline.

    State transitions contain no awaits. After IO, workers must recheck any
    live state needed to act on a previously captured Agent or batch.
    """

    pending: list[PendingMessage] = field(default_factory=list)
    judging: list[PendingMessage] = field(default_factory=list)
    approved: list[PendingMessage] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    judgment_task: asyncio.Task | None = None
    execution_task: asyncio.Task | None = None
    active: PendingMessage | None = None
    last_agent_finished_at: float = float("-inf")
    resume_after_rerun: bool = False
    active_ids: list[int] = field(default_factory=list)
    finished: dict[int, str] = field(default_factory=dict)


class ChatScheduler:
    """Judge incoming batches concurrently with one serialized Agent per session."""

    def __init__(
        self,
        settings: Settings,
        history: ChatHistory,
        judge: ReplyJudge,
        logger: logging.Logger,
    ) -> None:
        self.settings = settings
        self.history = history
        self.judge = judge
        self.logger = logger
        self.sessions: dict[str, SessionState] = {}
        self.closed = False

    async def submit(
        self, event: AstrMessageEvent, record_id: int, forced: bool
    ) -> PendingMessage:
        """Queue an event without starting a second Agent for the same session.

        Args:
            event: Original event kept alive by its plugin handler.
            record_id: Persisted chat record identifier.
            forced: Whether a private message or explicit mention bypasses judgment.

        Returns:
            Admission ticket whose ready future contains a batch or None.
        """
        ticket = PendingMessage(
            event,
            record_id,
            forced,
            asyncio.get_running_loop().create_future(),
            monotonic(),
        )
        if self.closed:
            ticket.release()
            return ticket
        key = event.unified_msg_origin
        state = self.sessions.setdefault(key, SessionState())
        state.pending.append(ticket)
        if len(state.pending) > self.settings.max_pending_messages:
            oldest = state.pending.pop(0)
            self._finish_records(state, [oldest.record_id], "merged")
            # Keep direct-address intent even when its original event is merged away.
            state.pending[0].forced |= oldest.forced
            oldest.release()
        self.logger.info(
            "Mingle queue for session %s: pending=%d, judging=%d, approved=%d, active=%d",
            key,
            len(state.pending),
            len(state.judging),
            len(state.approved),
            len(state.active_ids),
        )
        state.changed.set()
        if state.judgment_task is None or state.judgment_task.done():
            state.judgment_task = asyncio.create_task(self._judge_pending(key, state))
        return ticket

    async def _judge_pending(self, key: str, state: SessionState) -> None:
        """Merge and judge arrivals while execution independently runs the Agent.

        Args:
            key: Unified session identifier.
            state: Pending and active work for this session.
        """
        batch: list[PendingMessage] = []
        try:
            while state.pending and not self.closed:
                await self._wait_for_pending(state)
                batch, state.pending = _waiting_messages(state.pending), []
                if not batch:
                    continue
                state.judging = batch
                leader = batch[-1]
                forced = any(item.forced for item in batch)
                records = await self.history.get(
                    key,
                    limit=self.settings.initial_context_messages,
                    before_id=max(item.record_id for item in batch) + 1,
                )
                if leader.ready.done():
                    state.judging = []
                    batch = []
                    continue

                # Snapshot only after IO: the previous Agent may have finished
                # during the history read, starting a fresh cooldown.
                active = state.active
                if active is not None and active.finished.is_set():
                    active = None
                cooldown_until = (
                    state.last_agent_finished_at + self.settings.cooldown_seconds
                )
                if active is None and not forced and monotonic() < cooldown_until:
                    state.pending = batch + state.pending
                    state.judging = []
                    batch = []
                    continue

                decision = await self._evaluate_batch(
                    key, state, batch, records, active
                )
                # Recovery may have claimed this batch while judgment awaited IO.
                if leader.ready.done():
                    state.judging = []
                    batch = []
                    continue
                self._apply_control(key, state, leader, active, decision)
                allowed = (
                    forced or decision.should_reply
                ) and decision.action != "stop"
                if not allowed or self.closed or leader.ready.done():
                    self._finish_records(
                        state, [item.record_id for item in batch], "no_reply"
                    )
                    for item in batch:
                        item.release()
                else:
                    if (
                        decision.should_reply
                        and decision.action == "wait"
                        and (active is not None or not forced)
                    ):
                        # Freeze the tail once judgment returns. Later arrivals
                        # still need judgment, including stop/rerun requests.
                        batch.extend(_waiting_messages(state.pending))
                        state.pending = []
                    state.approved.extend(batch)
                    while len(state.approved) > self.settings.max_pending_messages:
                        oldest = state.approved.pop(0)
                        self._finish_records(state, [oldest.record_id], "merged")
                        oldest.release()
                    if state.execution_task is None or state.execution_task.done():
                        state.execution_task = asyncio.create_task(
                            self._execute_approved(key, state)
                        )
                state.judging = []
                self.logger.info(
                    "Mingle batch for session %s: messages=%d, forced=%s, admitted=%s; pending=%d, approved=%d, active=%d",
                    key,
                    len(batch),
                    forced,
                    allowed and not self.closed and not leader.ready.done(),
                    len(state.pending),
                    len(state.approved),
                    len(state.active_ids),
                )
                batch = []
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Chat scheduling failed for session %s", key)
        finally:
            for item in batch + state.pending:
                item.release()
            state.pending.clear()
            state.judging = []

    async def _wait_for_pending(self, state: SessionState) -> None:
        """Wait until the current arrivals satisfy merge timing and cooldown.

        Recompute from original arrival times on every wakeup. An active Agent
        or direct-address ticket bypasses cooldown, but never merge timing.

        Args:
            state: Session whose pending queue remains open to new arrivals.
        """
        while state.pending and not self.closed:
            state.pending = _waiting_messages(state.pending)
            if not state.pending:
                return
            merge_at = min(
                state.pending[-1].received_at + self.settings.debounce_seconds,
                state.pending[0].received_at + self.settings.max_wait_seconds,
            )
            active = state.active
            busy = active is not None and not active.finished.is_set()
            forced = any(item.forced for item in state.pending)
            ready_at = merge_at
            if not busy and not forced:
                ready_at = max(
                    merge_at,
                    state.last_agent_finished_at + self.settings.cooldown_seconds,
                )
            delay = ready_at - monotonic()
            if delay <= 0:
                return
            # No await between inspecting state and clearing the notification:
            # arrivals and Agent transitions cannot be lost in that interval.
            state.changed.clear()
            try:
                await asyncio.wait_for(state.changed.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def _evaluate_batch(
        self,
        key: str,
        state: SessionState,
        batch: list[PendingMessage],
        records: list[dict],
        active: PendingMessage | None,
    ) -> ReplyDecision:
        """Build judgment context and evaluate against a captured active turn.

        This method does not change queue ownership or stop an Agent. The caller
        must validate the captured active ticket before applying control actions.

        Args:
            key: Unified session identifier for retained history reads.
            state: Live queues and bounded completion outcomes.
            batch: Arrivals being judged, with the newest event as leader.
            records: Recent history already loaded for this batch.
            active: Active ticket captured after the initial history read.

        Returns:
            Reply admission and any requested control action. Idle direct-address
            batches are admitted without calling the judge.
        """
        active_ids = list(state.active_ids) if active is not None else []
        missing_ids = set(active_ids) - {item["id"] for item in records}
        if missing_ids:
            # Preserve active input even when arrivals push it outside the
            # recent-history window. Keep IDs tied to the captured Agent.
            active_records = await asyncio.gather(
                *(
                    self.history.get(
                        key,
                        limit=1,
                        after_id=record_id - 1,
                        before_id=record_id + 1,
                    )
                    for record_id in missing_ids
                )
            )
            records = sorted(
                records
                + [
                    item
                    for rows in active_records
                    for item in rows
                    if item["id"] in missing_ids
                ],
                key=lambda item: item["id"],
            )
        message_status = {
            "finished": [
                {"id": record_id, "outcome": outcome}
                for record_id, outcome in state.finished.items()
            ],
            "active": active_ids,
            "queued": [
                item.record_id for item in state.approved if not item.ready.done()
            ],
            "awaiting_judgment": [
                item.record_id for item in state.pending if not item.ready.done()
            ],
            "judging": [item.record_id for item in batch],
        }
        leader = batch[-1]
        if active is not None:
            return await self.judge.evaluate(
                leader.event,
                records,
                message_status=message_status,
                new_message_ids=[item.record_id for item in batch],
            )
        if any(item.forced for item in batch):
            self.logger.info(
                "Mingle judgment bypassed for forced idle batch in session %s", key
            )
            return ReplyDecision(True)
        return ReplyDecision(
            await self.judge.should_reply(
                leader.event,
                records,
                message_status=message_status,
                new_message_ids=[item.record_id for item in batch],
            )
        )

    def _apply_control(
        self,
        key: str,
        state: SessionState,
        leader: PendingMessage,
        active: PendingMessage | None,
        decision: ReplyDecision,
    ) -> None:
        """Apply stop/rerun only while the captured Agent and request remain valid.

        Args:
            key: Unified session identifier for logging.
            state: Live active turn and approved queue.
            leader: Ticket whose event requested judgment.
            active: Agent ticket captured before awaiting judgment.
            decision: Validated control action returned by the judge.
        """
        if decision.action not in {"rerun", "stop"} or active is None:
            return
        # A slow judgment must never stop a different, newer turn. A released
        # leader (for example, after group off) no longer authorizes control.
        if state.active is not active or active.finished.is_set():
            return
        if (
            leader.ready.done()
            or leader.event.is_stopped()
            or leader.event.get_extra("agent_stop_requested", False)
        ):
            return
        active.event.set_extra("agent_stop_requested", True)
        state.resume_after_rerun = decision.action == "rerun"
        self.logger.info(
            "Mingle requested Agent %s for session %s", decision.action, key
        )
        if decision.action == "stop":
            self._finish_records(
                state,
                [item.record_id for item in state.approved],
                "discarded_by_stop",
            )
            for item in state.approved:
                item.release()
            state.approved.clear()

    def claim_recovery_messages(self, ticket: PendingMessage) -> list[int] | None:
        """Fold waiting arrivals into the active handler's single recovery run.

        Release claimed handlers atomically so in-flight judgments cannot queue
        their messages again or stop the recovery with a stale control result.

        Args:
            ticket: Active handler after its previous Agent pipeline has unwound.

        Returns:
            Claimed record IDs, or None when the active handler is no longer valid.
        """
        if self.closed:
            return None
        event = ticket.event
        state = self.sessions.get(event.unified_msg_origin)
        if state is None or state.active is not ticket:
            return None
        if ticket.finished.is_set() or event.is_stopped():
            return None
        if event.get_extra("agent_stop_requested", False):
            return None
        if event.get_extra("agent_user_aborted", False):
            return None
        waiting = _waiting_messages(state.pending + state.judging + state.approved)
        record_ids = sorted(item.record_id for item in waiting)
        state.pending = []
        state.judging = []
        state.approved = []
        state.active_ids.extend(record_ids)
        for item in waiting:
            item.release()
        state.changed.set()
        return record_ids

    async def _execute_approved(self, key: str, state: SessionState) -> None:
        """Admit approved batches only after the previous pipeline has unwound.

        Args:
            key: Unified session identifier.
            state: Approved work and active execution for this session.
        """
        batch: list[PendingMessage] = []
        try:
            while state.approved and not self.closed:
                batch, state.approved = _waiting_messages(state.approved), []
                if not batch:
                    state.resume_after_rerun = False
                    continue
                leader = batch[-1]
                for item in batch[:-1]:
                    item.release()
                if state.resume_after_rerun:
                    leader.event.set_extra("mingle_resume_after_rerun", True)
                    state.resume_after_rerun = False
                state.active = leader
                state.active_ids = [item.record_id for item in batch]
                state.changed.set()
                leader.ready.set_result([item.record_id for item in batch])
                # The handler sets finished after the framework's history save,
                # not merely after its final response hook or first send.
                await leader.finished.wait()
                outcome = (
                    "interrupted"
                    if (
                        leader.event.get_extra("agent_user_aborted", False)
                        or leader.event.get_extra("agent_stop_requested", False)
                    )
                    else "completed"
                )
                self._finish_records(state, state.active_ids, outcome)
                if leader.event.get_extra("mingle_agent_started", False):
                    state.last_agent_finished_at = monotonic()
                state.active = None
                state.active_ids = []
                self.logger.info(
                    "Mingle Agent finished for session %s: outcome=%s, pending=%d, approved=%d",
                    key,
                    outcome,
                    len(state.pending),
                    len(state.approved),
                )
                state.changed.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Mingle Agent admission failed for session %s", key)
        finally:
            for item in batch + state.approved:
                item.release()
            state.approved.clear()
            state.active = None
            state.active_ids = []

    def _finish_records(
        self, state: SessionState, record_ids: list[int], outcome: str
    ) -> None:
        """Retain bounded live outcomes for later judgment context.

        Args:
            state: Session whose records reached a terminal scheduling state.
            record_ids: Mingle record identifiers, not platform message IDs.
            outcome: Completion, interruption, or no-reply outcome.
        """
        for record_id in record_ids:
            state.finished[record_id] = outcome
        while (
            len(state.finished)
            > self.settings.max_pending_messages
            + self.settings.initial_context_messages
        ):
            state.finished.pop(next(iter(state.finished)))

    async def close(self) -> None:
        """Stop admission, release waiting handlers, and stop owned active events."""
        self.closed = True
        tasks = []
        for state in self.sessions.values():
            if state.active is not None:
                state.active.event.set_extra("agent_stop_requested", True)
                state.active.event.stop_event()
            for item in state.pending + state.approved:
                item.release()
            if state.judgment_task is not None:
                state.judgment_task.cancel()
                tasks.append(state.judgment_task)
            if state.execution_task is not None:
                state.execution_task.cancel()
                tasks.append(state.execution_task)
        await asyncio.gather(*tasks, return_exceptions=True)
        self.sessions.clear()
