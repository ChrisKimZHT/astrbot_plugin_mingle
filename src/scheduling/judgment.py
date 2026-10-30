"""Read-only background work for a captured session judgment attempt."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal

from ..decision import ReplyDecision, ReplyJudge
from ..history import ChatHistory
from .state import PendingMessage, SessionState


@dataclass
class JudgmentAttempt:
    """Capture ownership and input across history reads and model evaluation."""

    batch: list[PendingMessage]
    revision: int
    waking: bool
    sleep_backlog: bool
    forced: bool
    stage: Literal["history", "supplement", "decision"] = "history"
    active: PendingMessage | None = None
    active_ids: list[int] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)

    async def load_history(
        self, history: ChatHistory, key: str, limit: int
    ) -> list[dict]:
        """Read the initial recent window without changing live session state.

        Args:
            history: Independent Mingle storage.
            key: Full session UMO.
            limit: Configured judgment context limit.
        """
        return await history.get(
            key,
            limit=limit,
            before_id=max(item.record_id for item in self.batch) + 1,
        )

    def missing_record_ids(self) -> set[int]:
        """Return retained active and wake-batch records outside the recent window."""
        loaded_ids = {item["id"] for item in self.records}
        missing_ids = set(self.active_ids) - loaded_ids
        if self.waking:
            batch_ids = {item.record_id for item in self.batch}
            missing_ids.update(batch_ids - loaded_ids)
        return missing_ids

    async def supplement_history(
        self, history: ChatHistory, key: str, missing_ids: set[int]
    ) -> list[dict]:
        """Load retained context without altering queues or the captured attempt.

        Args:
            history: Independent Mingle storage.
            key: Full session UMO.
            missing_ids: Exact database record IDs absent from the recent window.
        """
        requests = []
        for record_id in missing_ids:
            requests.append(
                history.get(
                    key,
                    limit=1,
                    after_id=record_id - 1,
                    before_id=record_id + 1,
                )
            )
        rows_by_request = await asyncio.gather(*requests)
        records = list(self.records)
        for rows in rows_by_request:
            for item in rows:
                if item["id"] in missing_ids:
                    records.append(item)
        return sorted(records, key=lambda item: item["id"])

    def message_status(self, state: SessionState) -> dict:
        """Snapshot status immediately before launching model work.

        Args:
            state: Current queues and bounded finish outcomes, read synchronously.
        """
        return {
            "finished": [
                {"id": record_id, "outcome": outcome}
                for record_id, outcome in state.finished.items()
            ],
            "active": [] if self.waking else list(self.active_ids),
            "queued": [
                item.record_id for item in state.approved if not item.ready.done()
            ],
            "awaiting_judgment": [
                item.record_id for item in state.pending if not item.ready.done()
            ],
            "judging": [item.record_id for item in self.batch],
        }

    async def evaluate(
        self, judge: ReplyJudge, logger: logging.Logger, key: str, status: dict
    ) -> ReplyDecision:
        """Return a decision without applying control actions or queue changes.

        Args:
            judge: Current configured judgment service.
            logger: Plugin logger.
            key: Full UMO for logging.
            status: Scheduling snapshot captured by the coordinator.
        """
        leader = self.batch[-1]
        new_ids = [item.record_id for item in self.batch]
        if self.waking:
            new_ids = [leader.record_id]
        kwargs = {
            "message_status": status,
            "new_message_ids": new_ids,
            "waking": self.waking,
            "sleep_backlog": self.sleep_backlog,
        }
        if self.active is not None:
            return await judge.evaluate(leader.event, self.records, **kwargs)
        if not self.waking and self.forced:
            logger.info(
                "Mingle judgment bypassed for forced idle batch in session %s", key
            )
            return ReplyDecision(True)
        should_reply = await judge.should_reply(leader.event, self.records, **kwargs)
        return ReplyDecision(should_reply)
