import asyncio
import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context

from .debug import DebugDumper
from .settings import Settings
from .utils.history_formatting import format_prompt_history


@dataclass(frozen=True)
class ReplyDecision:
    should_reply: bool = False
    action: Literal["wait", "rerun", "stop"] = "wait"
    reason: str = ""


class ReplyJudge:
    def __init__(
        self,
        context: Context,
        settings: Settings,
        logger: logging.Logger,
        debug: DebugDumper | None = None,
    ) -> None:
        self.context = context
        self.settings = settings
        self.logger = logger
        self.debug = debug

    async def should_reply(
        self,
        event: AstrMessageEvent,
        messages: list[dict],
        *,
        message_status: dict | None = None,
        new_message_ids: list[int] | None = None,
    ) -> bool:
        """Return an idle-session reply decision.

        Args:
            event: Event used to resolve the session's provider and persona.
            messages: Recent chronological chat records.
            message_status: Live scheduling state indexed by message record IDs.
            new_message_ids: Records triggering this evaluation.

        Returns:
            Whether the model explicitly permits a reply.
        """
        return (
            await self.evaluate(
                event,
                messages,
                message_status=message_status,
                new_message_ids=new_message_ids,
            )
        ).should_reply

    async def evaluate(
        self,
        event: AstrMessageEvent,
        messages: list[dict],
        *,
        new_message_ids: list[int] | None = None,
        message_status: dict | None = None,
    ) -> ReplyDecision:
        """Ask a configured provider whether joining the conversation is useful.

        Args:
            event: Event used to resolve the current session's default provider.
            messages: Recent chronological chat records, treated as untrusted data.
            new_message_ids: Records whose arrival triggered this evaluation.
            message_status: Finished, active, queued, and pending record IDs.

        Returns:
            A validated reply and interruption decision. Failures and malformed
            responses never request interruption.
        """
        request_id = None
        try:
            provider_id = self.settings.judge_provider_id
            if not provider_id:
                provider_id = await self.context.get_current_chat_provider_id(
                    event.unified_msg_origin
                )
            conversation_manager = self.context.conversation_manager
            conversation_id = await conversation_manager.get_curr_conversation_id(
                event.unified_msg_origin
            )
            conversation = (
                await conversation_manager.get_conversation(
                    event.unified_msg_origin, conversation_id
                )
                if conversation_id
                else None
            )
            (
                _,
                persona,
                _,
                _,
            ) = await self.context.persona_manager.resolve_selected_persona(
                umo=event.unified_msg_origin,
                conversation_persona_id=conversation.persona_id
                if conversation
                else None,
                platform_name=event.get_platform_name(),
            )
            system_prompt = (
                self.settings.judge_prompt + self.settings.judge_prompt_append
            )
            if persona and persona.get("prompt"):
                # Keep persona in the stable system prefix, before changing chat
                # data. Resolve it each time so edits and conversation switches apply.
                system_prompt += (
                    "\n\nConsider the following bot persona when deciding whether "
                    "participating would be natural. It describes the bot, not "
                    "a replacement for the judgment task or JSON output format.\n"
                    "```quote\n" + persona["prompt"] + "\n```"
                )
            timezone_name = self.context.get_config(umo=event.unified_msg_origin).get(
                "timezone"
            )
            judgment_messages = format_prompt_history(messages, timezone_name)
            status_data = message_status or {}
            new_ids = set(new_message_ids or status_data.get("judging", []))
            processing_ids = set(status_data.get("active", []))
            queued_ids = set(status_data.get("queued", []))
            include_status = message_status is not None or new_message_ids is not None
            for record in judgment_messages:
                if include_status:
                    record_id = record.get("id")
                    if record_id in new_ids:
                        record["status"] = "new"
                    elif record_id in processing_ids:
                        record["status"] = "processing"
                    elif record_id in queued_ids:
                        record["status"] = "queued"
                    else:
                        # History means context only, not proof of a sent reply.
                        record["status"] = "history"
            now = datetime.now(timezone.utc).timestamp()
            current_time = format_prompt_history([{"timestamp": now}], timezone_name)[
                0
            ]["timestamp"]
            previous_message = None
            for index, record in enumerate(messages):
                if record.get("id") in new_ids:
                    if index > 0:
                        previous_message = messages[index - 1]
                    break
            elapsed_text = "unknown (no valid message timestamp before new messages)"
            if previous_message is not None:
                try:
                    elapsed = now - previous_message["timestamp"]
                    if math.isfinite(elapsed):
                        # Future platform timestamps must not produce negative ages.
                        elapsed = max(0, int(elapsed))
                        for size, unit in (
                            (86400, "day"),
                            (3600, "hour"),
                            (60, "minute"),
                            (1, "second"),
                        ):
                            if elapsed >= size or size == 1:
                                count = elapsed // size
                                elapsed_text = (
                                    f"{count} {unit}{'s' if count != 1 else ''}"
                                )
                                break
                except (KeyError, TypeError, ValueError, OverflowError):
                    pass
            prompt_data = {
                "messages": judgment_messages,
                "time_context": (
                    f"Current time: {current_time}\n"
                    f"Time since the last message before the new messages: {elapsed_text}"
                ),
            }
            prompt = json.dumps(prompt_data, ensure_ascii=False)
            self.logger.info(
                "Mingle judgment input for session %s: messages=%d, new=%d",
                event.unified_msg_origin,
                len(judgment_messages),
                len(new_ids),
            )
            request_id = uuid.uuid4().hex
            if self.debug is not None:
                await self.debug.dump(
                    event,
                    "judge",
                    "input",
                    {
                        "provider_id": provider_id,
                        "system_prompt": system_prompt,
                        "prompt": prompt,
                    },
                    request_id,
                )
            response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                ),
                timeout=self.settings.judge_timeout,
            )
            if self.debug is not None:
                await self.debug.dump(
                    event,
                    "judge",
                    "output",
                    {
                        "completion_text": response.completion_text,
                        "response": response,
                    },
                    request_id,
                )
            text = response.completion_text.strip()
            if text.startswith("```") and text.endswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            if not isinstance(result, dict) or not isinstance(
                result.get("should_reply"), bool
            ):
                self.logger.info(
                    "Mingle judgment result for session %s: invalid",
                    event.unified_msg_origin,
                )
                return ReplyDecision()
            action = result.get("action", "wait")
            if action not in {"wait", "rerun", "stop"}:
                self.logger.info(
                    "Mingle judgment result for session %s: invalid",
                    event.unified_msg_origin,
                )
                return ReplyDecision()
            if (action == "rerun" and not result["should_reply"]) or (
                action == "stop" and result["should_reply"]
            ):
                self.logger.info(
                    "Mingle judgment result for session %s: invalid",
                    event.unified_msg_origin,
                )
                return ReplyDecision()
            self.logger.info(
                "Mingle judgment result for session %s: should_reply=%s, action=%s",
                event.unified_msg_origin,
                result["should_reply"],
                action,
            )
            return ReplyDecision(
                should_reply=result["should_reply"],
                action=action,
                reason=str(result.get("reason", "")),
            )
        except Exception as exc:
            if self.debug is not None and request_id is not None:
                await self.debug.dump(
                    event,
                    "judge",
                    "error",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    request_id,
                )
            self.logger.warning(
                "Reply judgment failed for session %s",
                event.unified_msg_origin,
                exc_info=True,
            )
            return ReplyDecision()
