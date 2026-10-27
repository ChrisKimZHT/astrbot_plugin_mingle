import asyncio
import json
import logging
import uuid

from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context

from .debug import DebugDumper
from .settings import Settings


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

    async def should_reply(self, event: AstrMessageEvent, messages: list[dict]) -> bool:
        """Ask a configured provider whether joining the conversation is useful.

        Args:
            event: Event used to resolve the current session's default provider.
            messages: Recent chronological chat records, treated as untrusted data.

        Returns:
            Whether a valid model decision explicitly permits a reply. Failures
            and malformed responses leave the bot silent.
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
            prompt = json.dumps(messages, ensure_ascii=False)
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
            return isinstance(result, dict) and result.get("should_reply") is True
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
            return False
