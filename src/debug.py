import asyncio
import base64
import hashlib
import json
import logging
import uuid
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from html import escape
from pathlib import Path
from typing import Literal

from astrbot.api.event import AstrMessageEvent
from astrbot.core.platform.message_type import MessageType
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from pydantic import BaseModel

from .settings import Settings


def _serialize_debug_value(value: object) -> object:
    """Serialize framework values without copying executable tool handlers.

    Args:
        value: Value that the JSON encoder cannot serialize directly.

    Returns:
        JSON-compatible fields, normalized values or a string representation.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if is_dataclass(value) and not isinstance(value, type):
        # Handlers can retain the entire plugin context; never deepcopy them.
        serialized_fields = {}
        for field in fields(value):
            if field.name in {"handler", "handler_module_path"}:
                continue
            serialized_fields[field.name] = getattr(value, field.name)
        return serialized_fields
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    return str(value)


def _render_debug_value(value: object) -> str:
    """Render JSON values recursively as escaped HTML tables.

    Args:
        value: A JSON-compatible value from the serialized snapshot.

    Returns:
        HTML preserving every value, including raw text for JSON-encoded strings.
    """
    if isinstance(value, str):
        raw = f"<pre>{escape(value)}</pre>"
        if value.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return raw
            if isinstance(parsed, (dict, list)):
                return (
                    _render_debug_value(parsed)
                    + "<details><summary>Original string</summary>"
                    + raw
                    + "</details>"
                )
        return raw
    if isinstance(value, dict) and value:
        rows = []
        for key, item in value.items():
            label = escape(str(key))
            rendered_item = _render_debug_value(item)
            rows.append(
                f'<tr><th scope="row">{label}</th><td>{rendered_item}</td></tr>'
            )
        body = "".join(rows)
        return (
            "<table><thead><tr><th>Field</th><th>Value</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )
    if not isinstance(value, list) or not value:
        return f"<pre>{escape(json.dumps(value, ensure_ascii=False))}</pre>"

    column_keys = {}
    if all(isinstance(item, dict) for item in value):
        for item in value:
            for key in item:
                column_keys[key] = None
    columns = list(column_keys)
    if columns:
        headings = "".join(f"<th>{escape(str(key))}</th>" for key in columns)
        rows = []
        for item in value:
            cells = []
            for key in columns:
                if key in item:
                    rendered_item = _render_debug_value(item[key])
                    cells.append(f"<td>{rendered_item}</td>")
                else:
                    cells.append("<td>(missing)</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        body = "".join(rows)
        return f"<table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table>"

    rows = []
    for index, item in enumerate(value):
        rendered_item = _render_debug_value(item)
        rows.append(f'<tr><th scope="row">{index}</th><td>{rendered_item}</td></tr>')
    body = "".join(rows)
    return (
        "<table><thead><tr><th>Index</th><th>Value</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def render_debug_html(document: dict, json_filename: str) -> str:
    """Build a standalone HTML view with bordered tables constrained to page width.

    Args:
        document: All fields of the serialized JSON dump.
        json_filename: Name of the original JSON file beside this HTML file.

    Returns:
        Complete HTML with tables and a relative link to the original JSON.
    """
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        "<title>Mingle debug dump</title>"
        "<style>"
        "table{width:100%;max-width:100%;table-layout:auto;border-collapse:collapse}"
        "th,td{border:1px solid;overflow-wrap:anywhere;vertical-align:top}"
        'th[scope="row"]{width:1%;min-width:8ch;text-align:left}'
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:100%}"
        "a{overflow-wrap:anywhere}"
        "</style></head><body>"
        "<h1>Mingle debug dump</h1>"
        f'<p><a href="{escape(json_filename, quote=True)}">Original JSON</a></p>'
        + _render_debug_value(document)
        + "</body></html>\n"
    )


class DebugDumper:
    """Write session-scoped plugin-layer LLM snapshots without affecting inference."""

    def __init__(
        self, directory: Path, settings: Settings, logger: logging.Logger
    ) -> None:
        self.directory = directory.resolve()
        self.settings = settings
        self.logger = logger

    def is_enabled(self, event: AstrMessageEvent) -> bool:
        """Check exact UMO and platform-scoped whole-group debug rules.

        Args:
            event: Message defining the authorized debug session.

        Returns:
            Whether a configured debug rule covers this event.
        """
        if event.unified_msg_origin in self.settings.debug_sessions:
            return True
        if event.is_private_chat() or not event.get_group_id():
            return False
        group_session = (
            f"{event.get_platform_id()}:{MessageType.GROUP_MESSAGE.value}:"
            f"{event.get_group_id()}"
        )
        return group_session in self.settings.debug_sessions

    async def capture_agent_request(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest,
        *,
        available: bool,
    ) -> None:
        """Capture the initial Agent request and retain its correlation ID.

        Args:
            event: Event defining the debug scope, including unmanaged requests.
            request: Plugin-layer input observed by the standard request hook.
            available: Whether the plugin has initialized and remains loaded.
        """
        if not available or not self.is_enabled(event):
            return
        request_id = uuid.uuid4().hex
        event.set_extra("mingle_debug_agent_request_id", request_id)
        await self.dump(
            event,
            "agent",
            "input",
            {
                "request": request,
                "selected_provider": event.get_extra("selected_provider"),
                "selected_model": event.get_extra("selected_model"),
            },
            request_id,
        )

    async def capture_agent_response(
        self,
        event: AstrMessageEvent,
        response: LLMResponse,
        *,
        available: bool,
    ) -> None:
        """Capture the raw final response for a previously observed request.

        Args:
            event: Event carrying the captured request's correlation ID.
            response: Final framework response, including error roles.
            available: Whether the plugin has initialized and remains loaded.
        """
        request_id = event.get_extra("mingle_debug_agent_request_id")
        if not available or not request_id:
            return
        await self.dump(
            event,
            "agent",
            "output",
            {
                "completion_text": response.completion_text,
                "response": response,
            },
            request_id,
        )

    async def dump(
        self,
        event: AstrMessageEvent,
        kind: Literal["judge", "agent"],
        phase: Literal["input", "output", "error"],
        payload: object,
        request_id: str,
    ) -> Path | None:
        """Write JSON and table-based HTML snapshots, logging only the HTML path.

        Args:
            event: Event supplying the current session and debug eligibility.
            kind: Decision model or conversation Agent.
            phase: Request, response, or failure snapshot.
            payload: Plugin-layer data; never a provider configuration or headers.
            request_id: Correlation identifier shared by a request and its response.

        Returns:
            Written JSON path, or None when disabled or either write fails.
        """
        if not self.is_enabled(event):
            return None
        now = datetime.now(timezone.utc)
        scope = hashlib.sha256(event.unified_msg_origin.encode("utf-8")).hexdigest()[
            :16
        ]
        path = (
            self.directory
            / now.strftime("%Y-%m-%d")
            / scope
            / (f"{now.strftime('%H%M%S_%f')}_{kind}_{phase}_{uuid.uuid4().hex}.json")
        )
        html_path = path.with_suffix(".html")
        document = {
            "capture_layer": "plugin",
            "request_id": request_id,
            "timestamp": now.isoformat(),
            "session": event.unified_msg_origin,
            "kind": kind,
            "phase": phase,
            "payload": payload,
        }

        try:
            await asyncio.to_thread(self._write_dump, path, document)
        except Exception:
            self.logger.warning(
                "Failed to write Mingle debug dump: %s", html_path, exc_info=True
            )
            return None
        return path

    def _write_dump(self, path: Path, document: dict) -> None:
        """Write JSON and HTML using atomic replacements in the file IO worker.

        Args:
            path: Absolute destination for the JSON snapshot.
            document: Plugin-layer snapshot including all payload fields.

        Raises:
            OSError: If writing or replacing files fails.
            TypeError: If the snapshot cannot be serialized.
        """
        html_path = path.with_suffix(".html")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        html_temporary = html_path.with_suffix(".html.tmp")
        try:
            text = json.dumps(
                document, ensure_ascii=False, indent=2, default=_serialize_debug_value
            )
            temporary.write_text(text + "\n", encoding="utf-8")
            temporary.replace(path)
            # Render serialized data so HTML contains exactly the JSON fields.
            html = render_debug_html(json.loads(text), path.name)
            html_temporary.write_text(html, encoding="utf-8")
            html_temporary.replace(html_path)
            # Announce success even if the waiting pipeline has been cancelled.
            self.logger.info("Mingle debug dump written: %s", html_path)
        finally:
            temporary.unlink(missing_ok=True)
            html_temporary.unlink(missing_ok=True)
