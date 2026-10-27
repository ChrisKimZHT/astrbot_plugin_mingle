# Mingle implementation agreements

## Scope and decisions

- This repository is an AstrBot plugin, not a fork of the framework.
- Do not monkey patch AstrBot or change its global configuration.
- Ask the user before changing important architecture or behavioral decisions.
- Keep logs and code comments in English. Use Google-style docstrings for
  non-obvious methods. Keep modules cohesive; do not introduce small one-use
  helpers merely to shorten methods.
- Do not write or expand README.md during the initial implementation phase.
  The existing template README is intentionally left untouched.

## Message and Agent flow

- `main.py` owns AstrBot registration, command bypass, request construction,
  response hooks, and history/image tools.
- Keep business modules in the `src/` package. The root `main.py` remains the
  AstrBot entry point and uses relative imports from `.src`. Keep tests in the
  root `tests/` directory and import business modules through the `src` package.
- `src/scheduler.py` owns one pending queue and one active turn per unified session.
  It merges consecutive messages, bounds the merge wait and pending event count,
  applies proactive cooldown, and waits for the admitted handler to finish.
- `src/decision.py` calls a judgment provider without tools. An empty provider ID
  follows AstrBot's current session provider; another configured provider may be
  selected in the plugin settings. Invalid decisions and failures mean silence.
- Compact only the judgment-model history input: format Unix timestamps as
  readable date/time with the configured AstrBot timezone and numeric offset
  (UTC fallback), shorten sender IDs to their last eight characters, and omit
  `platform_message_id`. Work on copies; stored records, Agent history tools,
  and identity/permission checks retain full IDs and original timestamps.
  Invalid timestamps become null. Debug dumps preserve the compact prompt
  string actually passed to the judgment provider.
- Resolve the judgment persona with AstrBot's `resolve_selected_persona`, using
  the current conversation persona and session override rules. Put its prompt
  inside a `quote` fenced block in the stable system prompt, after fixed judgment
  instructions. Changing chat records belong only in the user message. Never
  add message IDs, timestamps, or session IDs to the system prefix. Resolve the
  persona on every judgment so edits, resets, and switches take effect; do not
  maintain a stale persona cache. Reusing a judgment provider does not itself
  apply the Agent persona, so this injection is explicit.
- Judgment instructions are configurable through `judge_prompt` (the original
  default rules) and `judge_prompt_append` (empty by default). Concatenate them
  verbatim, without trimming, interpolation, or an implicit separator. Append
  the persona block afterwards; chat history stays in the separate user input.
  Keep the schema default synchronized with `DEFAULT_JUDGE_PROMPT` in settings.
- `src/history.py` owns an independent SQLite database and image cache under
  `StarTools.get_data_dir("astrbot_plugin_mingle")`.
- `src/settings.py` validates immutable runtime settings. `_conf_schema.json` exposes
  these settings in AstrBot's configuration UI.
- Keep debugging implementation in `src/debug.py`, including future debug
  features. `debug_sessions` is an independent list of full UMOs, empty by
  default. Platform-scoped whole-group UMOs also cover isolated member events.
  `/mingle debug on|off` is administrator-only and persists the current group
  or private chat setting without enabling normal Mingle chat handling.
- Debug capture is explicitly at plugin level: decision-model input/output,
  plus Agent initial request and final response through standard hooks. Do not
  wrap Providers or claim to capture every Agent step or final HTTP payload.
  Request hooks run at low priority to observe ordinary plugin modifications,
  but later hooks can still change the request. Matching native Agent requests
  are included even outside Mingle-managed groups when these hooks execute.
- Dumps are JSON with same-stem HTML companions under the plugin data directory
  `debug/<UTC-date>/<scope-hash>/`. Each input/output pair has a request ID. Use
  unique filenames, atomic writes, and off-thread file IO. Log only the absolute
  HTML path at INFO after both files are written successfully.
  Render every serialized JSON field as HTML tables, with minimal inline CSS
  for collapsed borders, 100% width, automatic table layout and long-text wrapping.
  Keep row-label columns compact with an eight-character minimum so nested
  tables do not repeatedly split available width into equal halves.
  No JavaScript, external dependencies, or omitted values. Escape all content and
  link the original JSON. For JSON-encoded prompt strings, display a parsed
  table plus the original string in native details/pre elements; never change
  the saved JSON or the actual model request to improve presentation.
  Write failures must not break inference. Do not dump provider configuration
  or authentication headers; prompts, history and returned model data are saved.
  Switching debug off stops new dumps and does not delete existing files.
  Existing debug files are not subject to chat/image retention cleanup.
- Decision input dumps store `payload.prompt` as the exact JSON-encoded string
  passed to `context.llm_generate`. Do not replace it with a parsed array in the
  dump; the snapshot should preserve the plugin-layer call arguments.
- Group sessions must be explicitly listed. Prefer full UMO identifiers; bare
  group IDs are supported. Empty group configuration enables no group chats.
- `/mingle group on|off` is restricted to AstrBot administrators, on both the
  group registration and its concrete subcommand. Reject private-chat use.
  Enable stores a platform-scoped group UMO; match that group UMO even when
  AstrBot isolates member conversations. Disable removes the group UMO, current
  event UMO, and legacy bare group ID, preserving unrelated allowlist entries.
  Save the plugin config asynchronously under a lock before updating runtime
  settings. On failure restore the in-memory config. Waiting events recheck
  the allowlist before requesting an Agent; active turns are not interrupted.
- Private messages are enabled by default and use the same scheduler. WebChat
  remains on its native flow. Private messages and explicit wake/mention events
  bypass the judgment model and proactive cooldown.
- Recognized command handlers, including management commands, retain their
  native flow. Other ordinary message handlers do not execute after Mingle has
  taken ownership of a message. LLM hooks and normal result decoration remain
  part of the default pipeline.
- Only the current AstrBot local runner is supported in v0.1.0. Unsupported
  runners and disabled AI fall through to the framework.
- A listener submits its original event, waits for batch admission, and yields
  `event.request_llm(prompt=..., conversation=...)`. Never pass an additional
  contexts list: the framework reloads the bound conversation under its lock.
- The handler resumes only after the explicit Agent request finishes, including
  framework history saving. Its finally block releases the batch and stops the
  default fallback. `should_call_llm(True)` disables fallback in this AstrBot
  version; it does not disable the yielded ProviderRequest.
- New messages received during a turn are persisted immediately and queued.
  They can be read through the history tool while the turn is running. They do
  not automatically steer or cancel the running Agent. Cross-sender follow-up
  and interruption/restart policies are deferred.
- AstrBot session isolation is respected. To share a group's conversation across
  members, disable its `unique_session` option. Do not override isolation here.
- Register the two Mingle tools normally. If a persona restricts its tools,
  explicitly allow `mingle_chat_history` and `mingle_view_image` in that persona.
  Do not bypass persona tool restrictions by silently injecting tools.
- Tool docstring Args must list only model-supplied arguments with explicit
  Google-style types. AstrBot injects `event`; listing it in Args makes this
  version's tool schema parser treat it as a model parameter.

## Storage and attachments

- Do not write to AstrBot's platform message table or alter its retention rules.
- Records are partitioned by full UMO. Stable platform message IDs plus sender
  identity deduplicate inbound messages within retained history, including after
  restarts. Platforms without IDs receive independent record identities.
- Query cursors are Mingle database IDs, not platform IDs. Tools receive the
  session from their event; never accept a model-supplied session or file path.
- Preserve sender identity, timestamps, message text, platform reply IDs and
  cached image indices. Tool output is chronological and bounded.
- The default limits are 30 days / 10,000 records per session, 7 days for images,
  512 MiB total image cache, and 20 MiB per image. All are configurable.
- Copy incoming images into content-hash-named files before the original event
  can clean up its media. Do not depend on remote URLs or temporary paths later.
  Repeated content shares one file. Expiration is based on last receipt.
- Validate image content, limit bytes read, use atomic file replacement, and
  return MCP image content to the Agent through `mingle_view_image`.
- Retention runs at startup, every five minutes, and after image-bearing writes.
  Unreferenced files left by interrupted writes are cleaned up as well.
- No automatic image summaries in v0.1.0. Audio, video and other files have type
  markers only; text already produced by AstrBot STT is preserved as text.
- Bot rows represent completed Agent response-hook output, not delivery receipts.
  They do not claim the platform successfully delivered a reply. Other plugins'
  independent direct sends are not automatically captured.
- Unloading stops admission, releases waiting tickets, cancels owned pipeline
  handlers and periodic cleanup, then closes SQLite. This shutdown cancellation
  is distinct from a normal `/stop` that lets AstrBot save interrupted history.

## Validation

From the AstrBot root, using its environment:

```powershell
python -m pytest data/plugins/astrbot_plugin_mingle/tests -q --confcutdir=data/plugins/astrbot_plugin_mingle/tests
ruff format data/plugins/astrbot_plugin_mingle --no-respect-gitignore
ruff check data/plugins/astrbot_plugin_mingle --no-respect-gitignore
```

`data/` is ignored by the parent repository, so lint the plugin explicitly with
`--no-respect-gitignore`. Also run the parent project's required Ruff checks.
Tests use real local SQLite/files and framework pipeline dispatch, with model
and platform calls replaced by fakes. Live provider/platform compatibility must
be verified separately with configured credentials; tests do not contact them.
