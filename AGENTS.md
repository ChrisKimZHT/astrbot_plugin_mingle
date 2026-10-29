# Mingle implementation agreements

## Scope and decisions

- This repository is an AstrBot plugin, not a fork of the framework.
- Do not monkey patch AstrBot or change its global configuration.
- Ask the user before changing important architecture or behavioral decisions.
- Keep logs and code comments in English. Use Google-style docstrings for
  non-obvious methods. Keep modules cohesive; do not introduce small one-use
  helpers merely to shorten methods.
- Keep production helper functions at module or class scope rather than nesting
  them inside methods. Pass file IO worker inputs explicitly instead of capturing
  mutable runtime state in closures.
- Keep README.md concise and user-facing: describe capabilities, setup and
  commands. Maintain implementation agreements here instead of in the README.

## Readability and behavior-preserving edits

- Prioritize readability and ease of debugging over reducing line count. Split
  complex conditions, calculations and function calls across multiple lines;
  introduce meaningful intermediate variables when useful.
- Never use nested ternary expressions. Use ternaries only for simple choices;
  use ordinary `if/else` statements for more complex branches.
- Use list, dictionary and set comprehensions only for simple transformations or
  filters. Use ordinary loops for multiple loop levels, complex conditions or
  long expressions. Apply the same readability rule to generator expressions.
- Split long conditions across multiple lines or introduce clearly named boolean
  variables. Preserve short-circuit evaluation when restructuring conditions.
- Avoid deep nesting. Use early returns or `continue` where they improve
  readability without changing behavior.
- In style-only edits, preserve existing functionality, public interfaces,
  return values and exception behavior. Do not bundle business logic rewrites
  into readability changes.
- Follow existing project configuration and agreements; they take precedence
  if these style guidelines conflict with them.
- Report potential bugs separately. Do not silently fix them unless the user
  requests a fix.
- Report only tests and checks actually run. Do not claim verification for
  anything that was not executed.

## Message and Agent flow

- `main.py` owns AstrBot registration, command bypass, request construction,
  response hooks, and thin history/image tool registration methods.
- `src/tools/history.py` and `src/tools/images.py` implement Mingle-required
  function-calling tools. Keep tool names, signatures and model-facing docstrings
  on the registered main.py methods; delegate execution with explicit dependencies.
  Storage stays in src/history.py and image preprocessing in src/utils/image_preprocessing.py.
  Use root-level toolbox/ for additional general-purpose model tools; core
  scheduling, decision and storage modules must not depend on that toolbox.
  main.py registers these tools through the normal AstrBot lifecycle.
  toolbox/time.py implements mingle_get_time with no model-supplied parameters.
  Resolve the session's AstrBot timezone on each call; missing/invalid zones use
  UTC. Include the date, time, numeric UTC offset, timezone and weekday. The tool
  is usable outside managed Mingle turns. The toolbox configuration group
  exposes toolbox_enabled (master switch) and toolbox_time_enabled, both true
  by default. All optional tools must have independent switches under this group.
  Filter disabled tools from a request-local ToolSet before debug capture, never
  mutate global registrations or enable tools excluded by persona settings. Also
  check switches at execution time to reject stale calls. The master switch
  overrides individual settings without changing them. Core Mingle tools and
  AstrBot built-in tools remain unaffected.
- toolbox/calculate.py implements mingle_calculate using a bounded arithmetic
  AST and decimal operations, never eval or arbitrary Python execution. Limit
  input length, AST nodes, exponents and intermediate magnitudes.
  toolbox/random.py implements mingle_random: integers with replacement or
  unique candidate strings without replacement. Bound result counts and input
  sizes. Register both tools in main.py with independent default-true switches
  toolbox_calculate_enabled and toolbox_random_enabled in the toolbox group.
- toolbox/images.py implements mingle_generate_image through the standard
  OpenAI /images/generations endpoint, with one PNG per call. Reuse an explicitly
  selected AstrBot OpenAI-compatible provider's model name, current key, base URL,
  proxy, headers and timeout without mutating its client or configuration. Resolve
  the model name on every call; do not add a separate image model setting.
  Return configuration errors for missing providers
  and unsupported Azure endpoints. Validate bounded image data off-thread, send
  directly to the event's current chat, and return only the send result to the
  model. A successful send call is not a delivery receipt. Do not automatically
  retry paid generation or expose credentials/upstream error bodies. Register
  normally in main.py with the independent default-true switch
  toolbox_image_generate_enabled under the toolbox master switch.
- Keep business modules in the `src/` package. The root `main.py` remains the
  AstrBot entry point and uses relative imports from `.src`. Keep tests in the
  root `tests/` directory and import business modules through the `src` package.
- `src/scheduler.py` owns separate judgment and execution tasks per unified
  session. Judgment merges and evaluates new arrivals while one Agent runs;
  approved batches queue for execution after the active handler fully unwinds.
  Bound both pending and approved event queues and the merge wait.
- Default timing is a three-second proactive cooldown, two-second merge debounce,
  and six-second maximum merge wait. Cooldown starts when an admitted Agent turn
  actually finishes. Preserve ordinary arrivals in the bounded pending queue
  during cooldown and judge them once both cooldown and merge timing allow it.
  Track original arrival times: readiness is max(cooldown expiry,
  min(last arrival + debounce, first arrival + maximum merge wait)). Do not restart
  merging at cooldown expiry, replay raw events, or poll without pending input.
  A declined batch is finished once and does not retry on another cooldown timer.
  Private/explicit wake batches and busy-session judgment bypass cooldown while
  keeping normal merge timing. Agent start/end wake the waiting judgment worker
  so it can recompute eligibility without resetting the merge window.
- Natural sleep uses scheduler.sleep_seconds (default 180, zero disables automatic
  sleep) and literal, case-sensitive scheduler.sleep_wake_keywords. Track the last
  successful managed current-session native send or Mingle image send through the
  normal tool-result hook; incoming messages and unsent final output do not reset
  this clock. Startup and waking grant a full timeout window. State is per full
  UMO and resets on restart. Sleep retains bounded pending tickets for explicit
  wakes or timed backlog reviews, and never interrupts the active Agent. Direct bot
  mentions or keywords wake the session; private messages, @all and framework wake
  prefixes alone do not. The wake batch must be judged, with all retained pending
  records included and only its latest message marked new; the others are history.
  A positive vote admits the complete waiting batch, including records beyond an
  empty conversation's initial context limit. /mingle sleep and /s use native
  administrator permissions and do not enter chat storage or scheduling. Sleep
  invalidates in-flight judgments and prevents recovery from claiming sleeping work.
- scheduler.sleep_backlog_minutes (default 30, zero disables) schedules a
  conservative review of bounded sleeping backlog without waking on arrivals.
  Count from sleep entry or the previous review, wait for any active Agent to
  unwind, and make no requests without pending input. Use wake judgment
  concurrency and append backlog rules to the judgment system prompt, after
  configured instructions and before the persona block, preferring silence unless
  clearly interested. Keep chat data in user input. Retain wake-style full context
  and latest-only new status.
  Declined batches finish once; timed admission leaves the session sleeping until
  a successful managed current-session send. A fresh sleep or explicit wake
  invalidates stale reviews; manual sleep during an active backlog turn prevents
  that turn's later sends from waking it again.
- `src/decision.py` calls a judgment provider without tools. An empty provider ID
  follows AstrBot's current session provider; another configured provider may be
  selected in the plugin settings. Invalid decisions and failures mean silence.
- Use shared formatting for judgment and generator history inputs: format Unix timestamps as
  readable date/time with the configured AstrBot timezone and numeric offset
  (UTC fallback), shorten sender IDs to their last eight characters, and omit
  `platform_message_id`. Work on copies; stored records, Agent history tools,
  and identity/permission checks retain full IDs and original timestamps.
  Invalid timestamps become null. Debug dumps preserve the compact prompt
  string actually passed to each model. Judgment alone adds scheduling status.
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
- The default judgment instructions reject clearly unfinished user input,
  including incomplete sentences and messages being sent in multiple parts,
  with `should_reply=false` so subsequent content can arrive first.
- Generator instructions use generator.generator_prompt and
  generator.generator_prompt_append, concatenated verbatim without trimming,
  interpolation or an implicit separator. Keep the schema default synchronized
  with DEFAULT_GENERATOR_PROMPT. Append these instructions to the system prompt
  through a managed request hook before debug capture, after AstrBot resolves
  the Agent persona. Keep batch metadata and chat JSON in the user prompt;
  prepend the fixed continuation notice there for reruns.
- Generator defaults require user-facing replies through the native
  send_message_to_user tool, including ordinary text replies. Explicitly override
  the native tool description's permission to reply directly in managed turns.
  The model decides message timing and recipients
  naturally within the current chat, and sends a brief starting message before
  long-running tools. Emit exactly <finished> when done; no final-output fallback
  is sent when the tool is unavailable or fails. Do not inject tools beyond native
  availability. Set enable_streaming=False on admitted managed events; do not
  change global streaming settings or unmanaged events.
  Capture the raw final response in debug, then suppress managed final media and
  reasoning in a lower-priority response hook. Preserve final text for AstrBot's
  history-saving guard; clear the final model result in a low-priority native
  result-decoration hook before sending. Log a warning when the trimmed final
  text is not <finished>.
  Do not stop the event. Record only successful current-session native send-tool
  text with per-turn indexed source keys as bot chat rows, never final output.
  Normal Agent history saving continues, including tool calls, results and the
  final completion marker.
- When a managed Agent returns assistant output other than trimmed <finished>,
  allow at most one recovery request after its pipeline saves history. Keep the
  same bound conversation and active scheduler ticket. Do not recover errors,
  stopped/interrupted turns, or requests without the native send_message_to_user
  tool. Recheck handling eligibility before recovery. Atomically claim valid
  pending, judging and approved tickets into the active batch; released judgments
  must not requeue them or apply stale stop/rerun actions. Later arrivals follow
  ordinary scheduling. Reload saved history and prepend this exact user notice:
  The previous turn did not follow the completion protocol: end with exactly
  <finished>. Content in the final response is not sent to the user. Review
  messages already sent successfully and use the send_message_to_user tool to
  send any remaining replies. Avoid duplicate messages and do not mention this
  recovery run.
  If arrivals were claimed, append:
  Meanwhile, the following new messages have arrived. You may reply to them
  separately or together:
  Keep the snapshot cursor format and assign a fresh turn ID and send-text list
  per request so successful sends are recorded once. Release the batch and start
  cooldown only after recovery has also fully unwound.
- `src/history.py` owns an independent SQLite database and image cache under
  `StarTools.get_data_dir("astrbot_plugin_mingle")`.
- `src/settings.py` validates immutable runtime settings. `_conf_schema.json` exposes
  these settings in AstrBot's configuration UI. Use one level of object groups
  in this order: general (基础设置), scheduler (调度器), judge (判别器),
  generator (生成器), history (聊天记录), images (图片缓存),
  toolbox (额外工具箱), followed by tool-specific configuration groups named
  after the actual registered tools, then debug (调试) and experimental (实验功能). Keep the master and individual enable
  switches in toolbox; place other tool settings in their own sections. Image generation and
  image editing each have such a section.
  CONFIG_GROUPS maps
  grouped configuration to flat immutable runtime fields. Flat legacy settings
  are unsupported; do not add migration or fallback logic for this test version.
  Management commands persist general.group_sessions and debug.debug_sessions
  while preserving other fields in those objects.
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
  While the plugin is enabled and initialized, suppress ordinary group messages
  outside this allowlist, including mentions/wake messages: disable the native
  LLM fallback and stop ordinary handler propagation without recording or
  scheduling them. Recognized commands bypass this gate. Disabling the plugin
  restores native framework behavior.
- Register `/mingle` through AstrBot's native command_group with group, debug
  and barrier subcommands. Keep all management handlers administrator-only; the root group is public
  so all members can use `/mingle whisper`. Register `/w` as its standalone
  shorthand. Both silently stop the entire event before Mingle storage or
  scheduling, including empty bodies and attachments, without calling models. Use the framework's command parsing and help rendering;
  do not patch the framework command-group help renderer. Put permission
  decorators above subcommand decorators so native registration creates the
  handler with its sub_command metadata before attaching permissions.
- `/mingle group on|off` is restricted to AstrBot administrators through its
  registered handler. Reject private-chat use.
  Enable stores a platform-scoped group UMO; match that group UMO even when
  AstrBot isolates member conversations. Disable removes the group UMO, current
  event UMO, and legacy bare group ID, preserving unrelated allowlist entries.
  Save the plugin config asynchronously under a lock before updating runtime
  settings. On failure restore the in-memory config. Waiting events recheck
  the allowlist before requesting an Agent; active turns are not interrupted.
- `/mingle barrier set|find|clear` is administrator-only and group-only.
  `find` shows the current session’s nearest retained message on either side,
  each prefixed by its timestamp in the session timezone (UTC fallback), with
  text limited to 25 characters including the ellipsis, separated by `--- barrier ---`; missing sides
  use a placeholder. Keep one persistent boundary per platform-scoped group
  in Mingle storage; setting it again replaces the previous boundary. Filter Mingle's Agent prompt history,
  history tool and image tool to records after it, including isolated member
  sessions. Clearing restores access to retained records. Do not change AstrBot's
  conversation history or the judgment-model history.
- Private messages are enabled by default and use the same scheduler. WebChat
  remains on its native flow. When idle and awake, private and explicit wake/mention
  events bypass judgment and cooldown. While busy, evaluate them for control
  actions too, but preserve their reply admission unless the action is stop.
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
- New arrivals are persisted immediately and evaluated in parallel with the
  active Agent. `should_reply` plus `action=wait` queues ordinary follow-ups;
  `action=rerun` requires should_reply=true, requests stopping the active turn,
  then starts a continuation after its pipeline completes history saving.
  `action=stop` requires should_reply=false and discards approved waiting work
  without creating a replacement. Invalid, failed or timed-out decisions never
  request stop/rerun. Legacy decisions without action default to wait. Custom
  judgment prompts must implement the wait/rerun/stop protocol to control runs.
- Stop only the snapshotted active event using `agent_stop_requested`, which
  AstrBot's runner watcher consumes. Do not call stop_event or cancel the old
  pipeline for normal rerun/stop: history saving must finish. Recheck active
  event identity after judgment; a late result cannot stop a newer turn. Do not
  patch same-sender follow-up capture or stop every event in the shared session.
- Judgment user input includes messages followed by time_context with the current
  session-local time (UTC fallback) and the age of the last message preceding
  this judgment's first new message in
  whole seconds, minutes, hours or days. Missing/invalid timestamps yield unknown;
  future timestamps yield zero seconds. Keep this dynamic context out of system.
  Each supplied message
  has only one scheduling field, status: new (this judgment's input), queued
  (reply approved), processing (active Agent batch), or history (other context).
  Do not send per-message is_new/sent_to_agent flags or top-level
  new_message_ids/message_status lists. History includes completed, interrupted,
  skipped and untracked records; it does not claim a successful reply.
  Keep detailed outcomes internal to the scheduler. Supplement recent history
  with retained active-batch records from SQLite when outside the recent window.
  Annotate copies without mutating stored history.
  Finish outcomes are bounded runtime metadata, not
  a durable audit log; older/untracked records must not be claimed completed.
  Dynamic status stays out of the stable system prefix.
- AstrBot session isolation is respected. To share a group's conversation across
  members, disable its `unique_session` option. Do not override isolation here.
- Register Mingle tools normally. If a persona restricts its tools,
  explicitly allow `mingle_chat_history` and `mingle_view_image` in that persona.
  Allow optional toolbox tools such as mingle_get_time there when wanted.
  Do not bypass persona tool restrictions by silently injecting tools.
- Tool docstring Args must list only model-supplied arguments with explicit
  Google-style types. AstrBot injects `event`; listing it in Args makes this
  version's tool schema parser treat it as a model parameter.

## Storage and attachments

- Redact newly recorded message bodies before truncation and SQLite insertion
  when history.redaction_enabled is true (default). Apply the same pure-code
  rules to user text, quoted text and bot text. Keep rules in src/utils/redaction.py.
  Preserve identity, deduplication and reply metadata. Do not mutate original
  events or rewrite existing records. This is text storage redaction, not a
  global model-input or attachment filter; framework history and debug files
  are outside its scope. Never log matched originals.
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
- `src/utils/image_preprocessing.py` prepares model-facing copies with AstrBot's public
  `compress_image` utility: longest edge 1280 pixels and JPEG quality 85.
  Convert GIFs to a static first frame before compression. Preserve original
  cached bytes, report the actual output MIME type, and remove compressor temp
  files after reading. Compression does not enforce a hard byte-size limit.
  Preparation errors return a tool error instead of sending an oversized original.
- On each Mingle-managed turn, remove historical image_url parts in
  `on_llm_request` before the debug capture and Agent reset. Replace images with
  text placeholders, preserving existing text, tool calls/results and checkpoints.
  Clean both request contexts and the conversation's in-memory history snapshot;
  do not overwrite the database before the turn. AstrBot saves the cleaned runtime
  history through its normal completion/interruption flow. Keep current request
  attachments and images returned by tools during this turn. Those images remain
  usable across steps until the next turn. Do not apply this policy to unmanaged
  native/debug-only requests or delete original cache files.
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

- Agent prompts attach initial_context_messages (default 30) only for an empty
  AstrBot conversation; subsequent turns attach records after the latest saved
  Mingle snapshot cursor through the current batch. Persist the cursor in the
  normal user prompt so resets and restarts follow conversation history. The
  same setting continues to bound the judgment context window.

- When an actual judgment returns should_reply=true with action=wait, atomically
  fold the currently pending valid tickets into that approved batch, both idle
  and busy. Later arrivals still require normal judgment. Declines, forced-only
  admission after a negative judgment, idle judgment bypass, rerun and stop do
  not fold pending tickets. Preserve the approved queue bound and wait for the
  active pipeline to unwind before execution.

- tests/conftest.py assigns a disposable ASTRBOT_ROOT before framework imports
  in the test process and restores it at teardown. Keep test import side effects
  out of the plugin checkout and real framework data directories.

- toolbox/images.py also implements mingle_edit_image using the standard OpenAI
  /images/edits endpoint with one to four current-session cached references,
  respecting managed-turn access and group boundaries. Reuse image preparation
  and a shared private request/validation/send function, with separate generation
  and editing entry functions registered as two tools. Bound combined prepared uploads to
  20 MiB. Keep an independent default-true toolbox_image_edit_enabled switch and
  provider selection under mingle_edit_image. Never accept model-supplied paths
  or sessions, retry paid edits automatically, or return images to the model.

- Experimental settings appear in the final experimental (实验功能) group.
  judge_parallel_requests controls awake judgments; judge_wake_parallel_requests
  controls the batch that wakes a sleeping session. Both are positive integers,
  default 1, under experimental. Pass the scheduler's captured waking flag to
  the judge; do not infer it from mentions or mutate shared settings. Each judgment sends
  N concurrent requests with identical resolved inputs and independent timeouts
  and debug request IDs, then waits for all results. Start requests at 200 ms
  offsets, with the first starting immediately; stagger delays precede debug
  capture and per-request timeout. Batch cancellation cancels delayed requests too.
  Failed or invalid requests
  contribute false/wait votes. Vote should_reply first, then count all action
  votes compatible with that winner (wait plus rerun for true, stop for false).
  Choose uniformly among tied highest counts at each level.
