import json
from copy import deepcopy

import pytest
from astrbot.core.agent.message import (
    bind_checkpoint_messages,
    dump_messages_with_checkpoints,
)
from mingle_tests_plugin.src.generation import (
    agent_history_cursor,
    remove_message_turns,
)


def make_turn(turn_id, batch_ids, snapshot, records=None, prefix="", run_id=None):
    if run_id is None:
        run_id = turn_id
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prefix
                    + f"New batch record IDs: {batch_ids}. "
                    + f"Snapshot ends at record {snapshot}.\n"
                    + json.dumps(records or []),
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": f"tool-{turn_id}",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"tool-{turn_id}", "content": "tool result"},
        {"role": "assistant", "content": "<finished>"},
        {"role": "_checkpoint", "content": {"id": f"mingle:turn:{run_id}:{turn_id}"}},
    ]


def test_remove_all_linked_requests_keeps_later_tools_and_native_checkpoints():
    first = make_turn("first", [1, 2], 2)
    recovery = make_turn(
        "recovery", [1, 2, 3], 3, prefix="Recovery notice\n\n", run_id="first"
    )
    later = make_turn("later", [4], 4)
    original = first + recovery + later
    snapshot = deepcopy(original)
    edited, count = remove_message_turns(
        original, {"id": 1, "role": "user", "source_key": "user:alice:1"}
    )
    assert count == 2
    assert original == snapshot
    assert edited[1:] == later
    assert agent_history_cursor(edited) == 4
    runtime = bind_checkpoint_messages(edited)
    assert [item.role for item in runtime] == ["user", "assistant", "tool", "assistant"]
    assert runtime[1].tool_calls[0].id == runtime[2].tool_call_id == "tool-later"
    assert dump_messages_with_checkpoints(runtime) == later


def test_bot_record_uses_source_identity_even_when_outputs_are_identical():
    first = make_turn("first", [1], 1)
    later = make_turn("later", [2], 2)
    edited, count = remove_message_turns(
        first + later, {"id": 8, "role": "bot", "source_key": "bot:first:sent:0"}
    )
    assert count == 1
    assert edited[1:] == later


@pytest.mark.parametrize("turn_id", ["first", "recovery"])
def test_selecting_either_bot_reply_removes_the_whole_managed_run(turn_id):
    first = make_turn("first", [1], 1, run_id="shared")
    recovery = make_turn("recovery", [1, 2], 2, run_id="shared")
    later = make_turn("later", [3], 3)
    edited, count = remove_message_turns(
        first + recovery + later,
        {"id": 8, "role": "bot", "source_key": f"bot:{turn_id}:sent:0"},
    )
    assert count == 2
    assert edited[1:] == later


def test_recovery_input_selects_its_run_and_rejects_a_missing_input_boundary():
    first = make_turn("first", [1], 1, run_id="shared")
    recovery = make_turn("recovery", [1, 2], 2, run_id="shared")
    record = {"id": 2, "role": "user", "source_key": "user:alice:2"}
    edited, count = remove_message_turns(first + recovery, record)
    assert count == 2
    assert bind_checkpoint_messages(edited) == []
    first[0]["content"] = "Only a compressed input remains"
    with pytest.raises(ValueError, match="input boundary"):
        remove_message_turns(first + recovery, record)


@pytest.mark.parametrize("kind", ["legacy", "summary", "missing_header"])
def test_unidentifiable_history_is_never_removed(kind):
    history = make_turn("new", [1], 1)
    if kind == "legacy":
        history.pop()
    elif kind == "summary":
        history = [{"role": "user", "content": "Our previous history summary"}]
    else:
        history[0]["content"] = "An input without a Mingle header"
    edited, count = remove_message_turns(
        history, {"id": 1, "role": "user", "source_key": "user:alice:1"}
    )
    assert count == 0
    assert edited == history


def test_first_new_turn_does_not_include_preceding_legacy_history():
    legacy = make_turn("old", [1], 1)[:-1]
    current = make_turn("new", [2], 2)
    edited, count = remove_message_turns(
        legacy + current, {"id": 2, "role": "user", "source_key": "user:alice:2"}
    )
    assert count == 1
    assert edited[1:] == legacy


def test_initial_context_match_and_deleting_last_turn_preserve_snapshot():
    history = make_turn("first", [3], 3, records=[{"id": 1, "text": "initial context"}])
    edited, count = remove_message_turns(
        history, {"id": 1, "role": "user", "source_key": "user:alice:1"}
    )
    assert count == 1
    assert agent_history_cursor(edited) == 3
    assert bind_checkpoint_messages(edited) == []
    assert agent_history_cursor([]) == 0
