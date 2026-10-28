import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from mingle_tests_plugin.toolbox import time as time_tool
from mingle_tests_plugin.toolbox.calculate import calculate
from mingle_tests_plugin.toolbox.random import draw_random


@pytest.mark.parametrize(
    ("zone", "expected", "weekday"),
    [
        ("Asia/Shanghai", "2026-10-28 01:30:00+08:00", "Wednesday"),
        ("UTC", "2026-10-27 17:30:00+00:00", "Tuesday"),
        ("America/New_York", "2026-10-27 13:30:00-04:00", "Tuesday"),
    ],
)
def test_time_tool_respects_timezone_date_boundary_and_offset(
    monkeypatch, zone, expected, weekday
):
    fixed = datetime(2026, 10, 27, 17, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(
        time_tool, "datetime", SimpleNamespace(now=lambda tz: fixed.astimezone(tz))
    )
    result = json.loads(time_tool.get_current_time(zone))
    assert result == {"datetime": expected, "timezone": zone, "weekday": weekday}


def test_time_tool_falls_back_to_utc_for_invalid_timezone(monkeypatch, caplog):
    fixed = datetime(2026, 10, 27, 17, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(
        time_tool, "datetime", SimpleNamespace(now=lambda tz: fixed.astimezone(tz))
    )
    result = json.loads(time_tool.get_current_time("Invalid/Zone"))
    assert result["timezone"] == "UTC"
    assert result["datetime"] == "2026-10-27 17:30:00+00:00"
    assert "using UTC" in caplog.text


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("0.1 + 0.2", "0.3"),
        ("(120 + 80) * 15 / 100", "30"),
        ("2 ** -3", "0.125"),
        ("-7 // 3", "-3"),
        ("-7 % 3", "2"),
        ("7 % -3", "-2"),
        ("1 / 3", "0.3333333333333333333333333333"),
        ("+2 - 2", "0"),
        (" 1e2 / 4 ", "25"),
    ],
)
def test_calculator_arithmetic(expression, expected):
    assert json.loads(calculate(expression))["result"] == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",
        "(1).__class__",
        "[1, 2]",
        "True + 1",
        "abs(-1)",
        "'hello' * 2",
        "1 << 2",
        "1 / 0",
        "1 % 0",
        "2 ** 101",
        "9 ** 9 ** 9",
        "4 ** 0.5",
        "1e101",
        "1e-100000000",
        "1e-100 * 1e-100",
        "0xFF",
        "1 +",
        "",
        " ",
        "1" * 257,
        "+".join(["1"] * 30),
        None,
    ],
)
def test_calculator_rejects_code_and_excessive_work(expression):
    assert "error" in json.loads(calculate(expression))


def test_random_integer_bounds_and_candidates():
    result = json.loads(draw_random(-3, 3, 100))
    assert result["mode"] == "integer"
    assert len(result["results"]) == 100
    assert all(type(value) is int and -3 <= value <= 3 for value in result["results"])
    assert json.loads(draw_random(2, 2, 3))["results"] == [2, 2, 2]
    choices = ["甲", "乙", "丙"]
    result = json.loads(draw_random(count=3, choices=choices))
    assert result["mode"] == "choices"
    assert sorted(result["results"]) == sorted(choices)
    assert choices == ["甲", "乙", "丙"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum": 2, "maximum": 1},
        {"minimum": True},
        {"maximum": 1e20},
        {"count": 0},
        {"count": 101},
        {"count": True},
        {"choices": []},
        {"choices": "abc"},
        {"choices": [1]},
        {"choices": [" "]},
        {"choices": ["a", "a"]},
        {"choices": ["a"], "count": 2},
        {"choices": ["a" * 201]},
        {"choices": [str(i) for i in range(101)]},
    ],
)
def test_random_rejects_invalid_arguments(kwargs):
    assert "error" in json.loads(draw_random(**kwargs))
