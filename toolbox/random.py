"""Bounded random integer generation and sampling for chat tools."""

import json
import secrets


def draw_random(
    minimum: int = 1,
    maximum: int = 6,
    count: int = 1,
    choices: list[str] | None = None,
) -> str:
    """Draw integers with replacement or candidate strings without replacement.

    Args:
        minimum: Inclusive integer lower bound, at least -1e12.
        maximum: Inclusive integer upper bound, at most 1e12.
        count: Number of results, between 1 and 100.
        choices: Optional nonempty list of up to 100 unique candidate strings.

    Returns:
        JSON with the draw mode and results, or a validation error.
    """
    if type(count) is not int or not 1 <= count <= 100:
        return json.dumps({"error": "Count must be an integer between 1 and 100."})
    if choices is not None:
        if (
            not isinstance(choices, list)
            or not 1 <= len(choices) <= 100
            or any(
                not isinstance(item, str) or not item.strip() or len(item) > 200
                for item in choices
            )
        ):
            return json.dumps(
                {
                    "error": "Choices must contain 1 to 100 nonempty strings, each at most 200 characters."
                }
            )
        if len(set(choices)) != len(choices) or count > len(choices):
            return json.dumps(
                {
                    "error": "Choices must be unique and count must not exceed their number."
                }
            )
        results = secrets.SystemRandom().sample(choices, count)
        return json.dumps({"mode": "choices", "results": results}, ensure_ascii=False)
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or not -(10**12) <= minimum <= maximum <= 10**12
    ):
        return json.dumps(
            {
                "error": "Integer bounds must satisfy -1e12 <= minimum <= maximum <= 1e12."
            }
        )
    results = [minimum + secrets.randbelow(maximum - minimum + 1) for _ in range(count)]
    return json.dumps(
        {"mode": "integer", "minimum": minimum, "maximum": maximum, "results": results}
    )
