import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mingle_tests_plugin.toolbox import images as image_edit


@pytest.mark.parametrize("available", [True, False])
async def test_edit_authorizes_every_reference_before_api(monkeypatch, available):
    event = SimpleNamespace(
        unified_msg_origin="test:GroupMessage:123",
        get_extra=lambda key, default=None: True,
        get_group_id=lambda: "123",
        is_private_chat=lambda: False,
        get_platform_id=lambda: "test",
    )
    history = SimpleNamespace(
        read_image=AsyncMock(
            side_effect=[
                (b"original", "image/png"),
                (b"second", "image/png") if available else None,
            ]
        )
    )
    prepare = AsyncMock(return_value=(b"prepared", "image/jpeg"))
    generate = AsyncMock(return_value='{"sent": true}')
    monkeypatch.setattr(image_edit, "prepare_image", prepare)
    monkeypatch.setattr(image_edit, "_request_image", generate)
    result = json.loads(
        await image_edit.edit_image(
            event,
            history,
            object(),
            logging.getLogger(__name__),
            prompt="Use image 2 style on image 1",
            record_ids=[10, 11],
            image_indices=[0, 1],
        )
    )
    assert history.read_image.await_count == 2
    for call in history.read_image.await_args_list:
        assert call.args[0] == event.unified_msg_origin
        assert call.kwargs["boundary"] == "test:GroupMessage:123"
    if available:
        assert result["sent"]
        assert generate.await_args.kwargs["reference_images"] == [
            ("reference_1.jpg", b"prepared", "image/jpeg"),
            ("reference_2.jpg", b"prepared", "image/jpeg"),
        ]
    else:
        assert "error" in result
        generate.assert_not_awaited()


@pytest.mark.parametrize(
    "managed,records,indices",
    [
        (False, [1], [0]),
        (True, [], []),
        (True, [1], []),
        (True, [True], [0]),
        (True, [1], [-1]),
        (True, [1] * 5, [0] * 5),
    ],
)
async def test_invalid_references_do_not_read_or_pay(managed, records, indices):
    history = SimpleNamespace(read_image=AsyncMock())
    event = SimpleNamespace(get_extra=lambda key, default=None: managed)
    result = json.loads(
        await image_edit.edit_image(
            event,
            history,
            None,
            logging.getLogger(__name__),
            prompt="Edit",
            record_ids=records,
            image_indices=indices,
        )
    )
    assert "error" in result
    history.read_image.assert_not_awaited()
