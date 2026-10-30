"""Wait for observable scheduling transitions rather than removed worker tasks."""

import asyncio


async def wait_for_judgment(state):
    async with asyncio.timeout(1):
        while state.pending or state.judging:  # noqa: ASYNC110 - Observe loop progress.
            await asyncio.sleep(0)


async def wait_for_execution(state):
    async with asyncio.timeout(1):
        while state.active is not None or state.approved:  # noqa: ASYNC110
            await asyncio.sleep(0)
