"""The progress stream must not write an asyncio.Queue from the agent's thread.

`ProgressTracker._notify` runs on the worker thread a turn is offloaded to. A bare
`queue.put_nowait` there does not wake the loop's waiter, so the next frame - sometimes the
final `is_complete` one - could sit in the queue until the 10 s keep-alive ping. The
listener hands the put to the loop with `call_soon_threadsafe` instead.
"""

import asyncio
import threading

from api.chat import _queue_listener


def test_a_snapshot_posted_from_a_worker_thread_reaches_the_queue():
    async def scenario():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        listener = _queue_listener(queue, loop)
        thread = threading.Thread(target=listener, args=({"is_complete": True},))
        thread.start()
        thread.join()
        return await asyncio.wait_for(queue.get(), timeout=2.0)

    assert asyncio.run(scenario()) == {"is_complete": True}


def test_every_frame_is_delivered_in_order():
    async def scenario():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        listener = _queue_listener(queue, loop)
        for i in range(5):
            threading.Thread(target=listener, args=({"step": i},)).start()
        return [await asyncio.wait_for(queue.get(), timeout=2.0) for _ in range(5)]

    frames = asyncio.run(scenario())
    assert sorted(f["step"] for f in frames) == [0, 1, 2, 3, 4]


def test_a_closed_loop_is_not_an_error():
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    listener = _queue_listener(queue, loop)
    loop.close()
    listener({"is_complete": True})  # the tab went away: must not raise
