"""Queue removal is not frame handoff or client receipt."""

from openhop_core.companion.message_queue import MessageQueue
from openhop_core.companion.models import QueuedMessage


def test_queue_operations_do_not_claim_frame_handoff():
    queue = MessageQueue(max_size=2)
    first = QueuedMessage(b"a", is_channel=True)
    second = QueuedMessage(b"b")
    queue.push(first)
    queue.push(second)
    assert queue.pop() is first
    assert not first.frame_enqueued
    queue.restore_front(first)
    assert queue.peek() is first
    assert queue.pop_last() is second
    assert not second.frame_enqueued
    assert queue.remove(first)
    assert not first.frame_enqueued


def test_frame_admission_marks_only_success_and_preserves_fifo():
    from types import SimpleNamespace
    from openhop_core.companion.frame_server import CompanionFrameServer

    queue = MessageQueue(max_size=2)
    first = QueuedMessage(b"a", text="first")
    second = QueuedMessage(b"b", text="second")
    queue.push(first)
    queue.push(second)
    bridge = SimpleNamespace(message_queue=queue, sync_next_message=queue.pop)
    server = CompanionFrameServer(bridge, "hash", port=0)
    server._write_frame = lambda frame: False
    assert server._sync_memory_message()
    assert queue.peek() is first and queue.count == 2
    assert not first.frame_enqueued
    server._write_frame = lambda frame: True
    assert server._sync_memory_message()
    assert first.frame_enqueued and queue.peek() is second


def test_stale_command_does_not_write_to_replacement_or_block_other_companion():
    import asyncio
    from types import SimpleNamespace
    from openhop_core.companion.frame_server import CompanionFrameServer
    from openhop_core.companion.frame_server.transport import _command_queue

    server = CompanionFrameServer(SimpleNamespace(), "01")
    other = CompanionFrameServer(SimpleNamespace(), "02")
    old = asyncio.Queue()
    server._write_queue = asyncio.Queue()
    server._client_writer = _OpenWriter()
    other._write_queue = asyncio.Queue()
    other._client_writer = _OpenWriter()
    token = _command_queue.set((server, old))
    try:
        assert not server._enqueue_frame(b"old response")
        assert server._write_queue.empty()
        assert other._enqueue_frame(b"independent push")
    finally:
        _command_queue.reset(token)
    assert server._enqueue_frame(b"current push")


class _OpenWriter:
    """A live attached-transport double for tests that stub the write queue."""

    def write(self, _data):
        return None

    async def drain(self):
        return None

    def close(self):
        return None

    async def wait_closed(self):
        return None

    def is_closing(self):
        return False
