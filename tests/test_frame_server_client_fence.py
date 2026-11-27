"""A command that outlived its client must not write into the successor.

Behavioural end-to-end pin for the client-replacement fence: client A issues
a sync command that suspends, client B connects and evicts A, and A's
resumed command is refused - the queued message rolls back into the offline
queue and is delivered to B instead.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from collections import deque
from unittest.mock import Mock

import pytest
from openhop_core.companion.constants import (
    CMD_SYNC_NEXT_MESSAGE,
    FRAME_INBOUND_PREFIX,
)
from openhop_core.companion.frame_server import CompanionFrameServer
from openhop_core.companion.message_queue import MessageQueue
from openhop_core.companion.models import QueuedMessage


def _command_frame(payload: bytes) -> bytes:
    return bytes([FRAME_INBOUND_PREFIX]) + struct.pack("<H", len(payload)) + payload


class _ScriptedReader:
    """Reader that replays scripted bytes, blocks when idle, EOF on close."""

    def __init__(self):
        self._chunks: deque[int] = deque()
        self._data = asyncio.Event()
        self._eof = False

    def feed(self, data: bytes) -> None:
        self._chunks.extend(data)
        self._data.set()

    def close(self) -> None:
        self._eof = True
        self._data.set()

    async def read(self, _n: int) -> bytes:
        while True:
            if self._chunks:
                return bytes([self._chunks.popleft()])
            if self._eof:
                return b""
            self._data.clear()
            if self._chunks:  # a feed raced the clear
                continue
            await self._data.wait()

    async def readexactly(self, n: int) -> bytes:
        if len(self._chunks) < n:
            raise asyncio.IncompleteReadError(partial=b"", expected=n)
        return bytes([self._chunks.popleft() for _ in range(n)])


class _RecordingWriter:
    """Minimal writer that records every frame written for its client."""

    def __init__(self):
        self.written: list[bytes] = []
        self.closed = False

    def get_extra_info(self, _name):
        return None

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self.closed


class _GatedWriter(_RecordingWriter):
    """Writer double whose ``wait_closed()`` stalls until released.

    An attach that is evicting this writer's client therefore stays suspended
    inside its attach section, holding the attach lock, until the test releases
    the gate — a deterministic stand-in for the eviction's real awaits.
    """

    def __init__(self):
        super().__init__()
        self.release_wait_closed = asyncio.Event()

    async def wait_closed(self) -> None:
        await self.release_wait_closed.wait()


async def _wait_until(condition, spins: int = 500) -> None:
    for _ in range(spins):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not reached within the spin budget")


@pytest.mark.asyncio
async def test_evicted_client_command_is_refused_and_message_goes_to_successor():
    queue = MessageQueue(max_size=4)
    entry = QueuedMessage(sender_key=b"k" * 32, text="queued text")
    queue.push(entry)
    bridge = Mock()
    bridge.get_time = Mock(return_value=0)
    bridge.sync_next_message = queue.pop
    bridge.message_queue = queue
    server = CompanionFrameServer(bridge, "hash", port=0, client_idle_timeout_sec=None)

    original_sync = server._cmd_handlers[CMD_SYNC_NEXT_MESSAGE]
    command_running = asyncio.Event()
    command_done = asyncio.Event()
    release_command = asyncio.Event()

    async def gated_sync(data):
        command_running.set()
        await release_command.wait()
        try:
            await original_sync(data)
        finally:
            command_done.set()

    server._cmd_handlers[CMD_SYNC_NEXT_MESSAGE] = gated_sync

    reader_a, writer_a = _ScriptedReader(), _RecordingWriter()
    handler_a = asyncio.create_task(server._handle_client(reader_a, writer_a))
    await _wait_until(lambda: server._client_writer is writer_a and server._writer_task is not None)
    queue_a = server._write_queue

    reader_a.feed(_command_frame(bytes([CMD_SYNC_NEXT_MESSAGE])))
    await _wait_until(lambda: command_running.is_set())

    # Client B connects and evicts A while A's command is suspended.
    reader_b, writer_b = _ScriptedReader(), _RecordingWriter()
    handler_b = asyncio.create_task(server._handle_client(reader_b, writer_b))
    await _wait_until(lambda: server._client_writer is writer_b)
    assert server._write_queue is not queue_a

    # A's resumed command pops the message, but the fence refuses the frame:
    # the message rolls back to the front of the offline queue.
    release_command.set()
    await _wait_until(lambda: command_done.is_set())
    assert queue.peek() is entry and queue.count == 1
    assert not entry.frame_enqueued
    assert writer_a.written == []

    # B asks and receives exactly the message A could not claim.
    reader_b.feed(_command_frame(bytes([CMD_SYNC_NEXT_MESSAGE])))
    await _wait_until(lambda: queue.count == 0)
    await _wait_until(lambda: writer_b.written)
    frames = b"".join(writer_b.written)
    assert frames.endswith(b"queued text")

    # Both handlers end cleanly; A's stale cleanup must not disturb B.
    reader_a.close()
    reader_b.close()
    await _wait_until(lambda: handler_a.done() and handler_b.done())
    await asyncio.gather(handler_a, handler_b)
    assert writer_b.written[-1].endswith(b"queued text")


def _queued_server():
    queue = MessageQueue(max_size=4)
    bridge = Mock()
    bridge.get_time = Mock(return_value=0)
    bridge.sync_next_message = queue.pop
    bridge.message_queue = queue
    return CompanionFrameServer(bridge, "hash", port=0, client_idle_timeout_sec=None)


@pytest.mark.asyncio
async def test_client_accepted_after_shutdown_never_starts_on_the_stopped_server():
    """A handler that runs once stop() has finished must not attach a client.

    The listener can hand a connection accepted before the close to its handler
    after the stop completed; the server is going away, so the handler must
    close that socket cleanly and leave the per-client state retired instead of
    starting a writer task and re-subscribing push callbacks.
    """
    server = _queued_server()

    reader_a, writer_a = _ScriptedReader(), _RecordingWriter()
    handler_a = asyncio.create_task(server._handle_client(reader_a, writer_a))
    await _wait_until(lambda: server._client_writer is writer_a)

    callbacks_after_attach = server.bridge.add_push_callback.call_count
    await server.stop()
    assert writer_a.closed
    assert server._client_writer is None and server._write_queue is None
    assert server._writer_task is None

    reader_b, writer_b = _ScriptedReader(), _RecordingWriter()
    handler_b = asyncio.create_task(server._handle_client(reader_b, writer_b))
    await _wait_until(lambda: handler_b.done())
    await asyncio.gather(handler_b)
    assert writer_b.closed and writer_b.written == []
    assert server._client_writer is None and server._write_queue is None
    assert server._writer_task is None
    assert server.bridge.add_push_callback.call_count == callbacks_after_attach

    reader_a.close()  # the attached client still drains its read loop
    await _wait_until(lambda: handler_a.done())
    await asyncio.gather(handler_a)


@pytest.mark.asyncio
async def test_shutdown_and_a_new_client_settle_in_attach_order():
    """stop() and a concurrent handler serialize on the attach lock.

    While stop() tears the current client down, a handler accepted in that
    window waits on the lock and is refused once shutdown is visible; it never
    attaches onto the closing server's retired per-client state.
    """
    server = _queued_server()

    reader_a, writer_a = _ScriptedReader(), _RecordingWriter()
    handler_a = asyncio.create_task(server._handle_client(reader_a, writer_a))
    await _wait_until(lambda: server._client_writer is writer_a)

    # Hold the attach lock so stop() and the new handler queue behind it in
    # start order: stop settles the teardown first, the handler is refused.
    attach_lock = server._client_attach_lock
    await attach_lock.acquire()
    stop_task = asyncio.create_task(server.stop())
    reader_b, writer_b = _ScriptedReader(), _RecordingWriter()
    handler_b = asyncio.create_task(server._handle_client(reader_b, writer_b))
    await _wait_until(lambda: not stop_task.done() and not handler_b.done())
    assert not writer_b.closed  # still waiting, nothing started yet

    attach_lock.release()
    await _wait_until(lambda: stop_task.done() and handler_b.done())
    await asyncio.gather(stop_task, handler_b)
    assert writer_b.closed and writer_b.written == []
    assert server._client_writer is None and server._write_queue is None
    assert server._writer_task is None
    assert server._closing

    reader_a.close()
    await _wait_until(lambda: handler_a.done())
    await asyncio.gather(handler_a)


@pytest.mark.asyncio
async def test_a_handler_queued_ahead_of_stop_is_refused(caplog):
    """A handler queued on the attach lock before stop() is still refused.

    asyncio locks are fair: a handler that began waiting on the attach lock
    before stop() was called acquires it ahead of stop() itself. Because
    stop() retires admission before it queues on the lock, that handler finds
    shutdown already begun and is refused exactly like a handler arriving
    after shutdown; it never publishes a client on the stopping server.
    """
    caplog.set_level(logging.INFO, logger="CompanionFrameServer")
    server = _queued_server()

    # An attached client whose eviction stalls inside the next handler's
    # attach section: the stall holds the attach lock deterministically.
    reader_0, writer_0 = _ScriptedReader(), _GatedWriter()
    handler_0 = asyncio.create_task(server._handle_client(reader_0, writer_0))
    await _wait_until(lambda: server._client_writer is writer_0)
    callbacks_after_attach = server.bridge.add_push_callback.call_count

    reader_a, writer_a = _ScriptedReader(), _RecordingWriter()
    handler_a = asyncio.create_task(server._handle_client(reader_a, writer_a))
    # A's attach reached the eviction's wait_closed(): it now holds the lock.
    await _wait_until(lambda: writer_0.closed)

    # The second handler queues on the attach lock; stop() then queues behind
    # it — task scheduling is FIFO, so stop() can never overtake this handler.
    reader_b, writer_b = _ScriptedReader(), _RecordingWriter()
    handler_b = asyncio.create_task(server._handle_client(reader_b, writer_b))
    stop_task = asyncio.create_task(server.stop())
    # stop() retires admission in its first step, before it queues on the lock
    # the in-flight attach holds.
    await _wait_until(lambda: server._closing)
    assert not writer_b.closed  # queued on the lock ahead of stop(), not refused yet

    writer_0.release_wait_closed.set()
    await stop_task
    await _wait_until(lambda: handler_a.done() and handler_b.done())
    await asyncio.gather(handler_a, handler_b)

    assert writer_b.closed and writer_b.written == []
    assert server._client_writer is None
    assert server._client_reader is None
    assert server._write_queue is None
    assert server._writer_task is None
    assert server.bridge.add_push_callback.call_count == callbacks_after_attach
    assert [r for r in caplog.records if "refused during shutdown" in r.getMessage()]
    connect_records = [
        r for r in caplog.records if "Companion client connected" in r.getMessage()
    ]
    assert len(connect_records) == 1  # only the original client ever attached

    reader_0.close()
    await _wait_until(lambda: handler_0.done())
    await asyncio.gather(handler_0)


@pytest.mark.asyncio
async def test_a_handler_mid_attach_when_stop_is_called_does_not_publish_a_client(caplog):
    """A handler already inside its attach section when stop() runs is refused.

    The attach section awaits the eviction of the current client while holding
    the attach lock, so shutdown can begin during that await. The section
    re-checks the closing flag after the await and closes the socket cleanly
    instead of publishing a client onto the stopping server.
    """
    caplog.set_level(logging.INFO, logger="CompanionFrameServer")
    server = _queued_server()

    reader_0, writer_0 = _ScriptedReader(), _GatedWriter()
    handler_0 = asyncio.create_task(server._handle_client(reader_0, writer_0))
    await _wait_until(lambda: server._client_writer is writer_0)
    callbacks_after_attach = server.bridge.add_push_callback.call_count

    # The next handler is inside its attach section, stalled on the eviction.
    reader_a, writer_a = _ScriptedReader(), _RecordingWriter()
    handler_a = asyncio.create_task(server._handle_client(reader_a, writer_a))
    await _wait_until(lambda: writer_0.closed)

    stop_task = asyncio.create_task(server.stop())
    # stop() retires admission in its first step, while the attach still holds
    # the lock inside its eviction await.
    await _wait_until(lambda: server._closing)
    writer_0.release_wait_closed.set()
    await stop_task
    await _wait_until(lambda: handler_a.done())
    await asyncio.gather(handler_a)

    assert writer_a.closed and writer_a.written == []
    assert server._client_writer is None
    assert server._client_reader is None
    assert server._write_queue is None
    assert server._writer_task is None
    assert server.bridge.add_push_callback.call_count == callbacks_after_attach
    assert [r for r in caplog.records if "refused during shutdown" in r.getMessage()]
    connect_records = [
        r for r in caplog.records if "Companion client connected" in r.getMessage()
    ]
    assert len(connect_records) == 1  # only the original client ever attached

    reader_0.close()
    await _wait_until(lambda: handler_0.done())
    await asyncio.gather(handler_0)


@pytest.mark.asyncio
async def test_a_restarted_server_accepts_clients_again():
    """start()/stop()/start() on one instance: the second start serves clients.

    The closing flag is lifecycle state, not a tombstone: ``start()`` clears
    what ``stop()`` set, so stopping and starting the same server again must
    not refuse every client with "refused during shutdown".
    """
    server = _queued_server()
    server.companion_hash = "ab"  # start() logs the hash as decimal or hex

    await server.start()
    port = server._server.sockets[0].getsockname()[1]
    reader_a, writer_a = await asyncio.open_connection("127.0.0.1", port)
    await _wait_until(lambda: server._client_writer is not None)
    assert server._closing is False

    await server.stop()
    assert server._closing is True
    writer_a.close()
    await writer_a.wait_closed()

    await server.start()
    port = server._server.sockets[0].getsockname()[1]
    assert server._closing is False
    reader_b, writer_b = await asyncio.open_connection("127.0.0.1", port)
    await _wait_until(lambda: server._client_writer is not None)
    assert server._closing is False

    writer_b.close()
    await writer_b.wait_closed()
    await server.stop()
    assert server._closing is True
