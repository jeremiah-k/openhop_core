"""A frame must never be admitted into a dead or closing transport.

Behavioural pin for the companion queue: a sync that runs while the client's
socket is already dead — but whose disconnect the server has not processed
yet — must not claim messages for that client. Whatever the sync admits into
the dying client's write queue is dropped when the disconnect is finally
processed, so anything retired on its behalf is lost. The transport must
refuse admission into a closed or closing writer, and a successor client
must receive what the dying one could not take.
"""

from __future__ import annotations

import asyncio
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
    """Minimal writer double: closing mirrors a dead socket on demand."""

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


async def _wait_until(condition, spins: int = 2000) -> None:
    for _ in range(spins):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached within the spin budget")


def _server_with_queue() -> tuple[CompanionFrameServer, MessageQueue, QueuedMessage]:
    queue = MessageQueue(max_size=4)
    entry = QueuedMessage(sender_key=b"k" * 32, text="queued text")
    queue.push(entry)
    bridge = Mock()
    bridge.get_time = Mock(return_value=0)
    bridge.sync_next_message = queue.pop
    bridge.message_queue = queue
    server = CompanionFrameServer(bridge, "hash", port=0, client_idle_timeout_sec=None)
    return server, queue, entry


def _track_sync_dispatches(server: CompanionFrameServer) -> list:
    """Record completed CMD_SYNC_NEXT_MESSAGE dispatches on the server.

    A frame counts as handled once its command handler returned, so waiting on
    the returned list proves the frame was dispatched to the transport and the
    admission path saw it — including the refusal a dead writer must produce.
    """
    original = server._cmd_handlers[CMD_SYNC_NEXT_MESSAGE]
    dispatches: list = []

    async def tracked(data):
        try:
            await original(data)
        finally:
            dispatches.append(True)

    server._cmd_handlers[CMD_SYNC_NEXT_MESSAGE] = tracked
    return dispatches


@pytest.mark.asyncio
async def test_sync_refuses_a_dead_transport_and_the_successor_gets_the_message():
    """The mid-drain kill shape: socket dead, disconnect unprocessed, sync runs.

    The client's writer is already closing while the read loop still exists,
    exactly like a process killed mid-drain whose reset the server has not
    handled yet. The sync must not admit the message into that dying client,
    and a successor client must still receive it.
    """
    server, queue, entry = _server_with_queue()
    sync_dispatches = _track_sync_dispatches(server)

    reader_a, writer_a = _ScriptedReader(), _RecordingWriter()
    handler_a = asyncio.create_task(server._handle_client(reader_a, writer_a))
    await _wait_until(lambda: server._client_writer is writer_a)

    # The socket dies (SIGKILL): the writer closes before any disconnect is
    # processed. The read loop is still attached and idle, so the server's
    # per-client state still names this client.
    writer_a.closed = True
    assert server._client_writer is writer_a and server._write_queue is not None

    # A queued GET_MSG that was already in flight is dispatched into the
    # dead transport; wait until its handler has actually run.
    reader_a.feed(_command_frame(bytes([CMD_SYNC_NEXT_MESSAGE])))
    await _wait_until(lambda: len(sync_dispatches) == 1)

    # Nothing may be claimed for the dying client.
    assert queue.peek() is entry and queue.count == 1
    assert not getattr(entry, "frame_enqueued", False)

    # The disconnect is processed, then a successor client connects and asks.
    reader_a.close()
    await _wait_until(lambda: server._client_writer is not writer_a)

    reader_b, writer_b = _ScriptedReader(), _RecordingWriter()
    handler_b = asyncio.create_task(server._handle_client(reader_b, writer_b))
    await _wait_until(lambda: server._client_writer is writer_b)
    reader_b.feed(_command_frame(bytes([CMD_SYNC_NEXT_MESSAGE])))
    await _wait_until(lambda: queue.count == 0)
    await _wait_until(lambda: writer_b.written)
    assert b"queued text" in b"".join(writer_b.written)

    reader_b.close()
    await _wait_until(lambda: handler_a.done() and handler_b.done())
    await asyncio.gather(handler_a, handler_b)


@pytest.mark.asyncio
async def test_sync_does_not_stream_into_a_closing_transport_between_frames():
    """A transport that starts closing mid-drain ends the drain, not the queue.

    After one admitted frame the writer starts closing; further syncs must
    refuse to admit, leaving the remaining messages for a successor client.
    """
    server, queue, _first = _server_with_queue()
    sync_dispatches = _track_sync_dispatches(server)
    second = QueuedMessage(sender_key=b"j" * 32, text="second text")
    queue.push(second)

    reader_a, writer_a = _ScriptedReader(), _RecordingWriter()
    handler_a = asyncio.create_task(server._handle_client(reader_a, writer_a))
    await _wait_until(lambda: server._client_writer is writer_a)

    # The first sync drains one message and admits it into the live transport.
    reader_a.feed(_command_frame(bytes([CMD_SYNC_NEXT_MESSAGE])))
    await _wait_until(lambda: len(sync_dispatches) == 1)
    assert queue.count == 1

    # The socket dies between two fetches; the client had already queued one
    # more GET_MSG before it died. Wait until that sync's handler has run.
    writer_a.closed = True
    reader_a.feed(_command_frame(bytes([CMD_SYNC_NEXT_MESSAGE])))
    await _wait_until(lambda: len(sync_dispatches) == 2)

    assert queue.count == 1
    assert queue.peek() is second

    reader_a.close()
    await _wait_until(lambda: server._client_writer is not writer_a)
    reader_b, writer_b = _ScriptedReader(), _RecordingWriter()
    handler_b = asyncio.create_task(server._handle_client(reader_b, writer_b))
    await _wait_until(lambda: server._client_writer is writer_b)
    reader_b.feed(_command_frame(bytes([CMD_SYNC_NEXT_MESSAGE])))
    await _wait_until(lambda: queue.count == 0)
    await _wait_until(lambda: writer_b.written)
    assert b"second text" in b"".join(writer_b.written)

    reader_b.close()
    await _wait_until(lambda: handler_a.done() and handler_b.done())
    await asyncio.gather(handler_a, handler_b)
