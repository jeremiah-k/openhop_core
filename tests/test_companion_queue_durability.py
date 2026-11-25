"""Durability of the companion outbound queue.

The companion protocol caps one frame payload at ``MAX_PAYLOAD_SIZE`` bytes
(firmware ``writeFrame()`` refuses anything larger).  Two properties are
pinned here because the message queue depends on both:

* ``_enqueue_frame`` reports whether the frame actually reached the write
  queue.  "No client attached", "payload too large" and "queue full" are
  *failures* that the caller has to see - a silent drop loses the message.
* ``_build_message_frame`` shortens the message text so the *complete* frame
  fits the cap.  An over-long message is truncated, never dropped, and the
  truncation must not split a multi-byte UTF-8 character.
"""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import Mock

import pytest

from openhop_core.companion.constants import (
    FRAME_OUTBOUND_PREFIX,
    MAX_PAYLOAD_SIZE,
    RESP_CODE_CHANNEL_MSG_RECV,
    RESP_CODE_CHANNEL_MSG_RECV_V3,
    RESP_CODE_CONTACT_MSG_RECV,
    RESP_CODE_CONTACT_MSG_RECV_V3,
)
from openhop_core.companion.frame_server import CompanionFrameServer
from openhop_core.companion.models import QueuedMessage
from openhop_core.protocol.constants import TXT_TYPE_PLAIN, TXT_TYPE_SIGNED_PLAIN


def _server(app_target_ver: int = 3) -> CompanionFrameServer:
    server = CompanionFrameServer(Mock(), "hash", port=0)
    server._app_target_ver = app_target_ver
    return server


def _direct(text: str, *, txt_type: int = TXT_TYPE_PLAIN, sender_prefix: bytes = b""):
    return QueuedMessage(
        sender_key=bytes([0xAA] * 32),
        txt_type=txt_type,
        timestamp=1_650_000_000,
        text=text,
        path_len=3,
        snr=2.0,
        sender_prefix=sender_prefix,
    )


def _channel(text: str):
    return QueuedMessage(
        sender_key=b"",
        txt_type=TXT_TYPE_PLAIN,
        timestamp=1_650_000_000,
        text=text,
        is_channel=True,
        channel_idx=2,
        path_len=5,
        snr=-3.75,
    )


# ---------------------------------------------------------------------------
# _enqueue_frame must report success/failure
# ---------------------------------------------------------------------------


def test_enqueue_frame_returns_true_when_appended():
    server = _server()
    server._write_queue = asyncio.Queue(maxsize=8)
    server._client_writer = _OpenWriter()
    assert server._enqueue_frame(b"\x01\x02\x03") is True
    raw = server._write_queue.get_nowait()
    assert raw == bytes([FRAME_OUTBOUND_PREFIX]) + struct.pack("<H", 3) + b"\x01\x02\x03"


def test_enqueue_frame_returns_false_without_client():
    server = _server()
    server._write_queue = None
    assert server._enqueue_frame(b"\x01") is False


def test_enqueue_frame_returns_false_on_oversize_payload():
    server = _server()
    server._write_queue = asyncio.Queue(maxsize=8)
    server._client_writer = _OpenWriter()
    assert server._enqueue_frame(bytes(MAX_PAYLOAD_SIZE + 1)) is False
    assert server._write_queue.empty()


def test_enqueue_frame_returns_false_when_queue_full():
    server = _server()
    server._write_queue = asyncio.Queue(maxsize=1)
    server._client_writer = _OpenWriter()
    assert server._enqueue_frame(b"\x01") is True
    assert server._enqueue_frame(b"\x02") is False
    assert server._write_queue.qsize() == 1


def test_write_frame_returns_enqueue_result():
    server = _server()
    server._write_queue = asyncio.Queue(maxsize=1)
    server._client_writer = _OpenWriter()
    assert server._write_frame(b"\x01") is True
    assert server._write_frame(b"\x02") is False


# ---------------------------------------------------------------------------
# _build_message_frame must fit the payload cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "app_target_ver",
    [1, 3],
    ids=["v1", "v3"],
)
@pytest.mark.parametrize(
    "message_factory",
    [_direct, _channel],
    ids=["direct", "channel"],
)
def test_build_message_frame_truncates_oversize_text(message_factory, app_target_ver):
    server = _server(app_target_ver)
    frame = server._build_message_frame(message_factory("A" * 1000))
    assert len(frame) <= MAX_PAYLOAD_SIZE
    # The frame is still a well-formed message frame, not a truncated blob.
    assert frame[0] in (
        RESP_CODE_CONTACT_MSG_RECV,
        RESP_CODE_CONTACT_MSG_RECV_V3,
        RESP_CODE_CHANNEL_MSG_RECV,
        RESP_CODE_CHANNEL_MSG_RECV_V3,
    )


@pytest.mark.parametrize(
    "app_target_ver",
    [1, 3],
    ids=["v1", "v3"],
)
def test_build_message_frame_truncation_is_utf8_safe(app_target_ver):
    """A multi-byte character is never split by the truncation."""
    server = _server(app_target_ver)
    # "€" is 3 bytes in UTF-8, so a naive byte cut would emit a broken char.
    text = "€" * 200
    frame = server._build_message_frame(_direct(text))
    assert len(frame) <= MAX_PAYLOAD_SIZE
    # Only the text tail is UTF-8; the header is binary.
    header_len = 13 if app_target_ver < 3 else 16
    decoded = frame[header_len:].decode("utf-8")  # must not raise
    assert "�" not in decoded
    assert decoded == "€" * ((len(frame) - header_len) // 3)


def test_build_message_frame_signed_plain_keeps_author_prefix_when_truncated():
    """The signed-author bytes are part of the header and must survive."""
    server = _server(3)
    prefix = b"\xaa\xbb\xcc\xdd"
    frame = server._build_message_frame(
        _direct("B" * 1000, txt_type=TXT_TYPE_SIGNED_PLAIN, sender_prefix=prefix)
    )
    assert len(frame) <= MAX_PAYLOAD_SIZE
    assert frame[0] == RESP_CODE_CONTACT_MSG_RECV_V3
    # header: code(1) snr(1) 0(1) 0(1) prefix(6) path_len(1) txt_type(1) ts(4) author(4)
    assert frame[16:20] == prefix


def test_build_message_frame_normal_size_message_is_unchanged():
    """A message that already fits is emitted byte-for-byte as before."""
    server = _server(3)
    msg = _direct("hello world")
    frame = server._build_message_frame(msg)
    assert len(frame) == 4 + 6 + 2 + 4 + len(b"hello world")
    assert frame.endswith(b"hello world")


def test_build_message_frame_channel_normal_size_message_is_unchanged():
    server = _server(1)
    msg = _channel("hi")
    frame = server._build_message_frame(msg)
    assert len(frame) == 4 + 4 + len(b"hi")
    assert frame.endswith(b"hi")


def test_build_message_frame_oversize_signed_frame_still_fits_with_author():
    server = _server(1)
    frame = server._build_message_frame(
        _direct("C" * 500, txt_type=TXT_TYPE_SIGNED_PLAIN, sender_prefix=b"\x01\x02\x03\x04")
    )
    assert len(frame) <= MAX_PAYLOAD_SIZE
    # V1 header: code(1) prefix(6) path_len(1) txt_type(1) ts(4) author(4)
    assert frame[13:17] == b"\x01\x02\x03\x04"


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
