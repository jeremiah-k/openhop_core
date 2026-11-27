"""The offline queue numbers admissions in arrival order.

``MessageQueue.push`` stamps every admitted entry with one monotonically
increasing arrival number at admission time — the number order is the
arrival order regardless of when a persistence layer stores the message —
and rejected messages consume no number. The events published for a queued
message carry the number; events that did not come from the queue carry
``None``.
"""

from __future__ import annotations

from openhop_core.companion.message_queue import MessageQueue
from openhop_core.companion.models import (
    ChannelDataEvent,
    ChannelMessageEvent,
    MessageEvent,
    QueuedMessage,
)


def _msg(text: str) -> QueuedMessage:
    return QueuedMessage(sender_key=b"k" * 32, text=text)


def test_push_stamps_monotonic_arrival_numbers():
    queue = MessageQueue(max_size=4)
    first, second, third = _msg("one"), _msg("two"), _msg("three")
    assert queue.push(first) and queue.push(second) and queue.push(third)
    assert first.arrival_seq == 1
    assert second.arrival_seq == 2
    assert third.arrival_seq == 3


def test_rejected_push_consumes_no_number():
    queue = MessageQueue(max_size=1)
    first = _msg("kept")
    assert queue.push(first)
    rejected = _msg("direct cannot displace direct")  # not a channel message
    assert queue.push(rejected) is False
    assert rejected.arrival_seq is None
    next_entry = _msg("next")
    assert queue.push(next_entry) is False  # still full, single direct slot
    assert next_entry.arrival_seq is None


def test_zero_capacity_queue_never_stamps():
    queue = MessageQueue(max_size=0)
    entry = _msg("nowhere")
    assert queue.push(entry) is False
    assert entry.arrival_seq is None


def test_restore_front_keeps_the_original_number():
    queue = MessageQueue(max_size=4)
    first, second = _msg("one"), _msg("two")
    queue.push(first)
    queue.push(second)
    queue.pop()
    queue.restore_front(first)
    assert queue.peek() is first
    assert first.arrival_seq == 1  # unchanged by the restore


def test_seed_arrival_counter_raises_the_floor_and_never_lowers():
    queue = MessageQueue(max_size=4)
    queue.seed_arrival_counter(41)
    entry = _msg("after backlog")
    queue.push(entry)
    assert entry.arrival_seq == 42
    queue.seed_arrival_counter(7)  # never lowers
    later = _msg("later")
    queue.push(later)
    assert later.arrival_seq == 43


def test_events_carry_the_admission_number_or_none():
    event = MessageEvent(
        sender_key=b"k" * 32,
        text="queued",
        timestamp=1,
        txt_type=0,
        arrival_seq=1,
    )
    assert event.arrival_seq == 1
    assert (
        MessageEvent(sender_key=b"k" * 32, text="t", timestamp=1, txt_type=0).arrival_seq
        is None
    )
    assert (
        ChannelMessageEvent(channel_name="c", sender_name="s", text="t", timestamp=1).arrival_seq
        is None
    )
    assert (
        ChannelDataEvent(channel_idx=0, path_len=0, data_type=1, payload=b"").arrival_seq is None
    )
