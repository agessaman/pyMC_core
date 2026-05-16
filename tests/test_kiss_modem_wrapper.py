"""
Tests for MeshCore KISS Modem Wrapper

Tests the KISS frame encoding/decoding, command/response handling,
and LoRaRadio interface implementation.
"""

import struct
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pymc_core.hardware.kiss_modem_wrapper import (
    CMD_DATA,
    CMD_GET_BATTERY,
    CMD_GET_NOISE_FLOOR,
    CMD_GET_RADIO,
    CMD_GET_STATS,
    CMD_GET_VERSION,
    CMD_PING,
    CMD_SET_RADIO,
    CMD_SET_TX_POWER,
    CMD_SIGN_DATA,
    HW_CMD_GET_DEVICE_NAME,
    HW_CMD_GET_MCU_TEMP,
    HW_CMD_GET_SIGNAL_REPORT,
    HW_CMD_GET_VERSION,
    HW_CMD_REBOOT,
    HW_CMD_SET_SIGNAL_REPORT,
    HW_RESP_DEVICE_NAME,
    HW_RESP_MCU_TEMP,
    HW_RESP_OK,
    HW_RESP_RX_META,
    HW_RESP_SIGNAL_REPORT,
    KISS_CMD_FULLDUPLEX,
    KISS_CMD_PERSISTENCE,
    KISS_CMD_SETHARDWARE,
    KISS_CMD_SLOTTIME,
    KISS_CMD_TXTAIL,
    KISS_FEND,
    KISS_FESC,
    KISS_TFEND,
    KISS_TFESC,
    RESP_BATTERY,
    RESP_ERROR,
    RESP_IDENTITY,
    RESP_NOISE_FLOOR,
    RESP_OK,
    RESP_PONG,
    RESP_RADIO,
    RESP_SIGNATURE,
    RESP_STATS,
    RESP_TX_DONE,
    RESP_VERSION,
    KissModemWrapper,
)


class TestKissFrameEncoding:
    """Test KISS frame encoding/decoding"""

    def test_encode_simple_frame(self):
        """Test encoding a simple frame without special characters"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        frame = modem._encode_kiss_frame(CMD_DATA, b"\x01\x02\x03")

        # Should be: FEND + CMD + data + FEND
        assert frame[0] == KISS_FEND
        assert frame[1] == CMD_DATA
        assert frame[2:5] == b"\x01\x02\x03"
        assert frame[5] == KISS_FEND

    def test_encode_frame_with_fend_escape(self):
        """Test encoding a frame containing FEND byte"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        frame = modem._encode_kiss_frame(CMD_DATA, bytes([0xC0]))  # FEND

        # FEND in data should be escaped as FESC + TFEND
        assert frame[0] == KISS_FEND
        assert frame[1] == CMD_DATA
        assert frame[2] == KISS_FESC
        assert frame[3] == KISS_TFEND
        assert frame[4] == KISS_FEND

    def test_encode_frame_with_fesc_escape(self):
        """Test encoding a frame containing FESC byte"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        frame = modem._encode_kiss_frame(CMD_DATA, bytes([0xDB]))  # FESC

        # FESC in data should be escaped as FESC + TFESC
        assert frame[0] == KISS_FEND
        assert frame[1] == CMD_DATA
        assert frame[2] == KISS_FESC
        assert frame[3] == KISS_TFESC
        assert frame[4] == KISS_FEND

    def test_encode_frame_with_multiple_escapes(self):
        """Test encoding a frame with multiple special characters"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        frame = modem._encode_kiss_frame(CMD_DATA, bytes([0xC0, 0xDB, 0xC0]))

        expected = bytes(
            [
                KISS_FEND,
                CMD_DATA,
                KISS_FESC,
                KISS_TFEND,  # escaped 0xC0
                KISS_FESC,
                KISS_TFESC,  # escaped 0xDB
                KISS_FESC,
                KISS_TFEND,  # escaped 0xC0
                KISS_FEND,
            ]
        )
        assert frame == expected

    def test_decode_simple_frame(self):
        """Test decoding Data frame then RxMeta (spec: data and metadata separate)"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        received_frames = []
        modem.on_frame_received = lambda data: received_frames.append(data)

        # Data frame: FEND + 0x00 + raw_packet + FEND (no in-frame metadata)
        data_frame = bytes([KISS_FEND, CMD_DATA, 0x01, 0x02, 0x03, KISS_FEND])
        # RxMeta: FEND + 0x06 + 0xF9 + SNR + RSSI + FEND (sent immediately after Data)
        rx_meta_frame = bytes(
            [KISS_FEND, KISS_CMD_SETHARDWARE, HW_RESP_RX_META, 0x10, 0xB0, KISS_FEND]
        )

        for byte in data_frame:
            modem._decode_kiss_byte(byte)
        for byte in rx_meta_frame:
            modem._decode_kiss_byte(byte)

        assert len(received_frames) == 1
        assert received_frames[0] == b"\x01\x02\x03"

    def test_decode_frame_with_escapes(self):
        """Test decoding Data frame with escaped FEND, then RxMeta"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        received_frames = []
        modem.on_frame_received = lambda data: received_frames.append(data)

        # Data frame: payload is escaped 0xC0 (FESC + TFEND)
        data_frame = bytes([KISS_FEND, CMD_DATA, KISS_FESC, KISS_TFEND, KISS_FEND])
        rx_meta_frame = bytes(
            [KISS_FEND, KISS_CMD_SETHARDWARE, HW_RESP_RX_META, 0x10, 0xB0, KISS_FEND]
        )

        for byte in data_frame:
            modem._decode_kiss_byte(byte)
        for byte in rx_meta_frame:
            modem._decode_kiss_byte(byte)

        assert len(received_frames) == 1
        assert received_frames[0] == bytes([0xC0])

    def test_decode_extracts_rssi_snr(self):
        """Test that RSSI and SNR are extracted from RxMeta frame"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        data_frame = bytes([KISS_FEND, CMD_DATA, 0xAA, 0xBB, KISS_FEND])
        # RxMeta: SNR=0x10 (4.0 dB), RSSI=0xB0 (-80)
        rx_meta_frame = bytes(
            [KISS_FEND, KISS_CMD_SETHARDWARE, HW_RESP_RX_META, 0x10, 0xB0, KISS_FEND]
        )

        for byte in data_frame:
            modem._decode_kiss_byte(byte)
        for byte in rx_meta_frame:
            modem._decode_kiss_byte(byte)

        assert modem.stats["last_snr"] == pytest.approx(4.0)
        assert modem.stats["last_rssi"] == -80

    def test_rx_callback_receives_per_packet_rssi_snr(self):
        """Test that a 3-arg callback receives (data, rssi, snr) per Data+RxMeta pair"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        received = []

        def capture(data, rssi, snr):
            received.append((data, rssi, snr))

        modem.on_frame_received = capture

        # First packet: Data then RxMeta (SNR=4.0 dB, RSSI=-80)
        data1 = bytes([KISS_FEND, CMD_DATA, 0x01, 0x02, KISS_FEND])
        meta1 = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, HW_RESP_RX_META, 0x10, 0xB0, KISS_FEND])
        for byte in data1:
            modem._decode_kiss_byte(byte)
        for byte in meta1:
            modem._decode_kiss_byte(byte)

        # Second packet: Data then RxMeta (SNR=2.0 dB, RSSI=-100)
        data2 = bytes([KISS_FEND, CMD_DATA, 0x03, 0x04, KISS_FEND])
        meta2 = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, HW_RESP_RX_META, 0x08, 0x9C, KISS_FEND])
        for byte in data2:
            modem._decode_kiss_byte(byte)
        for byte in meta2:
            modem._decode_kiss_byte(byte)

        assert len(received) == 2
        assert received[0] == (b"\x01\x02", -80, 4.0)
        assert received[1] == (b"\x03\x04", -100, 2.0)

    def test_data_frame_without_rx_meta_does_not_call_callback(self):
        """Spec: Data frame queues payload; callback only on following RxMeta"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        received = []
        modem.on_frame_received = lambda data: received.append(data)

        # Only Data frame, no RxMeta
        data_frame = bytes([KISS_FEND, CMD_DATA, 0x01, 0x02, 0x03, KISS_FEND])
        for byte in data_frame:
            modem._decode_kiss_byte(byte)

        assert len(received) == 0
        assert len(modem._pending_rx_queue) == 1
        assert modem._pending_rx_queue[0] == b"\x01\x02\x03"

    def test_port_non_zero_discarded(self):
        """Frames with port != 0 are ignored (type byte 0x10 = port 1, cmd 0)"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        received = []
        modem.on_frame_received = lambda data: received.append(data)

        # Type 0x10: port=1, cmd=0 (Data on port 1) - should be discarded
        frame = bytes([KISS_FEND, 0x10, 0x01, 0x02, 0x03, KISS_FEND])
        for byte in frame:
            modem._decode_kiss_byte(byte)

        assert len(received) == 0
        assert len(modem._pending_rx_queue) == 0


class TestCommandResponses:
    """Test command sending and response parsing"""

    def test_send_command_encodes_correctly(self):
        """Test that _send_command sends SetHardware frame (FEND + 0x06 + sub_cmd + data + FEND)"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        mock_serial = MagicMock()
        mock_serial.is_open = True
        modem.serial_conn = mock_serial
        modem.is_connected = True

        modem._send_command(CMD_GET_VERSION, timeout=0.1)

        assert mock_serial.write.called
        written_frame = mock_serial.write.call_args[0][0]

        assert written_frame[0] == KISS_FEND
        assert written_frame[1] == KISS_CMD_SETHARDWARE  # type SetHardware
        assert written_frame[2] == HW_CMD_GET_VERSION  # sub_cmd GetVersion
        assert written_frame[-1] == KISS_FEND

    def test_response_parsing_identity(self):
        """Test parsing SetHardware Identity response (FEND + 0x06 + 0x21 + pubkey + FEND)"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        pubkey = bytes(range(32))
        raw_bytes = (
            bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_IDENTITY]) + pubkey + bytes([KISS_FEND])
        )

        for byte in raw_bytes:
            modem._decode_kiss_byte(byte)

        # Without an active waiter, SetHardware responses are queued for later consumption.
        assert len(modem._response_queue) == 1
        assert modem._response_queue[0][0] == RESP_IDENTITY
        assert modem._response_queue[0][1] == pubkey

    def test_response_parsing_error(self):
        """Test parsing SetHardware Error response (FEND + 0x06 + 0x2A + code + FEND)"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        raw_bytes = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_ERROR, 0x05, KISS_FEND])

        for byte in raw_bytes:
            modem._decode_kiss_byte(byte)

        assert len(modem._response_queue) == 1
        assert modem._response_queue[0][0] == RESP_ERROR
        assert modem._response_queue[0][1][0] == 0x05

    def test_send_command_uses_queued_late_response(self):
        """If a matching response is already queued, _send_command returns it without writing."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        # Queue a late PONG from a previous ping.
        modem._response_queue.append((RESP_PONG, b""))

        modem._write_frame = MagicMock(return_value=True)

        resp = modem._send_command(CMD_PING, timeout=0.1)
        assert resp == (RESP_PONG, b"")
        assert modem._write_frame.call_count == 0

    def test_send_command_correlates_expected_response(self):
        """Non-matching responses are queued; waiter completes only on expected sub_cmd."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        # Set up a serial conn so _send_command can write.
        mock_serial = MagicMock()
        mock_serial.is_open = True
        mock_serial.write.side_effect = lambda b: len(b)
        modem.serial_conn = mock_serial

        result_holder: dict[str, object] = {}

        def caller():
            result_holder["resp"] = modem._send_command(CMD_PING, timeout=0.5)

        t = threading.Thread(target=caller)
        t.start()

        # Feed an unrelated identity response first; should be queued, not delivered.
        pubkey = bytes(range(32))
        identity_bytes = (
            bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_IDENTITY]) + pubkey + bytes([KISS_FEND])
        )
        for b in identity_bytes:
            modem._decode_kiss_byte(b)

        # Now feed the expected PONG.
        pong_bytes = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_PONG, KISS_FEND])
        for b in pong_bytes:
            modem._decode_kiss_byte(b)

        t.join(timeout=1.0)
        assert result_holder.get("resp") == (RESP_PONG, b"")

        # The unrelated identity response should remain queued.
        assert len(modem._response_queue) == 1
        assert modem._response_queue[0][0] == RESP_IDENTITY

    def test_send_command_is_single_flight(self):
        """Concurrent _send_command calls must not interleave shared waiter state."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        wrote_first = threading.Event()
        allow_first_write = threading.Event()
        wrote_second = threading.Event()

        def mock_write_frame(frame: bytes) -> bool:
            # sub_cmd is the first payload byte (frame[2]) in SetHardware frames.
            if frame[2] == CMD_GET_VERSION:
                wrote_first.set()
                allow_first_write.wait(timeout=1.0)
            elif frame[2] == CMD_PING:
                wrote_second.set()
            return True

        modem._write_frame = mock_write_frame

        results: dict[str, object] = {}

        def call_version():
            results["v"] = modem._send_command(CMD_GET_VERSION, timeout=0.5)

        def call_ping():
            results["p"] = modem._send_command(CMD_PING, timeout=0.5)

        t1 = threading.Thread(target=call_version)
        t2 = threading.Thread(target=call_ping)

        t1.start()
        assert wrote_first.wait(timeout=1.0)

        # Start second call while first is still holding the command lock in _write_frame.
        t2.start()
        assert not wrote_second.wait(timeout=0.1)

        # Let the first command proceed and respond.
        allow_first_write.set()
        version_bytes = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_VERSION, 0x01, KISS_FEND])
        for b in version_bytes:
            modem._decode_kiss_byte(b)

        # Now second command can write and receive response.
        assert wrote_second.wait(timeout=1.0)
        pong_bytes = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_PONG, KISS_FEND])
        for b in pong_bytes:
            modem._decode_kiss_byte(b)

        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

        assert results.get("v") is not None
        assert results.get("p") == (RESP_PONG, b"")

    def test_send_command_timeout_clears_waiter_state(self):
        """Timeout path must clear active waiter metadata for later commands."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True
        modem._write_frame = MagicMock(return_value=True)

        resp = modem._send_command(CMD_GET_VERSION, timeout=0.05)
        assert resp is None
        assert modem._expected_response_subcmds is None
        assert modem._active_request_subcmd is None

        # Ensure no lock leak by issuing another command.
        resp2 = modem._send_command(CMD_PING, timeout=0.05)
        assert resp2 is None
        assert modem._expected_response_subcmds is None
        assert modem._active_request_subcmd is None

    def test_send_command_write_failure_clears_waiter_state(self):
        """Write-failure path must clear active waiter metadata."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True
        modem._write_frame = MagicMock(return_value=False)

        resp = modem._send_command(CMD_GET_VERSION, timeout=0.1)
        assert resp is None
        assert modem._expected_response_subcmds is None
        assert modem._active_request_subcmd is None

    def test_response_queue_drop_oldest_when_full(self):
        """Unmatched SetHardware responses should drop oldest when queue is full."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        maxlen = modem._response_queue.maxlen or 0
        for i in range(maxlen):
            modem._response_queue.append((0xA0 + (i % 10), bytes([i % 256])))
        oldest = modem._response_queue[0]

        # No active waiter; incoming response should be enqueued as unmatched.
        frame = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_IDENTITY, 0x42, KISS_FEND])
        for b in frame:
            modem._decode_kiss_byte(b)

        assert len(modem._response_queue) == maxlen
        assert modem._response_queue[0] != oldest
        assert modem._response_queue[-1] == (RESP_IDENTITY, b"\x42")

    def test_send_command_ok_policy_allowlisted_command(self):
        """Allowlisted SetHardware commands may resolve with HW_RESP_OK."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True
        mock_serial = MagicMock()
        mock_serial.is_open = True
        mock_serial.write.side_effect = lambda b: len(b)
        modem.serial_conn = mock_serial

        result_holder: dict[str, object] = {}

        def caller():
            result_holder["resp"] = modem._send_command(CMD_SET_TX_POWER, b"\x16", timeout=0.5)

        t = threading.Thread(target=caller)
        t.start()

        ok_frame = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, HW_RESP_OK, KISS_FEND])
        for b in ok_frame:
            modem._decode_kiss_byte(b)

        t.join(timeout=1.0)
        assert result_holder.get("resp") == (HW_RESP_OK, b"")

    def test_send_command_ok_policy_non_allowlisted_command(self):
        """Non-allowlisted commands should not complete on HW_RESP_OK."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True
        mock_serial = MagicMock()
        mock_serial.is_open = True
        mock_serial.write.side_effect = lambda b: len(b)
        modem.serial_conn = mock_serial

        result_holder: dict[str, object] = {}

        def caller():
            result_holder["resp"] = modem._send_command(CMD_PING, timeout=0.5)

        t = threading.Thread(target=caller)
        t.start()

        ok_frame = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, HW_RESP_OK, KISS_FEND])
        for b in ok_frame:
            modem._decode_kiss_byte(b)

        pong_frame = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_PONG, KISS_FEND])
        for b in pong_frame:
            modem._decode_kiss_byte(b)

        t.join(timeout=1.0)
        assert result_holder.get("resp") == (RESP_PONG, b"")
        assert len(modem._response_queue) == 1
        assert modem._response_queue[0] == (HW_RESP_OK, b"")

    def test_send_command_preserves_unrelated_response_order(self):
        """Multiple unrelated responses remain queued in arrival order."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True
        mock_serial = MagicMock()
        mock_serial.is_open = True
        mock_serial.write.side_effect = lambda b: len(b)
        modem.serial_conn = mock_serial

        result_holder: dict[str, object] = {}

        def caller():
            result_holder["resp"] = modem._send_command(CMD_PING, timeout=0.5)

        t = threading.Thread(target=caller)
        t.start()

        identity = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_IDENTITY, 0xAA, KISS_FEND])
        version = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_VERSION, 0x01, KISS_FEND])
        stats = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_STATS, 0x02, KISS_FEND])
        for frame in (identity, version, stats):
            for b in frame:
                modem._decode_kiss_byte(b)

        pong = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_PONG, KISS_FEND])
        for b in pong:
            modem._decode_kiss_byte(b)

        t.join(timeout=1.0)
        assert result_holder.get("resp") == (RESP_PONG, b"")
        assert [entry[0] for entry in modem._response_queue] == [
            RESP_IDENTITY,
            RESP_VERSION,
            RESP_STATS,
        ]
        assert [entry[1] for entry in modem._response_queue] == [b"\xAA", b"\x01", b"\x02"]

    def test_tx_done_response(self):
        """Test SetHardware TxDone (0xF8) response sets event"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True
        modem._tx_done_event = threading.Event()

        raw_bytes = bytes([KISS_FEND, KISS_CMD_SETHARDWARE, RESP_TX_DONE, 0x01, KISS_FEND])

        for byte in raw_bytes:
            modem._decode_kiss_byte(byte)

        assert modem._tx_done_event.is_set()
        assert modem._tx_done_result is True

    @pytest.mark.asyncio
    async def test_send_offloads_get_airtime_to_thread(self):
        """send() must not call blocking get_airtime on the asyncio event loop."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.send_frame = MagicMock(return_value=True)

        to_thread_mock = AsyncMock(return_value=42)
        with patch("pymc_core.hardware.kiss_modem_wrapper.asyncio.to_thread", to_thread_mock):
            result = await modem.send(b"payload")

        assert result is not None
        assert result["airtime_ms"] == 42
        to_thread_mock.assert_awaited_once_with(modem.get_airtime, len(b"payload"), 1.0)


class TestKissAsyncTelemetry:
    """Async-safe telemetry entrypoints delegate blocking work via asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_get_status_async_delegates_to_thread(self):
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        to_thread_mock = AsyncMock(return_value={"ok": True})
        with patch("pymc_core.hardware.kiss_modem_wrapper.asyncio.to_thread", to_thread_mock):
            result = await modem.get_status_async(1.25)
        assert result == {"ok": True}
        to_thread_mock.assert_awaited_once_with(modem._sync_get_status, 1.25)

    @pytest.mark.asyncio
    async def test_get_noise_floor_async_delegates_to_thread(self):
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        to_thread_mock = AsyncMock(return_value=-95)
        with patch("pymc_core.hardware.kiss_modem_wrapper.asyncio.to_thread", to_thread_mock):
            result = await modem.get_noise_floor_async(0.75)
        assert result == -95
        to_thread_mock.assert_awaited_once_with(modem.get_noise_floor, 0.75)

    @pytest.mark.asyncio
    async def test_get_modem_stats_async_delegates_to_thread(self):
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        stats = {"rx": 1, "tx": 2, "errors": 0}
        to_thread_mock = AsyncMock(return_value=stats)
        with patch("pymc_core.hardware.kiss_modem_wrapper.asyncio.to_thread", to_thread_mock):
            result = await modem.get_modem_stats_async(None)
        assert result == stats
        to_thread_mock.assert_awaited_once_with(modem.get_modem_stats, None)

    def test_get_noise_floor_forwards_timeout_to_send_command(self):
        """Optional timeout on sync getter must reach _send_command."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True
        calls: list[tuple] = []

        def mock_send_command(cmd, data=b"", timeout=5.0):
            calls.append((cmd, data, timeout))
            if cmd == CMD_GET_NOISE_FLOOR:
                return (RESP_NOISE_FLOOR, struct.pack("<h", -88))
            return None

        modem._send_command = mock_send_command
        assert modem.get_noise_floor(timeout=0.42) == -88
        assert len(calls) == 1
        assert calls[0][0] == CMD_GET_NOISE_FLOOR
        assert calls[0][2] == 0.42


class TestRadioConfiguration:
    """Test radio configuration encoding"""

    def test_radio_config_struct_format(self):
        """Test that radio config is packed correctly"""
        KissModemWrapper(port="/dev/null", auto_configure=False)

        freq_hz = 869618000
        bw_hz = 62500
        sf = 8
        cr = 8

        # This is what configure_radio should pack
        expected = struct.pack("<IIBB", freq_hz, bw_hz, sf, cr)

        assert len(expected) == 10
        # Verify unpacking
        unpacked = struct.unpack("<IIBB", expected)
        assert unpacked == (freq_hz, bw_hz, sf, cr)

    def test_configure_radio_sends_correct_commands(self):
        """Test that configure_radio sends SET_RADIO and SET_TX_POWER"""
        modem = KissModemWrapper(
            port="/dev/null",
            auto_configure=False,
            radio_config={
                "frequency": 869618000,
                "bandwidth": 62500,
                "spreading_factor": 8,
                "coding_rate": 8,
                "power": 22,
            },
        )

        # Track sent commands
        sent_commands = []

        def mock_send_command(cmd, data=b"", timeout=5.0):
            sent_commands.append((cmd, data))
            return (RESP_OK, b"")

        modem._send_command = mock_send_command
        modem.is_connected = True

        result = modem.configure_radio()

        assert result is True
        assert len(sent_commands) == 2

        # First command: SET_RADIO
        assert sent_commands[0][0] == CMD_SET_RADIO
        assert len(sent_commands[0][1]) == 10  # 4 + 4 + 1 + 1

        # Second command: SET_TX_POWER
        assert sent_commands[1][0] == CMD_SET_TX_POWER
        assert sent_commands[1][1] == bytes([22])


class TestCryptoOperations:
    """Test cryptographic operation methods"""

    def test_get_random_validates_length(self):
        """Test get_random validates length parameter"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        # Length too small
        assert modem.get_random(0) is None

        # Length too large
        assert modem.get_random(65) is None

    def test_sign_data_sends_correct_command(self):
        """Test sign_data sends correct command"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        signature = bytes(range(64))

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == CMD_SIGN_DATA:
                return (RESP_SIGNATURE, signature)
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        result = modem.sign_data(b"test data")
        assert result == signature

    def test_verify_signature_validates_lengths(self):
        """Test verify_signature validates input lengths"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        # Invalid pubkey length
        assert modem.verify_signature(b"short", bytes(64), b"data") is None

        # Invalid signature length
        assert modem.verify_signature(bytes(32), b"short", b"data") is None

    def test_encrypt_data_validates_key_length(self):
        """Test encrypt_data validates key length"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        # Invalid key length
        assert modem.encrypt_data(b"short_key", b"plaintext") is None

    def test_decrypt_data_validates_lengths(self):
        """Test decrypt_data validates input lengths"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        # Invalid key length
        assert modem.decrypt_data(b"short", bytes(2), b"ciphertext") is None

        # Invalid MAC length
        assert modem.decrypt_data(bytes(32), b"x", b"ciphertext") is None

    def test_key_exchange_validates_pubkey_length(self):
        """Test key_exchange validates pubkey length"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        assert modem.key_exchange(b"short_pubkey") is None


class TestLoRaRadioInterface:
    """Test LoRaRadio interface implementation"""

    def test_set_rx_callback(self):
        """Test setting RX callback"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        callback = MagicMock()
        modem.set_rx_callback(callback)

        assert modem.on_frame_received == callback

    def test_get_last_rssi(self):
        """Test get_last_rssi returns stats value"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.stats["last_rssi"] = -85

        assert modem.get_last_rssi() == -85

    def test_get_last_snr(self):
        """Test get_last_snr returns stats value"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.stats["last_snr"] = 7.5

        assert modem.get_last_snr() == 7.5

    def test_get_stats_returns_copy(self):
        """Test get_stats returns a copy of stats dict"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.stats["frames_sent"] = 100

        stats = modem.get_stats()
        stats["frames_sent"] = 999

        # Original should be unchanged
        assert modem.stats["frames_sent"] == 100


class TestSendFrame:
    """Test send_frame functionality"""

    def test_send_frame_validates_size(self):
        """Test send_frame validates packet size"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        # Too small (< 2 bytes)
        assert modem.send_frame(b"\x00") is False

        # Too large (> 255 bytes)
        assert modem.send_frame(bytes(256)) is False

    def test_send_frame_requires_connection(self):
        """Test send_frame requires connection"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = False

        assert modem.send_frame(b"\x00\x01") is False

    def test_send_frame_queues_to_buffer(self):
        """Test send_frame adds to TX buffer"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        assert len(modem.tx_buffer) == 0

        result = modem.send_frame(b"\x01\x02\x03")

        assert result is True
        assert len(modem.tx_buffer) == 1

        # Verify frame is properly encoded
        frame = modem.tx_buffer[0]
        assert frame[0] == KISS_FEND
        assert frame[1] == CMD_DATA
        assert frame[-1] == KISS_FEND


class TestSerialWriteSerialization:
    """Test UART write serialization across concurrent callers."""

    def test_write_frame_serializes_data_and_sethardware_callers(self):
        """Data TX and SetHardware writes must not interleave at the UART layer."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        class BlockingSerial:
            def __init__(self):
                self.is_open = True
                self._active = 0
                self.max_active = 0
                self._state_lock = threading.Lock()
                self.first_write_entered = threading.Event()
                self.release_first_write = threading.Event()
                self.first_seen = False
                self.writes = []
                self.flush_count = 0

            def write(self, data):
                with self._state_lock:
                    self._active += 1
                    self.max_active = max(self.max_active, self._active)

                if not self.first_seen:
                    self.first_seen = True
                    self.first_write_entered.set()
                    self.release_first_write.wait(timeout=1.0)

                self.writes.append(bytes(data))

                with self._state_lock:
                    self._active -= 1
                return len(data)

            def flush(self):
                self.flush_count += 1

        serial_conn = BlockingSerial()
        modem.serial_conn = serial_conn
        modem.is_connected = True

        data_frame = modem._encode_kiss_frame(CMD_DATA, b"\x01\x02\x03")
        sethw_frame = modem._encode_kiss_frame(KISS_CMD_SETHARDWARE, bytes([CMD_PING]))

        results: dict[str, bool] = {}
        second_started = threading.Event()
        second_done = threading.Event()

        def write_data():
            results["data"] = modem._write_frame(data_frame)

        def write_sethw():
            second_started.set()
            results["sethw"] = modem._write_frame(sethw_frame)
            second_done.set()

        t1 = threading.Thread(target=write_data)
        t1.start()
        assert serial_conn.first_write_entered.wait(timeout=1.0)

        t2 = threading.Thread(target=write_sethw)
        t2.start()
        assert second_started.wait(timeout=1.0)

        # While the first writer is blocked inside serial.write, a second caller
        # should not enter serial.write concurrently.
        assert not second_done.wait(timeout=0.05)
        assert serial_conn.max_active == 1

        serial_conn.release_first_write.set()

        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

        assert results.get("data") is True
        assert results.get("sethw") is True
        assert serial_conn.max_active == 1
        assert serial_conn.flush_count == 2
        assert serial_conn.writes == [data_frame, sethw_frame]

    def test_ping_and_noise_floor_under_concurrent_data_load(self):
        """Concurrent data TX should not cause ping/noise-floor command timeouts."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        class RespondingSerial:
            def __init__(self):
                self.is_open = True
                self._modem: KissModemWrapper | None = None
                self.flush_count = 0

            def set_modem(self, m: KissModemWrapper) -> None:
                self._modem = m

            def write(self, data):
                frame = bytes(data)
                if (
                    self._modem is not None
                    and len(frame) >= 4
                    and frame[0] == KISS_FEND
                    and frame[-1] == KISS_FEND
                    and frame[1] == KISS_CMD_SETHARDWARE
                ):
                    sub_cmd = frame[2]
                    if sub_cmd == CMD_PING:
                        response_sub = RESP_PONG
                        response_payload = b""
                    elif sub_cmd == CMD_GET_NOISE_FLOOR:
                        response_sub = RESP_NOISE_FLOOR
                        response_payload = struct.pack("<h", -95)
                    else:
                        response_sub = None
                        response_payload = b""

                    if response_sub is not None:

                        def emit() -> None:
                            resp = (
                                bytes([KISS_FEND, KISS_CMD_SETHARDWARE, response_sub])
                                + response_payload
                                + bytes([KISS_FEND])
                            )
                            for b in resp:
                                self._modem._decode_kiss_byte(b)

                        threading.Thread(target=emit, daemon=True).start()
                return len(frame)

            def flush(self):
                self.flush_count += 1

        serial_conn = RespondingSerial()
        serial_conn.set_modem(modem)
        modem.serial_conn = serial_conn

        stop_event = threading.Event()
        data_frame = modem._encode_kiss_frame(CMD_DATA, b"\xAA\xBB\xCC")

        def data_tx_worker() -> None:
            for _ in range(200):
                if stop_event.is_set():
                    return
                modem._write_frame(data_frame)

        tx_thread = threading.Thread(target=data_tx_worker)
        tx_thread.start()

        try:
            for _ in range(40):
                ping_resp = modem._send_command(CMD_PING, timeout=0.2)
                assert ping_resp is not None
                assert ping_resp[0] == RESP_PONG

                noise = modem.get_noise_floor(timeout=0.2)
                assert noise == -95
        finally:
            stop_event.set()
            tx_thread.join(timeout=1.0)

    def test_tx_worker_and_sethardware_queries_make_progress_together(self):
        """Queued data TX should still make progress while periodic SetHardware queries run."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        class RespondingSerial:
            def __init__(self):
                self.is_open = True
                self._modem: KissModemWrapper | None = None
                self.flush_count = 0
                self.data_writes = 0

            def set_modem(self, m: KissModemWrapper) -> None:
                self._modem = m

            def write(self, data):
                frame = bytes(data)
                if (
                    self._modem is not None
                    and len(frame) >= 4
                    and frame[0] == KISS_FEND
                    and frame[-1] == KISS_FEND
                    and frame[1] == KISS_CMD_SETHARDWARE
                ):
                    sub_cmd = frame[2]
                    if sub_cmd == CMD_PING:
                        response_sub = RESP_PONG
                        response_payload = b""
                    elif sub_cmd == CMD_GET_NOISE_FLOOR:
                        response_sub = RESP_NOISE_FLOOR
                        response_payload = struct.pack("<h", -92)
                    else:
                        response_sub = None
                        response_payload = b""

                    if response_sub is not None:

                        def emit() -> None:
                            resp = (
                                bytes([KISS_FEND, KISS_CMD_SETHARDWARE, response_sub])
                                + response_payload
                                + bytes([KISS_FEND])
                            )
                            for b in resp:
                                self._modem._decode_kiss_byte(b)

                        threading.Thread(target=emit, daemon=True).start()
                elif len(frame) >= 2 and frame[0] == KISS_FEND and frame[1] == CMD_DATA:
                    self.data_writes += 1

                return len(frame)

            def flush(self):
                self.flush_count += 1

        serial_conn = RespondingSerial()
        serial_conn.set_modem(modem)
        modem.serial_conn = serial_conn

        for _ in range(120):
            assert modem.send_frame(b"\x01\x02\x03")

        modem.stop_event.clear()
        tx_thread = threading.Thread(target=modem._tx_worker, daemon=True)
        tx_thread.start()

        try:
            for _ in range(25):
                ping_resp = modem._send_command(CMD_PING, timeout=0.3)
                assert ping_resp is not None
                assert ping_resp[0] == RESP_PONG

                noise = modem.get_noise_floor(timeout=0.3)
                assert noise == -92

            deadline = time.time() + 2.0
            while modem.tx_buffer and time.time() < deadline:
                time.sleep(0.01)
            assert len(modem.tx_buffer) == 0
            assert serial_conn.data_writes > 0
        finally:
            modem.stop_event.set()
            tx_thread.join(timeout=1.0)

    def test_write_error_does_not_poison_future_writes(self):
        """A serial write error should fail fast and transition to degraded mode."""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        class FlakySerial:
            def __init__(self):
                self.is_open = True
                self.calls = 0
                self.flush_count = 0

            def write(self, data):
                self.calls += 1
                if self.calls == 1:
                    raise OSError("simulated serial failure")
                return len(data)

            def flush(self):
                self.flush_count += 1

        serial_conn = FlakySerial()
        modem.serial_conn = serial_conn
        modem._start_reconnect_worker = MagicMock()

        frame = modem._encode_kiss_frame(CMD_DATA, b"\xAA\xBB")
        assert modem._write_frame(frame) is False
        assert modem._degraded is True
        assert modem.is_connected is False
        modem._start_reconnect_worker.assert_called_once()


class TestQueryMethods:
    """Test modem query methods"""

    def test_get_radio_config_parses_response(self):
        """Test get_radio_config parses response correctly"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        freq = 869618000
        bw = 62500
        sf = 8
        cr = 8
        response_data = struct.pack("<IIBB", freq, bw, sf, cr)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == CMD_GET_RADIO:
                return (RESP_RADIO, response_data)
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        config = modem.get_radio_config()

        assert config["frequency"] == freq
        assert config["bandwidth"] == bw
        assert config["spreading_factor"] == sf
        assert config["coding_rate"] == cr

    def test_get_modem_stats_parses_response(self):
        """Test get_modem_stats parses response correctly"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        rx = 100
        tx = 50
        errors = 5
        response_data = struct.pack("<III", rx, tx, errors)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == CMD_GET_STATS:
                return (RESP_STATS, response_data)
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        stats = modem.get_modem_stats()

        assert stats["rx"] == rx
        assert stats["tx"] == tx
        assert stats["errors"] == errors

    def test_get_battery_parses_response(self):
        """Test get_battery parses millivolt response"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        millivolts = 3700
        response_data = struct.pack("<H", millivolts)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == CMD_GET_BATTERY:
                return (RESP_BATTERY, response_data)
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        result = modem.get_battery()
        assert result == millivolts

    def test_ping_returns_true_on_pong(self):
        """Test ping returns True when modem responds with PONG"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == CMD_PING:
                return (RESP_PONG, b"")
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        assert modem.ping() is True

    def test_ping_returns_false_on_timeout(self):
        """Test ping returns False on timeout"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            return None  # Simulate timeout

        modem._send_command = mock_send_command
        modem.is_connected = True

        assert modem.ping() is False

    def test_get_mcu_temp_parses_response(self):
        """Test get_mcu_temp parses signed int16 tenths of °C"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        # 253 tenths = 25.3 °C
        response_data = struct.pack("<h", 253)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == HW_CMD_GET_MCU_TEMP:
                return (HW_RESP_MCU_TEMP, response_data)
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        assert modem.get_mcu_temp() == pytest.approx(25.3)

    def test_get_mcu_temp_returns_none_on_no_callback_error(self):
        """Test get_mcu_temp returns None when modem returns NoCallback error"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == HW_CMD_GET_MCU_TEMP:
                return (RESP_ERROR, bytes([0x03]))  # HW_ERR_NO_CALLBACK
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        assert modem.get_mcu_temp() is None

    def test_get_device_name_parses_utf8(self):
        """Test get_device_name returns UTF-8 decoded string"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        name = "TestDevice"
        response_data = name.encode("utf-8")

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == HW_CMD_GET_DEVICE_NAME:
                return (HW_RESP_DEVICE_NAME, response_data)
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        assert modem.get_device_name() == "TestDevice"

    def test_reboot_sends_command(self):
        """Test reboot sends HW_CMD_REBOOT SetHardware command"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        sent = []

        def mock_send_command(cmd, data=b"", timeout=5.0):
            sent.append((cmd, data))
            return (HW_RESP_OK, b"")

        modem._send_command = mock_send_command
        modem.is_connected = True

        modem.reboot()

        assert len(sent) == 1
        assert sent[0][0] == HW_CMD_REBOOT
        assert sent[0][1] == b""


class TestEventLoop:
    """Test event loop integration for thread-safe async"""

    def test_set_event_loop(self):
        """Test setting event loop"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        loop = MagicMock()

        modem.set_event_loop(loop)

        assert modem._event_loop is loop

    def test_dispatch_uses_event_loop_when_set(self):
        """Test that dispatch uses call_soon_threadsafe when loop is set"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        loop = MagicMock()
        modem.set_event_loop(loop)

        callback = MagicMock()
        modem.on_frame_received = callback

        modem._dispatch_rx_callback(b"test", -80, 4.0)

        # Should have called call_soon_threadsafe
        loop.call_soon_threadsafe.assert_called_once()

    def test_dispatch_direct_when_no_event_loop(self):
        """Test that dispatch invokes callback directly when no loop set"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        received = []

        def callback(data, rssi, snr):
            received.append((data, rssi, snr))

        modem.on_frame_received = callback

        modem._dispatch_rx_callback(b"test", -80, 4.0)

        assert len(received) == 1
        assert received[0] == (b"test", -80, 4.0)


class TestRadioConfigCompatibility:
    """Test radio config key compatibility"""

    def test_power_key(self):
        """Test that 'power' key is used"""
        modem = KissModemWrapper(
            port="/dev/null",
            auto_configure=False,
            radio_config={"power": 15},
        )

        sent_commands = []

        def mock_send_command(cmd, data=b"", timeout=5.0):
            sent_commands.append((cmd, data))
            return (RESP_OK, b"")

        modem._send_command = mock_send_command
        modem.is_connected = True

        modem.configure_radio()

        # Find SET_TX_POWER command
        tx_power_cmd = next((c for c in sent_commands if c[0] == CMD_SET_TX_POWER), None)
        assert tx_power_cmd is not None
        assert tx_power_cmd[1] == bytes([15])

    def test_tx_power_key_fallback(self):
        """Test that 'tx_power' key is used when 'power' is not present"""
        modem = KissModemWrapper(
            port="/dev/null",
            auto_configure=False,
            radio_config={"tx_power": 20},
        )

        sent_commands = []

        def mock_send_command(cmd, data=b"", timeout=5.0):
            sent_commands.append((cmd, data))
            return (RESP_OK, b"")

        modem._send_command = mock_send_command
        modem.is_connected = True

        modem.configure_radio()

        # Find SET_TX_POWER command
        tx_power_cmd = next((c for c in sent_commands if c[0] == CMD_SET_TX_POWER), None)
        assert tx_power_cmd is not None
        assert tx_power_cmd[1] == bytes([20])

    def test_power_takes_precedence_over_tx_power(self):
        """Test that 'power' takes precedence over 'tx_power'"""
        modem = KissModemWrapper(
            port="/dev/null",
            auto_configure=False,
            radio_config={"power": 10, "tx_power": 20},
        )

        sent_commands = []

        def mock_send_command(cmd, data=b"", timeout=5.0):
            sent_commands.append((cmd, data))
            return (RESP_OK, b"")

        modem._send_command = mock_send_command
        modem.is_connected = True

        modem.configure_radio()

        # Find SET_TX_POWER command - should use 'power' value
        tx_power_cmd = next((c for c in sent_commands if c[0] == CMD_SET_TX_POWER), None)
        assert tx_power_cmd is not None
        assert tx_power_cmd[1] == bytes([10])


class TestKissTuningMethods:
    """Test KISS config commands: persistence, slottime, txtail, full_duplex, signal report"""

    def test_set_kiss_persistence_sends_correct_frame(self):
        """Test set_kiss_persistence sends KISS_CMD_PERSISTENCE with value 0-255"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        written = []

        def capture_write(frame):
            written.append(bytes(frame))
            return True

        modem._write_frame = capture_write
        modem.is_connected = True

        result = modem.set_kiss_persistence(63)
        assert result is True
        assert len(written) == 1
        # FEND + 0x02 + 0x3F + FEND
        assert written[0][0] == KISS_FEND
        assert written[0][1] == KISS_CMD_PERSISTENCE
        assert written[0][2] == 63
        assert written[0][3] == KISS_FEND

    def test_set_kiss_persistence_clamps_value(self):
        """Test set_kiss_persistence clamps to 0-255"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        written = []

        def capture_write(frame):
            written.append(bytes(frame))
            return True

        modem._write_frame = capture_write
        modem.is_connected = True

        modem.set_kiss_persistence(300)
        assert written[0][2] == 255
        written.clear()
        modem.set_kiss_persistence(-1)
        assert written[0][2] == 0

    def test_set_kiss_slottime_sends_correct_frame(self):
        """Test set_kiss_slottime sends KISS_CMD_SLOTTIME with value in 10ms units"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        written = []

        def capture_write(frame):
            written.append(bytes(frame))
            return True

        modem._write_frame = capture_write
        modem.is_connected = True

        result = modem.set_kiss_slottime(100)
        assert result is True
        assert len(written) == 1
        assert written[0][0] == KISS_FEND
        assert written[0][1] == KISS_CMD_SLOTTIME
        assert written[0][2] == 10  # 100ms / 10
        assert written[0][3] == KISS_FEND

    def test_set_kiss_txtail_sends_correct_frame(self):
        """Test set_kiss_txtail sends KISS_CMD_TXTAIL with value in 10ms units"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        written = []

        def capture_write(frame):
            written.append(bytes(frame))
            return True

        modem._write_frame = capture_write
        modem.is_connected = True

        result = modem.set_kiss_txtail(50)
        assert result is True
        assert written[0][1] == KISS_CMD_TXTAIL
        assert written[0][2] == 5  # 50ms / 10

    def test_set_kiss_full_duplex_sends_correct_frame(self):
        """Test set_kiss_full_duplex sends KISS_CMD_FULLDUPLEX 0x01 or 0x00"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        written = []

        def capture_write(frame):
            written.append(bytes(frame))
            return True

        modem._write_frame = capture_write
        modem.is_connected = True

        modem.set_kiss_full_duplex(True)
        assert written[0][1] == KISS_CMD_FULLDUPLEX
        assert written[0][2] == 0x01
        written.clear()
        modem.set_kiss_full_duplex(False)
        assert written[0][2] == 0x00

    def test_set_signal_report_returns_true_on_ok_response(self):
        """Test set_signal_report returns True when modem responds OK or SignalReport"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == HW_CMD_SET_SIGNAL_REPORT:
                return (HW_RESP_SIGNAL_REPORT, bytes([0x01]))
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        assert modem.set_signal_report(True) is True
        assert modem.set_signal_report(False) is True

    def test_set_signal_report_returns_true_on_ok(self):
        """Test set_signal_report returns True when modem responds HW_RESP_OK"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == HW_CMD_SET_SIGNAL_REPORT:
                return (HW_RESP_OK, b"")
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        assert modem.set_signal_report(True) is True

    def test_set_signal_report_returns_false_on_error_or_timeout(self):
        """Test set_signal_report returns False on error or timeout"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        assert modem.set_signal_report(True) is False

    def test_get_signal_report_returns_true_when_enabled(self):
        """Test get_signal_report returns True when modem reports enabled"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == HW_CMD_GET_SIGNAL_REPORT:
                return (HW_RESP_SIGNAL_REPORT, bytes([0x01]))
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        assert modem.get_signal_report() is True

    def test_get_signal_report_returns_false_when_disabled(self):
        """Test get_signal_report returns False when modem reports disabled"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            if cmd == HW_CMD_GET_SIGNAL_REPORT:
                return (HW_RESP_SIGNAL_REPORT, bytes([0x00]))
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        assert modem.get_signal_report() is False

    def test_get_signal_report_returns_none_on_timeout(self):
        """Test get_signal_report returns None on timeout or error"""
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)

        def mock_send_command(cmd, data=b"", timeout=5.0):
            return None

        modem._send_command = mock_send_command
        modem.is_connected = True

        assert modem.get_signal_report() is None


class TestContextManager:
    """Test context manager functionality"""

    def test_context_manager_calls_connect_disconnect(self):
        """Test context manager calls connect and disconnect"""
        with patch.object(KissModemWrapper, "connect", return_value=True) as mock_connect:
            with patch.object(KissModemWrapper, "disconnect") as mock_disconnect:
                with KissModemWrapper(port="/dev/null", auto_configure=False) as modem:
                    pass  # keep reference so __del__ doesn't run before assert

                mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()
                _ = modem  # hold ref so __del__ runs after assert, not before


class TestSerialRecovery:
    """Test serial degraded-state and reconnect behavior."""

    def test_write_frame_marks_degraded_and_triggers_reconnect(self):
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True

        class _FailingSerial:
            is_open = True

            def write(self, _data):
                raise OSError(5, "Input/output error")

        modem.serial_conn = _FailingSerial()
        modem._start_reconnect_worker = MagicMock()

        frame = modem._encode_kiss_frame(CMD_DATA, b"\x01\x02")
        assert modem._write_frame(frame) is False
        assert modem._degraded is True
        assert modem.is_connected is False
        assert modem.serial_conn is None
        modem._start_reconnect_worker.assert_called_once()

    def test_send_command_fails_fast_while_reconnecting(self):
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True
        modem._reconnecting_event.set()
        modem._write_frame = MagicMock(return_value=True)

        assert modem._send_command(CMD_PING, timeout=0.1) is None
        modem._write_frame.assert_not_called()

    def test_send_command_allowed_from_reconnect_thread_during_reconnect(self):
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True
        modem._reconnecting_event.set()
        modem.reconnect_thread = threading.current_thread()
        modem._response_queue.append((RESP_PONG, b""))

        assert modem._send_command(CMD_PING, timeout=0.1) == (RESP_PONG, b"")

    def test_send_command_allowed_from_reconnect_thread_while_degraded(self):
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem.is_connected = True
        modem._degraded = True
        modem.reconnect_thread = threading.current_thread()
        modem._response_queue.append((RESP_PONG, b""))

        assert modem._send_command(CMD_PING, timeout=0.1) == (RESP_PONG, b"")

    def test_reconnect_worker_recovers_after_open_failure(self):
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem._reconnecting_event.set()
        modem._degraded = True
        modem._degraded_reason = "test failure"
        modem._reconnect_base_delay_s = 0.0
        modem._reconnect_max_delay_s = 0.0

        modem._open_serial_and_start_threads = MagicMock(side_effect=[False, True])
        modem._run_post_connect_handshake = MagicMock(return_value=True)
        modem._stop_io_threads = MagicMock()

        with patch("pymc_core.hardware.kiss_modem_wrapper.time.sleep", return_value=None):
            modem._reconnect_worker()

        assert modem._open_serial_and_start_threads.call_count == 2
        assert modem._run_post_connect_handshake.call_count == 1
        assert modem._degraded is False
        assert modem._reconnecting_event.is_set() is False

    def test_start_reconnect_worker_guard_prevents_duplicate_thread(self):
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem._reconnecting_event.set()
        modem._start_reconnect_worker()
        assert modem.reconnect_thread is None

    def test_connect_clears_reconnecting_gate_after_success(self):
        modem = KissModemWrapper(port="/dev/null", auto_configure=False)
        modem._reconnecting_event.set()
        modem._open_serial_and_start_threads = MagicMock(return_value=True)
        modem._run_post_connect_handshake = MagicMock(return_value=True)

        assert modem.connect() is True
        assert modem._reconnecting_event.is_set() is False
