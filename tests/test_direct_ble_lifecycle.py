"""Regression tests for direct-BLE reconnects and AX210 crash mitigation.

All tests use fake Bluetooth clients. No radio, BlueZ adapter, or shaver is
accessed by the test suite.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.philips_shaver import transport as tr
from custom_components.philips_shaver.coordinator import PhilipsShaverCoordinator
from custom_components.philips_shaver.exceptions import TransportError


ADDRESS = "F4:B3:B1:AA:BB:CC"


def _direct_transport(monkeypatch) -> tr.BleakTransport:
    scanner = SimpleNamespace(name="hci0")
    device = SimpleNamespace(address=ADDRESS, name="Philips QP4530")
    monkeypatch.setattr(
        tr, "local_bluez_scanner_device_from_address",
        lambda hass, address: SimpleNamespace(scanner=scanner, ble_device=device),
    )
    monkeypatch.setattr(tr, "is_local_bluez_connection", lambda client: True)
    monkeypatch.setattr(tr, "describe_connection_path", lambda *args: "hci0")
    return tr.BleakTransport(SimpleNamespace(), ADDRESS)


def _client() -> SimpleNamespace:
    return SimpleNamespace(
        is_connected=True,
        _connected_scanner=None,
        disconnect=AsyncMock(),
    )


async def test_repeated_connect_reuses_active_client(monkeypatch) -> None:
    transport = _direct_transport(monkeypatch)
    client = _client()
    establish = AsyncMock(return_value=client)
    monkeypatch.setattr(tr, "bleak_establish", establish)

    await transport.connect()
    await transport.connect()

    assert transport._client is client
    establish.assert_awaited_once()


async def test_concurrent_connect_calls_use_one_radio_connection(monkeypatch) -> None:
    transport = _direct_transport(monkeypatch)
    client = _client()
    entered, release = asyncio.Event(), asyncio.Event()

    async def establish(*args, **kwargs):
        entered.set()
        await release.wait()
        return client

    establish_mock = AsyncMock(side_effect=establish)
    monkeypatch.setattr(tr, "bleak_establish", establish_mock)

    first = asyncio.create_task(transport.connect())
    await asyncio.wait_for(entered.wait(), timeout=5)
    second = asyncio.create_task(transport.connect())
    release.set()
    await asyncio.gather(first, second)

    establish_mock.assert_awaited_once()
    assert transport._client is client


async def test_stale_disconnect_does_not_clear_new_client(monkeypatch) -> None:
    transport = _direct_transport(monkeypatch)
    first, second = _client(), _client()
    callbacks = []
    clients = iter((first, second))

    async def establish(*args, disconnected_callback, **kwargs):
        callbacks.append(disconnected_callback)
        return next(clients)

    monkeypatch.setattr(tr, "bleak_establish", AsyncMock(side_effect=establish))
    on_disconnect = MagicMock()
    transport.set_disconnect_callback(on_disconnect)

    await transport.connect()
    # Link one drops but its callback is delivered after the replacement
    # connection has already completed.
    first.is_connected = False
    await transport.connect()
    assert transport._client is second

    callbacks[0](first)
    assert transport._client is second
    assert transport.is_connected
    on_disconnect.assert_not_called()

    callbacks[1](second)
    assert transport._client is None
    on_disconnect.assert_called_once_with()


async def test_intentional_disconnect_ignores_delayed_callback(monkeypatch) -> None:
    transport = _direct_transport(monkeypatch)
    client = _client()
    callbacks = []

    async def establish(*args, disconnected_callback, **kwargs):
        callbacks.append(disconnected_callback)
        return client

    monkeypatch.setattr(tr, "bleak_establish", AsyncMock(side_effect=establish))
    on_disconnect = MagicMock()
    transport.set_disconnect_callback(on_disconnect)
    await transport.connect()

    async def disconnect_client():
        # Simulate BlueZ invoking the callback while disconnect() is awaited.
        callbacks[0](client)
        client.is_connected = False

    client.disconnect = AsyncMock(side_effect=disconnect_client)
    await transport.disconnect()

    assert transport._client is None
    assert transport._connection_path is None
    assert transport._connected_scanner is None
    client.disconnect.assert_awaited_once()
    on_disconnect.assert_not_called()


async def test_disconnected_during_establish_is_not_accepted(monkeypatch) -> None:
    transport = _direct_transport(monkeypatch)
    client = _client()
    client.is_connected = False
    monkeypatch.setattr(tr, "bleak_establish", AsyncMock(return_value=client))

    with pytest.raises(TransportError, match="disconnected during BLE setup"):
        await transport.connect()

    assert transport._client is None


def _coordinator(transport) -> PhilipsShaverCoordinator:
    """Construct only the state the live monitor uses, with no HA instance."""
    coordinator = object.__new__(PhilipsShaverCoordinator)
    coordinator.address = ADDRESS
    coordinator.transport = transport
    coordinator._is_esp_bridge = False
    coordinator._connection_lock = asyncio.Lock()
    coordinator._wake_event = asyncio.Event()
    coordinator._adv_wake = False
    coordinator._live_setup_done = False
    coordinator._full_read_done = True
    coordinator._live_chars = []
    coordinator._clear_adv_history = MagicMock()
    coordinator._start_all_notifications = AsyncMock(return_value=1)
    coordinator.async_set_updated_data = MagicMock()
    coordinator.data = {}
    return coordinator


async def test_existing_link_is_used_for_live_setup() -> None:
    ready = asyncio.Event()

    async def notifications():
        ready.set()
        return 1

    transport = SimpleNamespace(
        is_connected=True,
        connect=AsyncMock(),
        disconnect=AsyncMock(),
        unsubscribe_all=AsyncMock(),
        set_disconnect_callback=MagicMock(),
    )
    coord = _coordinator(transport)
    coord._start_all_notifications = AsyncMock(side_effect=notifications)

    task = asyncio.create_task(coord._start_live_monitoring())
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        transport.connect.assert_not_awaited()
        coord._start_all_notifications.assert_awaited_once()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_disconnect_wake_without_advertisement_cannot_connect() -> None:
    rearmed = asyncio.Event()
    transport = SimpleNamespace(
        is_connected=False,
        connect=AsyncMock(),
        disconnect=AsyncMock(),
        unsubscribe_all=AsyncMock(),
        set_disconnect_callback=MagicMock(),
    )
    coord = _coordinator(transport)
    coord._clear_adv_history = MagicMock(side_effect=rearmed.set)
    coord._wake_event.set()  # disconnect nudge, _adv_wake remains False

    task = asyncio.create_task(coord._start_live_monitoring())
    try:
        await asyncio.wait_for(rearmed.wait(), timeout=5)
        # Another disconnect nudge while waiting must not attempt connect.
        coord._wake_event.set()
        await asyncio.sleep(0)
        transport.connect.assert_not_awaited()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_failed_setup_uses_backoff_without_blind_retries(monkeypatch) -> None:
    sleeping = asyncio.Event()
    real_sleep = asyncio.sleep
    requested_delays = []

    async def capture_sleep(seconds):
        requested_delays.append(seconds)
        if seconds >= 30:
            sleeping.set()
            await asyncio.Event().wait()  # cancellable, no real 30s wait
        else:
            await real_sleep(0)

    # Module-level asyncio is shared with Python, so keep the real sleep
    # callable for short yields and restore it via monkeypatch teardown.
    from custom_components.philips_shaver import coordinator as coordinator_module

    monkeypatch.setattr(coordinator_module.asyncio, "sleep", capture_sleep)
    failed = asyncio.Event()

    async def connect():
        raise TransportError("test setup failure")

    async def disconnect():
        failed.set()

    transport = SimpleNamespace(
        is_connected=False,
        connect=AsyncMock(side_effect=connect),
        disconnect=AsyncMock(side_effect=disconnect),
        unsubscribe_all=AsyncMock(),
        set_disconnect_callback=MagicMock(),
    )
    coord = _coordinator(transport)
    coord._adv_wake = True
    coord._wake_event.set()

    task = asyncio.create_task(coord._start_live_monitoring())
    try:
        await asyncio.wait_for(failed.wait(), timeout=5)
        await asyncio.wait_for(sleeping.wait(), timeout=5)
        assert requested_delays == [30]
        transport.connect.assert_awaited_once()
        transport.disconnect.assert_awaited_once()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
