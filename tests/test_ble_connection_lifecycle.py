"""Direct Bluetooth lifecycle regressions, without a radio or shaver.

This PR deliberately leaves the coordinator's advertisement-driven reconnect
policy alone; it does not add a cooldown or exponential backoff.
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


def _transport(monkeypatch) -> tr.BleakTransport:
    device = SimpleNamespace(address=ADDRESS, name="Philips QP4530")
    monkeypatch.setattr(
        tr,
        "async_last_service_info",
        lambda hass, address: SimpleNamespace(device=device),
    )
    monkeypatch.setattr(tr, "describe_connection_path", lambda *args: "hci0")
    return tr.BleakTransport(SimpleNamespace(), ADDRESS)


def _client() -> SimpleNamespace:
    return SimpleNamespace(
        is_connected=True,
        _connected_scanner=None,
        disconnect=AsyncMock(),
    )


async def test_repeated_connect_reuses_active_client(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    client = _client()
    establish = AsyncMock(return_value=client)
    monkeypatch.setattr(tr, "bleak_establish", establish)

    await transport.connect()
    await transport.connect()

    assert transport._client is client
    establish.assert_awaited_once()


async def test_concurrent_connects_establish_only_once(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    client = _client()

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


async def test_stale_disconnect_callback_cannot_clear_replacement(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    first, second = _client(), _client()
    clients = iter((first, second))
    callbacks = []

    async def establish(*args, disconnected_callback, **kwargs):
        callbacks.append(disconnected_callback)
        return next(clients)

    monkeypatch.setattr(tr, "bleak_establish", AsyncMock(side_effect=establish))
    changed = MagicMock()
    transport.set_disconnect_callback(changed)

    await transport.connect()
    first.is_connected = False
    await transport.connect()
    callbacks[0](first)

    assert transport._client is second
    assert transport.is_connected
    changed.assert_not_called()

    callbacks[1](second)
    assert transport._client is None
    changed.assert_called_once_with()


async def test_intentional_disconnect_notifies_coordinator_once(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    client = _client()
    callbacks = []

    async def establish(*args, disconnected_callback, **kwargs):
        callbacks.append(disconnected_callback)
        return client

    monkeypatch.setattr(tr, "bleak_establish", AsyncMock(side_effect=establish))
    changed = MagicMock()
    transport.set_disconnect_callback(changed)
    await transport.connect()

    async def disconnect_client():
        # BlueZ might invoke the callback while disconnect() is in progress.
        callbacks[0](client)
        client.is_connected = False

    client.disconnect = AsyncMock(side_effect=disconnect_client)
    await transport.disconnect()

    assert transport._client is None
    assert transport._connection_path is None
    assert transport._connected_scanner is None
    client.disconnect.assert_awaited_once()
    changed.assert_called_once_with()

    callbacks[0](client)  # another callback after the method returned
    changed.assert_called_once_with()


async def test_intentional_disconnect_from_lost_link_still_notifies(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    client = _client()
    monkeypatch.setattr(tr, "bleak_establish", AsyncMock(return_value=client))
    changed = MagicMock()
    transport.set_disconnect_callback(changed)
    await transport.connect()

    client.is_connected = False  # link gone, but BlueZ callback not delivered
    await transport.disconnect()

    client.disconnect.assert_not_awaited()
    changed.assert_called_once_with()
    assert transport._client is None


async def test_redundant_disconnect_has_no_state_change(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    changed = MagicMock()
    transport.set_disconnect_callback(changed)

    await transport.disconnect()

    changed.assert_not_called()


async def test_disconnection_notifies_even_if_bluez_raises(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    client = _client()
    client.disconnect = AsyncMock(side_effect=OSError("adapter disappeared"))
    monkeypatch.setattr(tr, "bleak_establish", AsyncMock(return_value=client))
    changed = MagicMock()
    transport.set_disconnect_callback(changed)
    await transport.connect()

    await transport.disconnect()

    changed.assert_called_once_with()
    assert transport._client is None


async def test_connect_rejects_client_already_disconnected(monkeypatch) -> None:
    transport = _transport(monkeypatch)
    client = _client()
    client.is_connected = False
    monkeypatch.setattr(tr, "bleak_establish", AsyncMock(return_value=client))

    with pytest.raises(TransportError, match="disconnected during BLE setup"):
        await transport.connect()

    assert transport._client is None


async def test_failed_setup_clears_initializing_even_without_ble_client() -> None:
    """Failed establish() means Bleak never sends a disconnect callback."""
    started = asyncio.Event()
    transport = SimpleNamespace(
        is_connected=False,
        connect=AsyncMock(side_effect=TransportError("connection failed")),
        disconnect=AsyncMock(),
        set_disconnect_callback=MagicMock(),
        unsubscribe_all=AsyncMock(),
    )

    coord = object.__new__(PhilipsShaverCoordinator)
    coord.address = ADDRESS
    coord.transport = transport
    coord._is_esp_bridge = False
    coord._connection_lock = asyncio.Lock()
    coord._wake_event = asyncio.Event()
    coord._wake_event.set()
    coord._adv_wake = True
    coord._live_setup_done = False
    coord._full_read_done = True
    coord.data = {"device_state": "on", "_connecting": True}
    coord.async_set_updated_data = MagicMock(side_effect=lambda data: started.set())

    task = asyncio.create_task(coord._start_live_monitoring())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assert coord.data["device_state"] == "off"
        assert "_connecting" not in coord.data
        transport.disconnect.assert_awaited_once()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
