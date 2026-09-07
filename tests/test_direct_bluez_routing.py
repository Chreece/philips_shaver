from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from homeassistant.data_entry_flow import FlowResultType

from custom_components.philips_shaver import transport as tr
from custom_components.philips_shaver.config_flow import PhilipsShaverConfigFlow

ADDRESS = "F4:B3:B1:AA:BB:CC"


def test_local_selector_ignores_stronger_proxy(monkeypatch) -> None:
    class LocalScanner:
        pass

    local_device = object()
    local = SimpleNamespace(
        scanner=LocalScanner(),
        advertisement=SimpleNamespace(rssi=-82),
        ble_device=local_device,
    )
    proxy = SimpleNamespace(
        scanner=SimpleNamespace(),
        advertisement=SimpleNamespace(rssi=-42),
        ble_device=object(),
    )
    monkeypatch.setattr(tr, "HaScanner", LocalScanner)
    monkeypatch.setattr(
        tr,
        "async_scanner_devices_by_address",
        lambda hass, address, connectable=True: [proxy, local],
    )

    selected = tr.local_bluez_scanner_device_from_address(
        SimpleNamespace(), ADDRESS
    )

    assert selected is local
    assert tr.local_bluez_device_from_address(SimpleNamespace(), ADDRESS) is local_device


def test_direct_preview_prefers_local_over_stronger_proxy(monkeypatch) -> None:
    flow = PhilipsShaverConfigFlow()
    flow.flow_id = "test-flow"
    flow.handler = "philips_shaver"
    flow.discovery_info = SimpleNamespace(address=ADDRESS, name="Philips Shaver")
    flow.hass = SimpleNamespace()
    monkeypatch.setattr(
        "custom_components.philips_shaver.config_flow.describe_available_paths",
        MagicMock(return_value=[
            {"name": "aquarium-multisensor", "rssi": -42, "is_local": False},
            {"name": "hci0", "rssi": -82, "is_local": True},
        ]),
    )

    via, warning, _values = flow._transport_lines()

    assert via == " via **Direct Bluetooth** (hci0, -82 dBm)"
    assert warning == ""


async def test_manual_selection_does_not_abort_for_discovery_flow(monkeypatch) -> None:
    class FakeTask:
        def done(self) -> bool:
            return False

    flow = PhilipsShaverConfigFlow()
    flow.flow_id = "test-flow"
    flow.handler = "philips_shaver"
    flow.discovery_info = None

    def create_task(coro, *args, **kwargs):
        coro.close()
        return FakeTask()

    flow.hass = SimpleNamespace(async_create_task=MagicMock(side_effect=create_task))
    flow.async_set_unique_id = AsyncMock()
    flow._abort_if_already_configured = MagicMock()
    monkeypatch.setattr(
        "custom_components.philips_shaver.config_flow.describe_available_paths",
        MagicMock(return_value=[{"name": "hci0", "rssi": -60, "is_local": True}]),
    )
    monkeypatch.setattr(
        "custom_components.philips_shaver.dbus_pairing.is_dbus_available",
        lambda: False,
    )

    result = await flow.async_step_user_bleak({"address": ADDRESS})

    assert result["type"] == FlowResultType.SHOW_PROGRESS
    flow.async_set_unique_id.assert_awaited_once_with(
        ADDRESS, raise_on_progress=False
    )
    flow._abort_if_already_configured.assert_called_once_with()


async def test_runtime_connect_keeps_local_scanner_for_rssi(monkeypatch) -> None:
    scanner = SimpleNamespace()
    device = SimpleNamespace(address=ADDRESS, name="Philips QP4530")
    scanner_device = SimpleNamespace(scanner=scanner, ble_device=device)
    client = SimpleNamespace(is_connected=True)
    establish = AsyncMock(return_value=client)

    monkeypatch.setattr(
        tr, "local_bluez_scanner_device_from_address",
        lambda hass, address: scanner_device,
    )
    monkeypatch.setattr(tr, "bleak_establish", establish)
    monkeypatch.setattr(tr, "describe_connection_path", lambda *args: "hci0")

    transport = tr.BleakTransport(SimpleNamespace(), ADDRESS)
    await transport.connect()

    assert transport._connected_scanner is scanner
    assert transport.connection_path == "hci0"
    assert establish.await_args.args[0] is tr.ORIGINAL_BLEAK_CLIENT
    assert establish.await_args.args[1] is device
