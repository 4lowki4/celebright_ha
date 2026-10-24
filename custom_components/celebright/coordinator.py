"""DataUpdateCoordinator — polls Celebright device statuses and scene lists."""
from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import CelebrightCloudAPI, CelebrightConnectionError, DeviceInfo, DeviceState, SceneInfo
from .const import CONF_EMAIL, CONF_PASSWORD, DOMAIN, SCAN_INTERVAL_SECONDS

_LOGGER = logging.getLogger(__name__)

# Scenes/schedule change rarely — auto-refresh them at most this often (seconds).
SCENE_REFRESH_INTERVAL = 1800  # 30 minutes

# After an HA-initiated command the cloud status endpoint lags the device by
# up to a poll cycle or two. Hold the optimistic state until the cloud agrees,
# or this many seconds pass (then trust the cloud again).
OPTIMISTIC_HOLD_SECONDS = 90


class CelebrightCoordinator(DataUpdateCoordinator[dict[str, DeviceState]]):
    """Polls device statuses; device info and scene lists are fetched once at setup."""

    def __init__(self, hass: HomeAssistant, entry_data: dict[str, Any]) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=SCAN_INTERVAL_SECONDS),
        )
        self.client = CelebrightCloudAPI(
            email=entry_data[CONF_EMAIL],
            password=entry_data[CONF_PASSWORD],
        )
        # Public, stable device metadata — keyed by device_id
        self.device_infos: dict[str, DeviceInfo] = {}
        # Saved scene lists — keyed by device_id
        self.scenes: dict[str, list[SceneInfo]] = {}
        # Optimistic active-scene tracking — device_id → scene_uuid | None
        self._active_scenes: dict[str, str | None] = {}
        # Pending optimistic on/off — device_id → (expected_is_on, expires_monotonic)
        self._pending: dict[str, tuple[bool, float]] = {}
        # When scenes/device-info were last pulled (monotonic seconds)
        self._last_scene_refresh: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def async_setup(self) -> None:
        await self.client.async_connect()
        # Copy device infos to a public attribute so all entities can read them
        # without reaching into the client's private state.
        self.device_infos = dict(self.client._device_infos)
        for device_id in self.device_infos:
            await self._fetch_scenes(device_id)
        self._last_scene_refresh = time.monotonic()

    async def async_shutdown(self) -> None:
        await self.client.async_disconnect()

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    async def _async_update_data(self) -> dict[str, DeviceState]:
        try:
            statuses = await self.client.async_get_device_statuses()
        except CelebrightConnectionError as err:
            raise UpdateFailed(f"Celebright unreachable: {err}") from err

        # Reconcile any optimistic state from a recent HA command: keep it until
        # the cloud reports the same on/off, or the hold window expires.
        now = time.monotonic()
        previous = self.data or {}
        for device_id, state in statuses.items():
            pending = self._pending.get(device_id)
            if not pending:
                continue
            expected_on, expires = pending
            if state.is_on == expected_on or now >= expires:
                self._pending.pop(device_id, None)
                continue
            prev = previous.get(device_id)
            if prev is not None:
                state.is_on = prev.is_on
                state.current_display = prev.current_display
                state.color_hex = prev.color_hex
                state.brightness = prev.brightness
                state.active_scene_uuid = prev.active_scene_uuid
            else:
                state.is_on = expected_on

        for device_id, state in statuses.items():
            if not state.is_on:
                self._active_scenes[device_id] = None
            elif state.active_scene_uuid is not None:
                # Real MQTT-reported value — authoritative. Keep the optimistic
                # cache in sync so a later poll that can't reach MQTT falls back
                # to the last known-good real value instead of a stale
                # HA-initiated selection that may no longer be what's showing
                # (e.g. the device's own schedule took over in the meantime).
                self._active_scenes[device_id] = state.active_scene_uuid
            elif device_id in self._active_scenes:
                state.active_scene_uuid = self._active_scenes[device_id]

        # Periodically pick up scene/schedule changes made in the Celebright app.
        if time.monotonic() - self._last_scene_refresh >= SCENE_REFRESH_INTERVAL:
            await self._refresh_all(notify=False)

        return statuses

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    async def _fetch_scenes(self, device_id: str) -> None:
        try:
            self.scenes[device_id] = await self.client.async_get_scenes(device_id)
            _LOGGER.debug("Loaded %d scenes for %s", len(self.scenes[device_id]), device_id)
        except Exception as err:
            _LOGGER.warning("Could not load scenes for %s: %s", device_id, err)
            self.scenes.setdefault(device_id, [])

    def scene_name_for_uuid(self, device_id: str, uuid: str | None) -> str | None:
        if not uuid:
            return None
        for scene in self.scenes.get(device_id, []):
            if scene.uuid == uuid:
                return scene.name
        return None

    def uuid_for_scene_name(self, device_id: str, name: str) -> str | None:
        for scene in self.scenes.get(device_id, []):
            if scene.name == name:
                return scene.uuid
        return None

    def set_active_scene(self, device_id: str, scene_uuid: str | None) -> None:
        self._active_scenes[device_id] = scene_uuid

    def set_optimistic(
        self,
        device_id: str,
        *,
        is_on: bool,
        display: str,
        color_hex: str | None = None,
        scene_uuid: str | None = None,
    ) -> None:
        """Show a just-sent command's result immediately, before the cloud catches up.

        Mutates the current state in place, notifies entities, and records a
        pending expectation that _async_update_data holds until the cloud agrees.
        """
        state = (self.data or {}).get(device_id)
        if state is None:
            return
        state.is_on = is_on
        state.current_display = display
        state.active_scene_uuid = scene_uuid if is_on else None
        if is_on and color_hex:
            state.color_hex = color_hex
            try:
                state.brightness = max(
                    int(color_hex[i:i + 2], 16) for i in (0, 2, 4)
                )
            except ValueError:
                state.brightness = None
        elif not is_on:
            state.color_hex = None
            state.brightness = None
        self._active_scenes[device_id] = state.active_scene_uuid
        self._pending[device_id] = (is_on, time.monotonic() + OPTIMISTIC_HOLD_SECONDS)
        self.async_update_listeners()

    # ------------------------------------------------------------------
    # Refresh (scenes + device info/schedule)
    # ------------------------------------------------------------------

    async def async_refresh_scenes(self) -> None:
        """Manually re-pull scene lists (and device info/schedule) from the account."""
        await self._refresh_all(notify=True)

    async def _refresh_all(self, notify: bool) -> None:
        """Re-fetch device infos and scene lists for all devices."""
        try:
            self.device_infos = dict(await self.client.async_get_device_infos())
            for device_id in self.device_infos:
                await self._fetch_scenes(device_id)
            self._last_scene_refresh = time.monotonic()
            _LOGGER.debug("Refreshed scenes/device info for %d device(s)", len(self.device_infos))
        except Exception as err:  # noqa: BLE001 - refresh is best-effort
            _LOGGER.warning("Scene refresh failed: %s", err)
            # Back off so a failing refresh doesn't hammer the API every poll
            self._last_scene_refresh = time.monotonic()
        if notify:
            self.async_update_listeners()

    # ------------------------------------------------------------------
    # Schedule writes
    # ------------------------------------------------------------------

    async def async_create_event(self, device_id: str, **kwargs: Any) -> dict[str, Any]:
        """Create (or, with event_uuid, update) a schedule event, then refresh."""
        ack = await self.client.async_create_event(device_id, **kwargs)
        await self._refresh_device_infos("create_event")
        return ack

    async def async_delete_event(self, device_id: str, event_uuid: str) -> dict[str, Any]:
        """Delete a schedule event by UUID, then refresh cached device info."""
        ack = await self.client.async_delete_event(device_id, event_uuid)
        await self._refresh_device_infos("delete_event")
        return ack

    async def _refresh_device_infos(self, why: str) -> None:
        """Re-fetch device infos so schedule changes appear in entities."""
        try:
            self.device_infos = dict(await self.client.async_get_device_infos())
        except Exception as err:  # noqa: BLE001 - refresh is best-effort
            _LOGGER.debug("Could not refresh device infos after %s: %s", why, err)
        self.async_update_listeners()

    def event_uuid_for_name(self, device_id: str, name: str) -> str | None:
        """Resolve a schedule event UUID by its event name (first match)."""
        info = self.device_infos.get(device_id)
        if not info:
            return None
        for ev in info.events:
            if ev.name == name:
                return ev.uuid
        return None

    def first_device_id(self) -> str | None:
        return next(iter(self.device_infos), None)
