"""HTTP client for the TurtleBot3 2D simulator."""

from __future__ import annotations

from typing import Any

import requests

SERVER_URL = "http://127.0.0.1:5000"

_session = requests.Session()


def get_state() -> dict[str, Any]:
    """Fetch full state snapshot (pose + LIDAR sectors + waypoints + stats)."""
    r = _session.get(f"{SERVER_URL}/state", timeout=2)
    return r.json()


def get_pose() -> dict[str, Any]:
    r = _session.get(f"{SERVER_URL}/pose", timeout=2)
    return r.json()


def get_scan() -> dict[str, Any]:
    r = _session.get(f"{SERVER_URL}/scan", timeout=2)
    return r.json()


def drive(vx: float, wz: float, duration: float | None = None) -> None:
    """Send velocity command."""
    payload: dict[str, Any] = {"vx": vx, "wz": wz}
    if duration is not None:
        payload["duration"] = duration
    _session.post(f"{SERVER_URL}/drive", json=payload, timeout=2)


def stop() -> None:
    _session.post(f"{SERVER_URL}/stop", timeout=2)


def reset() -> None:
    _session.post(f"{SERVER_URL}/reset", timeout=2)


def reach_waypoint(index: int) -> None:
    _session.post(f"{SERVER_URL}/waypoints/reach", json={"index": index}, timeout=2)
