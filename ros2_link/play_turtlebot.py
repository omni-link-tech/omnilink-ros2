"""Play TurtleBot3 navigation using OmniLink tool calling.

The AI agent calls the ``make_move`` tool, which acts as a local
LIDAR-aware waypoint navigator.  The model triggers the tool once;
the tool reads the robot state (pose + LIDAR), computes obstacle-aware
steering, drives toward waypoints, and marks them as reached.

This keeps API credit usage to a minimum (one call to kick off).

Usage
-----
    python -u play_turtlebot.py
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

# ── Path setup ─────────────────────────────────────────────────────────
_HERE = str(pathlib.Path(__file__).resolve().parent)
LIB_PATH = str(pathlib.Path(__file__).resolve().parents[3] / "omnilink-lib" / "src")
if _HERE in sys.path:
    sys.path.remove(_HERE)
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

from omnilink.tool_runner import ToolRunner

if _HERE not in sys.path:
    sys.path.append(_HERE)

from turtlebot_api import get_state, drive, stop, reach_waypoint
from turtlebot_engine import TurtleBotNavigator, state_summary


class TurtleBotRunner(ToolRunner):
    agent_name = "turtlebot-ros2-agent"
    display_name = "TurtleBot3"
    tool_description = "Navigate waypoints with LIDAR obstacle avoidance."
    poll_interval = 0.05
    memory_every = 10
    ask_every = 30
    commands = "stop_game, pause_game, resume_game"

    def __init__(self) -> None:
        self._navigator = TurtleBotNavigator()
        self._last_reached = 0

    def get_state(self) -> dict[str, Any]:
        return get_state()

    def execute_action(self, state: dict[str, Any]) -> None:
        cmd = self._navigator.decide_action(state)

        # Mark waypoint reached on the simulator side
        wp_idx = cmd.get("waypoint_reached", -1)
        if wp_idx >= 0:
            try:
                reach_waypoint(wp_idx)
            except Exception:
                pass

        try:
            drive(cmd["vx"], cmd["wz"])
        except Exception:
            pass

    def state_summary(self, state: dict[str, Any]) -> str:
        return state_summary(state, self._navigator)

    def is_game_over(self, state: dict[str, Any]) -> bool:
        return state.get("mission_complete", False)

    def game_over_message(self, state: dict[str, Any]) -> str:
        dist = state.get("total_distance_m", 0)
        elapsed = state.get("elapsed_s", 0)
        collisions = state.get("collision_count", 0)
        wp_total = state.get("waypoints_total", 0)
        return (
            f"MISSION COMPLETE — All {wp_total} waypoints reached!\n"
            f"Distance: {dist:.1f}m | Time: {elapsed:.0f}s | Collisions: {collisions}"
        )

    def log_events(self, state: dict[str, Any]) -> None:
        reached = state.get("waypoints_reached", 0)
        if reached != self._last_reached:
            total = state.get("waypoints_total", 0)
            print(f"  Waypoint {reached}/{total} reached!")
            self._last_reached = reached


if __name__ == "__main__":
    TurtleBotRunner().run()
