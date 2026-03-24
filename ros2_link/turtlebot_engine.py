"""Local TurtleBot3 AI engine — LIDAR-aware waypoint navigation.

This is the ``make_move`` tool: given the current robot state (pose + LIDAR),
it computes velocity commands to navigate through waypoints while avoiding
obstacles detected by the simulated LIDAR scanner.

Navigation strategy:
  1. Pure-pursuit steering toward the next unreached waypoint.
  2. VFH-lite obstacle avoidance — if an obstacle is detected within the
     safety margin, the robot steers away from the nearest blocked sector.
  3. Waypoint is marked as reached when the robot is within threshold distance.
"""

from __future__ import annotations

import math
from typing import Any


# Navigation constants
WAYPOINT_REACH_DIST = 0.35      # metres — waypoint reached threshold
MAX_VX = 0.20                   # m/s — cruise speed
MIN_VX = 0.04                   # m/s — creep speed while turning or avoiding
MAX_WZ = 2.0                    # rad/s — max angular velocity
KP_ANGULAR = 2.5                # proportional gain for heading correction
HEADING_SLOW_THRESH = 0.5       # rad — slow down when heading error exceeds this

# Obstacle avoidance
OBSTACLE_STOP_DIST = 0.25       # m — emergency stop
OBSTACLE_SLOW_DIST = 0.60       # m — start slowing
OBSTACLE_AVOID_DIST = 0.90      # m — begin avoidance steering
AVOIDANCE_WZ = 1.8              # rad/s — avoidance angular velocity

# Sector indices (matching the 8-sector layout from the simulator)
SECTOR_NAMES = ["front", "front_left", "left", "rear_left",
                "rear", "rear_right", "right", "front_right"]


def _normalise_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class TurtleBotNavigator:
    """Stateful waypoint navigator with LIDAR-based obstacle avoidance."""

    def __init__(self) -> None:
        self.current_wp = 0
        self.total_reached = 0
        self.started = False
        self._avoidance_direction = 0  # -1 = right, +1 = left, 0 = none

    @property
    def finished(self) -> bool:
        return False  # runs until all waypoints are reached (checked externally)

    def decide_action(self, state: dict[str, Any]) -> dict[str, Any]:
        """Decide drive command from full state snapshot.

        Returns dict with keys: vx, wz, action, waypoint_reached (index or -1).
        """
        self.started = True
        rx = state["x"]
        ry = state["y"]
        ryaw = state["yaw"]
        sectors = state.get("lidar_sectors", {})
        wp_list = state.get("waypoints", [])

        # Find the next unreached waypoint
        target_idx = -1
        for i, wp in enumerate(wp_list):
            if not wp["reached"]:
                target_idx = i
                break

        # All done
        if target_idx < 0:
            return {"vx": 0.0, "wz": 0.0, "action": "STOP", "waypoint_reached": -1}

        target = wp_list[target_idx]
        tx, ty = target["x"], target["y"]

        # Check if we've reached it
        dist_to_wp = math.hypot(tx - rx, ty - ry)
        if dist_to_wp < WAYPOINT_REACH_DIST:
            self.total_reached += 1
            return {
                "vx": 0.0, "wz": 0.0,
                "action": "REACHED",
                "waypoint_reached": target_idx,
            }

        # ── Obstacle avoidance layer ──
        front = sectors.get("front", 3.5)
        front_left = sectors.get("front_left", 3.5)
        front_right = sectors.get("front_right", 3.5)
        left = sectors.get("left", 3.5)
        right = sectors.get("right", 3.5)
        min_front = min(front, front_left, front_right)

        # Emergency stop
        if front < OBSTACLE_STOP_DIST:
            # Choose avoidance direction based on which side has more room
            if right > left:
                self._avoidance_direction = -1
            else:
                self._avoidance_direction = 1
            return {
                "vx": 0.0,
                "wz": AVOIDANCE_WZ * self._avoidance_direction,
                "action": "AVOID_STOP",
                "waypoint_reached": -1,
            }

        # Active avoidance steering
        if min_front < OBSTACLE_AVOID_DIST:
            # Determine direction: steer toward the more open side
            left_clearance = (front_left + left) / 2
            right_clearance = (front_right + right) / 2

            if right_clearance > left_clearance:
                self._avoidance_direction = -1  # turn right
            else:
                self._avoidance_direction = 1   # turn left

            # Scale speed based on proximity
            speed_factor = max(0.0, (min_front - OBSTACLE_STOP_DIST) /
                               (OBSTACLE_AVOID_DIST - OBSTACLE_STOP_DIST))
            vx = MIN_VX + (MAX_VX - MIN_VX) * speed_factor * 0.5
            wz = AVOIDANCE_WZ * self._avoidance_direction * (1.0 - speed_factor * 0.5)

            return {
                "vx": vx, "wz": wz,
                "action": "AVOIDING",
                "waypoint_reached": -1,
            }

        # ── Pure-pursuit toward waypoint ──
        self._avoidance_direction = 0
        bearing = math.atan2(ty - ry, tx - rx)
        heading_error = _normalise_angle(bearing - ryaw)

        # Angular velocity — proportional control
        wz = KP_ANGULAR * heading_error
        wz = max(-MAX_WZ, min(MAX_WZ, wz))

        # Linear velocity — reduce when turning
        if abs(heading_error) > HEADING_SLOW_THRESH:
            vx = MIN_VX
        else:
            vx = MAX_VX * (1.0 - abs(heading_error) / math.pi)
            vx = max(MIN_VX, min(MAX_VX, vx))

        # Slow down near obstacles even when not in avoidance mode
        if min_front < OBSTACLE_SLOW_DIST:
            slow_factor = (min_front - OBSTACLE_STOP_DIST) / (OBSTACLE_SLOW_DIST - OBSTACLE_STOP_DIST)
            vx *= max(0.2, slow_factor)

        action = "FORWARD"
        if abs(heading_error) > 0.3:
            action = "LEFT" if heading_error > 0 else "RIGHT"

        return {"vx": vx, "wz": wz, "action": action, "waypoint_reached": -1}


def state_summary(state: dict[str, Any], navigator: TurtleBotNavigator) -> str:
    """Build a concise text summary for the OmniLink agent."""
    x = state.get("x", 0.0)
    y = state.get("y", 0.0)
    yaw_deg = state.get("yaw_deg", 0.0)
    sectors = state.get("lidar_sectors", {})
    wp_list = state.get("waypoints", [])
    reached = state.get("waypoints_reached", 0)
    total = state.get("waypoints_total", 0)
    dist_m = state.get("total_distance_m", 0.0)
    elapsed = state.get("elapsed_s", 0.0)
    collisions = state.get("collision_count", 0)
    lidar_min = state.get("lidar_min", 3.5)

    # Find current target
    target_str = "None (all reached)"
    dist_str = "N/A"
    for i, wp in enumerate(wp_list):
        if not wp["reached"]:
            target_str = f"WP{i+1} ({wp['x']:.1f}, {wp['y']:.1f})"
            dist_str = f"{math.hypot(wp['x'] - x, wp['y'] - y):.2f}m"
            break

    front = sectors.get("front", 3.5)
    obstacle_warning = ""
    if front < 0.5:
        obstacle_warning = " ⚠ OBSTACLE AHEAD"

    return (
        f"Pose: ({x:.2f}, {y:.2f}) heading={yaw_deg:.1f}°\n"
        f"Target: {target_str}  Distance: {dist_str}\n"
        f"LIDAR min: {lidar_min:.2f}m  Front: {front:.2f}m{obstacle_warning}\n"
        f"Waypoints: {reached}/{total}  Collisions: {collisions}\n"
        f"Odometry: {dist_m:.1f}m  Elapsed: {elapsed:.0f}s\n"
        f"Status: {'Mission complete' if reached == total else 'Navigating'}"
    )
