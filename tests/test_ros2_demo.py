"""Tests for the OmniLink ROS2 TurtleBot3 demo.

Tests cover:
  - Simulator geometry helpers (collision, raycasting)
  - LIDAR scanning
  - Navigation engine (waypoint pursuit, obstacle avoidance)
  - Flask REST API endpoints
"""

from __future__ import annotations

import json
import math
import sys
import pathlib
import pytest

# ── Path setup ─────────────────────────────────────────────────────────
_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
_LINK = str(pathlib.Path(__file__).resolve().parents[1] / "ros2_link")
for p in [_ROOT, _LINK]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ═══════════════════════════════════════════════════════════════════════
# 1. Simulator unit tests
# ═══════════════════════════════════════════════════════════════════════

class TestGeometryHelpers:
    """Test internal geometry functions from turtlebot_sim."""

    def test_clamp(self):
        from turtlebot_sim import _clamp
        assert _clamp(5.0, 0.0, 10.0) == 5.0
        assert _clamp(-1.0, 0.0, 10.0) == 0.0
        assert _clamp(15.0, 0.0, 10.0) == 10.0

    def test_rect_contains(self):
        from turtlebot_sim import _rect_contains
        # Point inside
        assert _rect_contains(1.0, 1.0, 2.0, 2.0, 2.0, 2.0)
        # Point outside
        assert not _rect_contains(1.0, 1.0, 2.0, 2.0, 5.0, 5.0)
        # With margin
        assert _rect_contains(1.0, 1.0, 2.0, 2.0, 0.9, 1.5, margin=0.2)

    def test_rect_contains_edge(self):
        from turtlebot_sim import _rect_contains
        # Point on the edge
        assert _rect_contains(1.0, 1.0, 2.0, 2.0, 1.0, 1.0)
        assert _rect_contains(1.0, 1.0, 2.0, 2.0, 3.0, 3.0)

    def test_ray_rect_intersect_hit(self):
        from turtlebot_sim import _ray_rect_intersect
        # Ray going right, hitting a box at x=2
        t = _ray_rect_intersect(0.0, 0.5, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0)
        assert t is not None
        assert abs(t - 2.0) < 0.01

    def test_ray_rect_intersect_miss(self):
        from turtlebot_sim import _ray_rect_intersect
        # Ray going up, box is to the right
        t = _ray_rect_intersect(0.0, 0.0, 0.0, 1.0, 5.0, 5.0, 1.0, 1.0)
        assert t is None

    def test_point_in_obstacle(self):
        from turtlebot_sim import _point_in_obstacle
        # Default obstacles include one at (3.0, 3.5, 0.6, 0.6)
        assert _point_in_obstacle(3.2, 3.7)
        assert not _point_in_obstacle(0.5, 0.5)


class TestLidar:
    """Test LIDAR raycasting."""

    def test_lidar_returns_360_rays(self):
        from turtlebot_sim import cast_lidar, LIDAR_RAYS
        ranges = cast_lidar()
        assert len(ranges) == LIDAR_RAYS

    def test_lidar_values_are_positive(self):
        from turtlebot_sim import cast_lidar
        ranges = cast_lidar()
        assert all(r > 0 for r in ranges)

    def test_lidar_detects_nearby_wall(self):
        """Robot at (0.5, 0.5) should detect left wall nearby."""
        import turtlebot_sim as sim
        old_x, old_y, old_yaw = sim.robot_x, sim.robot_y, sim.robot_yaw
        try:
            sim.robot_x = 0.5
            sim.robot_y = 0.5
            sim.robot_yaw = math.pi  # facing left
            ranges = sim.cast_lidar()
            # Front ray (index 0) should hit left wall at ~0.5m
            assert ranges[0] < 1.0
        finally:
            sim.robot_x, sim.robot_y, sim.robot_yaw = old_x, old_y, old_yaw


class TestPhysics:
    """Test physics step integration."""

    def test_physics_step_moves_robot(self):
        import turtlebot_sim as sim
        old_x, old_y = sim.robot_x, sim.robot_y
        old_vx, old_wz = sim.cmd_vx, sim.cmd_wz
        try:
            sim.robot_x = 4.0
            sim.robot_y = 4.0
            sim.robot_yaw = 0.0
            with sim.state_lock:
                sim.cmd_vx = 0.2
                sim.cmd_wz = 0.0
            sim.physics_step()
            # Robot should have moved to the right
            assert sim.robot_x > 4.0
        finally:
            sim.robot_x, sim.robot_y = old_x, old_y
            with sim.state_lock:
                sim.cmd_vx, sim.cmd_wz = old_vx, old_wz

    def test_physics_step_wall_collision(self):
        import turtlebot_sim as sim
        old_x, old_y = sim.robot_x, sim.robot_y
        old_vx, old_wz = sim.cmd_vx, sim.cmd_wz
        try:
            sim.robot_x = sim.ROBOT_RADIUS + 0.01
            sim.robot_y = 4.0
            sim.robot_yaw = math.pi  # facing left wall
            with sim.state_lock:
                sim.cmd_vx = 0.2
                sim.cmd_wz = 0.0
            sim.physics_step()
            # Should be clamped to at least ROBOT_RADIUS
            assert sim.robot_x >= sim.ROBOT_RADIUS
        finally:
            sim.robot_x, sim.robot_y = old_x, old_y
            with sim.state_lock:
                sim.cmd_vx, sim.cmd_wz = old_vx, old_wz


# ═══════════════════════════════════════════════════════════════════════
# 2. Navigation engine tests
# ═══════════════════════════════════════════════════════════════════════

class TestTurtleBotNavigator:
    """Test the LIDAR-aware navigation engine."""

    def _make_state(self, x=1.0, y=1.0, yaw=0.0,
                    front=3.5, front_left=3.5, front_right=3.5,
                    left=3.5, right=3.5,
                    waypoints=None):
        if waypoints is None:
            waypoints = [
                {"x": 3.0, "y": 1.0, "reached": False},
                {"x": 5.0, "y": 3.0, "reached": False},
            ]
        return {
            "x": x, "y": y, "yaw": yaw, "yaw_deg": math.degrees(yaw),
            "lidar_sectors": {
                "front": front, "front_left": front_left,
                "left": left, "rear_left": 3.5,
                "rear": 3.5, "rear_right": 3.5,
                "right": right, "front_right": front_right,
            },
            "lidar_min": min(front, front_left, front_right, left, right),
            "waypoints": waypoints,
            "waypoints_reached": sum(1 for w in waypoints if w["reached"]),
            "waypoints_total": len(waypoints),
            "collision_count": 0,
            "total_distance_m": 0.0,
            "elapsed_s": 0.0,
        }

    def test_navigate_toward_waypoint(self):
        from turtlebot_engine import TurtleBotNavigator
        nav = TurtleBotNavigator()
        state = self._make_state(x=1.0, y=1.0, yaw=0.0)
        cmd = nav.decide_action(state)
        # Should drive forward (waypoint is at x=3, y=1 — straight ahead)
        assert cmd["vx"] > 0
        assert cmd["action"] in ("FORWARD", "LEFT", "RIGHT")

    def test_waypoint_reached(self):
        from turtlebot_engine import TurtleBotNavigator
        nav = TurtleBotNavigator()
        # Place robot right at the first waypoint
        state = self._make_state(x=3.0, y=1.0, yaw=0.0)
        cmd = nav.decide_action(state)
        assert cmd["action"] == "REACHED"
        assert cmd["waypoint_reached"] == 0

    def test_all_waypoints_reached(self):
        from turtlebot_engine import TurtleBotNavigator
        nav = TurtleBotNavigator()
        waypoints = [
            {"x": 3.0, "y": 1.0, "reached": True},
            {"x": 5.0, "y": 3.0, "reached": True},
        ]
        state = self._make_state(waypoints=waypoints)
        cmd = nav.decide_action(state)
        assert cmd["action"] == "STOP"
        assert cmd["vx"] == 0.0

    def test_obstacle_avoidance_emergency_stop(self):
        from turtlebot_engine import TurtleBotNavigator, OBSTACLE_STOP_DIST
        nav = TurtleBotNavigator()
        # Obstacle very close in front
        state = self._make_state(front=0.15, front_left=0.2, front_right=0.2)
        cmd = nav.decide_action(state)
        assert cmd["action"] == "AVOID_STOP"
        assert cmd["vx"] == 0.0
        assert cmd["wz"] != 0.0  # should be turning

    def test_obstacle_avoidance_steering(self):
        from turtlebot_engine import TurtleBotNavigator
        nav = TurtleBotNavigator()
        # Obstacle in front-left, more room on right
        state = self._make_state(front=0.5, front_left=0.3, front_right=0.8, right=3.5, left=0.5)
        cmd = nav.decide_action(state)
        assert cmd["action"] == "AVOIDING"
        # Should steer right (negative wz) because right has more clearance
        assert cmd["wz"] < 0

    def test_turn_toward_waypoint(self):
        from turtlebot_engine import TurtleBotNavigator
        nav = TurtleBotNavigator()
        # Waypoint is to the left (bearing ~pi/2), robot facing right (yaw=0)
        waypoints = [{"x": 1.0, "y": 5.0, "reached": False}]
        state = self._make_state(x=1.0, y=1.0, yaw=0.0, waypoints=waypoints)
        cmd = nav.decide_action(state)
        # Should turn left (positive wz)
        assert cmd["wz"] > 0

    def test_state_summary(self):
        from turtlebot_engine import TurtleBotNavigator, state_summary
        nav = TurtleBotNavigator()
        state = self._make_state()
        summary = state_summary(state, nav)
        assert "Pose:" in summary
        assert "Waypoints:" in summary
        assert "Navigating" in summary

    def test_state_summary_mission_complete(self):
        from turtlebot_engine import TurtleBotNavigator, state_summary
        nav = TurtleBotNavigator()
        waypoints = [{"x": 3.0, "y": 1.0, "reached": True}]
        state = self._make_state(waypoints=waypoints)
        state["waypoints_reached"] = 1
        state["waypoints_total"] = 1
        summary = state_summary(state, nav)
        assert "Mission complete" in summary


# ═══════════════════════════════════════════════════════════════════════
# 3. Flask REST API tests
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def client():
    """Create Flask test client."""
    from turtlebot_sim import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestFlaskAPI:
    """Test REST API endpoints using Flask test client."""

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_get_pose(self, client):
        resp = client.get("/pose")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "x" in data
        assert "y" in data
        assert "yaw" in data

    def test_get_scan(self, client):
        resp = client.get("/scan")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["ranges"]) == 360
        assert data["num_rays"] == 360

    def test_get_state(self, client):
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "lidar_sectors" in data
        assert "waypoints" in data
        assert "waypoints_reached" in data
        assert len(data["lidar_sectors"]) == 8

    def test_drive(self, client):
        resp = client.post("/drive", json={"vx": 0.1, "wz": 0.5})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert abs(data["vx"] - 0.1) < 0.01

    def test_drive_clamps_velocity(self, client):
        resp = client.post("/drive", json={"vx": 10.0, "wz": 10.0})
        data = resp.get_json()
        assert data["vx"] <= 0.22  # MAX_VX
        assert data["wz"] <= 2.84  # MAX_WZ

    def test_stop(self, client):
        client.post("/drive", json={"vx": 0.2, "wz": 0.0})
        resp = client.post("/stop")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_reset(self, client):
        import turtlebot_sim as sim
        resp = client.post("/reset")
        assert resp.status_code == 200
        assert sim.robot_x == 1.0
        assert sim.robot_y == 1.0

    def test_get_waypoints(self, client):
        resp = client.get("/waypoints")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["waypoints"]) == 5  # DEFAULT_WAYPOINTS

    def test_reach_waypoint(self, client):
        # Reset first
        client.post("/reset")
        resp = client.post("/waypoints/reach", json={"index": 0})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        # Verify it's marked
        state = client.get("/state").get_json()
        assert state["waypoints"][0]["reached"] is True

    def test_reach_waypoint_invalid_index(self, client):
        resp = client.post("/waypoints/reach", json={"index": 999})
        assert resp.status_code == 400

    def test_pause_resume(self, client):
        resp = client.post("/pause")
        assert resp.get_json()["paused"] is True
        resp = client.post("/resume")
        assert resp.get_json()["paused"] is False

    def test_drive_with_duration(self, client):
        resp = client.post("/drive", json={"vx": 0.1, "wz": 0.0, "duration": 1.0})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
