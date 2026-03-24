#!/usr/bin/env python3
"""TurtleBot3 2D Simulator with LIDAR — OmniLink ROS2 Demo.

A lightweight 2D simulator that models a TurtleBot3 Burger with:
  - Differential-drive kinematics
  - 360° LIDAR scanner (simulated raycasting)
  - Static obstacles (walls, boxes)
  - Waypoint markers for navigation missions

Exposes a Flask REST API compatible with the OmniLink ToolRunner pattern.

Usage
-----
    python -u turtlebot_sim.py              # GUI mode (default)
    python -u turtlebot_sim.py --headless   # headless mode (no Pygame window)
"""

from __future__ import annotations

import argparse
import math
import random
import threading
import time
from typing import Any

from flask import Flask, jsonify, request

# ────────────────────────────── Tunables ──────────────────────────────
SIM_HZ = 60                    # physics tick rate
DT = 1.0 / SIM_HZ
ROBOT_RADIUS = 0.105           # TurtleBot3 Burger radius (m)
WHEEL_BASE = 0.160             # wheel separation (m)
MAX_VX = 0.22                  # max linear velocity (m/s) — TurtleBot3 spec
MAX_WZ = 2.84                  # max angular velocity (rad/s)
LIDAR_RANGE = 3.5              # max LIDAR range (m) — LDS-01 spec
LIDAR_RAYS = 360               # one ray per degree
ARENA_W = 8.0                  # arena width (m)
ARENA_H = 8.0                  # arena height (m)

# Pygame rendering
PX_PER_M = 80                  # pixels per metre
WIN_W = int(ARENA_W * PX_PER_M)
WIN_H = int(ARENA_H * PX_PER_M)

# Colours (RGB)
COL_BG = (30, 30, 30)
COL_GRID = (45, 45, 45)
COL_WALL = (100, 100, 100)
COL_OBSTACLE = (180, 80, 50)
COL_ROBOT = (50, 180, 80)
COL_HEADING = (220, 220, 220)
COL_LIDAR = (0, 200, 255, 60)
COL_WAYPOINT = (255, 200, 50)
COL_WP_REACHED = (80, 80, 80)
COL_TRAIL = (50, 130, 60)
COL_TEXT = (200, 200, 200)

# ────────────────────────── Default Mission ───────────────────────────
DEFAULT_WAYPOINTS: list[tuple[float, float]] = [
    (2.0, 2.0),
    (6.0, 2.0),
    (6.0, 6.0),
    (2.0, 6.0),
    (4.0, 4.0),
]

DEFAULT_OBSTACLES: list[dict[str, Any]] = [
    {"x": 3.0, "y": 3.5, "w": 0.6, "h": 0.6},
    {"x": 5.0, "y": 4.5, "w": 0.8, "h": 0.4},
    {"x": 4.0, "y": 1.5, "w": 0.4, "h": 1.0},
    {"x": 1.5, "y": 5.0, "w": 0.5, "h": 0.5},
    {"x": 6.5, "y": 3.0, "w": 0.4, "h": 0.8},
]

# ─────────────────────────── Shared State ─────────────────────────────
state_lock = threading.Lock()
robot_x = 1.0
robot_y = 1.0
robot_yaw = 0.0               # radians
cmd_vx = 0.0
cmd_wz = 0.0
cmd_until: float | None = None
lidar_ranges: list[float] = [LIDAR_RANGE] * LIDAR_RAYS
trail: list[tuple[float, float]] = []

waypoints = list(DEFAULT_WAYPOINTS)
waypoint_status: list[bool] = [False] * len(DEFAULT_WAYPOINTS)
obstacles = list(DEFAULT_OBSTACLES)

sim_running = True
sim_paused = False
collision_count = 0
total_distance = 0.0
start_time = time.time()


# ───────────────────────── Geometry Helpers ────────────────────────────
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _rect_contains(ox: float, oy: float, ow: float, oh: float,
                   px: float, py: float, margin: float = 0.0) -> bool:
    return (ox - margin <= px <= ox + ow + margin and
            oy - margin <= py <= oy + oh + margin)


def _ray_rect_intersect(rx: float, ry: float, dx: float, dy: float,
                        ox: float, oy: float, ow: float, oh: float) -> float | None:
    """Ray-AABB intersection. Returns distance or None."""
    tmin = 0.0
    tmax = LIDAR_RANGE

    for axis in range(2):
        origin = rx if axis == 0 else ry
        direction = dx if axis == 0 else dy
        box_min = ox if axis == 0 else oy
        box_max = (ox + ow) if axis == 0 else (oy + oh)

        if abs(direction) < 1e-9:
            if origin < box_min or origin > box_max:
                return None
        else:
            t1 = (box_min - origin) / direction
            t2 = (box_max - origin) / direction
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin if tmin <= LIDAR_RANGE else None


def _point_in_obstacle(px: float, py: float, margin: float = 0.0) -> bool:
    for obs in obstacles:
        if _rect_contains(obs["x"], obs["y"], obs["w"], obs["h"], px, py, margin):
            return True
    return False


# ────────────────────────── LIDAR Simulation ──────────────────────────
def cast_lidar() -> list[float]:
    """Cast LIDAR_RAYS rays from robot position, return distances."""
    ranges = []
    for i in range(LIDAR_RAYS):
        angle = robot_yaw + math.radians(i)
        dx = math.cos(angle)
        dy = math.sin(angle)
        min_dist = LIDAR_RANGE

        # Check arena walls
        for wall_check in [
            (0.0 - robot_x, dx),     # left wall
            (ARENA_W - robot_x, dx), # right wall
            (0.0 - robot_y, dy),     # bottom wall
            (ARENA_H - robot_y, dy), # top wall
        ]:
            diff, direction = wall_check
            if abs(direction) > 1e-9:
                t = diff / direction
                if 0 < t < min_dist:
                    min_dist = t

        # Check obstacles
        for obs in obstacles:
            t = _ray_rect_intersect(robot_x, robot_y, dx, dy,
                                    obs["x"], obs["y"], obs["w"], obs["h"])
            if t is not None and t < min_dist:
                min_dist = t

        ranges.append(round(min_dist, 3))
    return ranges


# ──────────────────────── Physics Step ────────────────────────────────
def physics_step() -> None:
    global robot_x, robot_y, robot_yaw, cmd_vx, cmd_wz, cmd_until
    global collision_count, total_distance

    with state_lock:
        # Auto-expire timed commands
        if cmd_until is not None and time.time() >= cmd_until:
            cmd_vx = 0.0
            cmd_wz = 0.0
            cmd_until = None

        vx = _clamp(cmd_vx, -MAX_VX, MAX_VX)
        wz = _clamp(cmd_wz, -MAX_WZ, MAX_WZ)

    # Integrate kinematics
    old_x, old_y = robot_x, robot_y
    robot_yaw += wz * DT
    # Normalise to [-pi, pi]
    robot_yaw = math.atan2(math.sin(robot_yaw), math.cos(robot_yaw))
    new_x = robot_x + vx * math.cos(robot_yaw) * DT
    new_y = robot_y + vx * math.sin(robot_yaw) * DT

    # Collision check
    clamped_x = _clamp(new_x, ROBOT_RADIUS, ARENA_W - ROBOT_RADIUS)
    clamped_y = _clamp(new_y, ROBOT_RADIUS, ARENA_H - ROBOT_RADIUS)

    if _point_in_obstacle(clamped_x, clamped_y, ROBOT_RADIUS):
        # Reject move — stay in place
        collision_count += 1
    else:
        robot_x = clamped_x
        robot_y = clamped_y

    dist = math.hypot(robot_x - old_x, robot_y - old_y)
    total_distance += dist

    # Update trail (subsample)
    if not trail or math.hypot(robot_x - trail[-1][0], robot_y - trail[-1][1]) > 0.05:
        trail.append((robot_x, robot_y))
        if len(trail) > 2000:
            trail.pop(0)


# ────────────────────── Pygame Rendering ──────────────────────────────
def _m2px(mx: float, my: float) -> tuple[int, int]:
    return int(mx * PX_PER_M), int((ARENA_H - my) * PX_PER_M)


def render(screen: Any, font: Any) -> None:
    import pygame

    screen.fill(COL_BG)

    # Grid
    for gx in range(int(ARENA_W) + 1):
        px = gx * PX_PER_M
        pygame.draw.line(screen, COL_GRID, (px, 0), (px, WIN_H))
    for gy in range(int(ARENA_H) + 1):
        py = gy * PX_PER_M
        pygame.draw.line(screen, COL_GRID, (0, py), (WIN_W, py))

    # Obstacles
    for obs in obstacles:
        rect = pygame.Rect(
            int(obs["x"] * PX_PER_M),
            int((ARENA_H - obs["y"] - obs["h"]) * PX_PER_M),
            int(obs["w"] * PX_PER_M),
            int(obs["h"] * PX_PER_M),
        )
        pygame.draw.rect(screen, COL_OBSTACLE, rect)
        pygame.draw.rect(screen, (220, 120, 80), rect, 2)

    # Trail
    if len(trail) > 1:
        pts = [_m2px(t[0], t[1]) for t in trail]
        pygame.draw.lines(screen, COL_TRAIL, False, pts, 2)

    # Waypoints
    for i, (wx, wy) in enumerate(waypoints):
        px, py = _m2px(wx, wy)
        col = COL_WP_REACHED if waypoint_status[i] else COL_WAYPOINT
        pygame.draw.circle(screen, col, (px, py), 10, 2)
        label = font.render(str(i + 1), True, col)
        screen.blit(label, (px - label.get_width() // 2, py - label.get_height() // 2))

    # LIDAR visualisation (every 10th ray)
    lidar_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    rpx, rpy = _m2px(robot_x, robot_y)
    for i in range(0, LIDAR_RAYS, 10):
        angle = robot_yaw + math.radians(i)
        dist = lidar_ranges[i]
        ex = robot_x + dist * math.cos(angle)
        ey = robot_y + dist * math.sin(angle)
        epx, epy = _m2px(ex, ey)
        pygame.draw.line(lidar_surf, COL_LIDAR, (rpx, rpy), (epx, epy), 1)
    screen.blit(lidar_surf, (0, 0))

    # Robot
    pygame.draw.circle(screen, COL_ROBOT, (rpx, rpy), int(ROBOT_RADIUS * PX_PER_M))
    hx = rpx + int(ROBOT_RADIUS * PX_PER_M * 1.5 * math.cos(-robot_yaw))
    hy = rpy + int(ROBOT_RADIUS * PX_PER_M * 1.5 * math.sin(-robot_yaw))
    pygame.draw.line(screen, COL_HEADING, (rpx, rpy), (hx, hy), 3)

    # HUD
    elapsed = time.time() - start_time
    reached = sum(waypoint_status)
    hud_lines = [
        f"TurtleBot3 — OmniLink ROS2 Demo",
        f"Pose: ({robot_x:.2f}, {robot_y:.2f}) yaw={math.degrees(robot_yaw):.1f}°",
        f"Vel: vx={cmd_vx:.2f} m/s  wz={cmd_wz:.2f} rad/s",
        f"Waypoints: {reached}/{len(waypoints)}  Collisions: {collision_count}",
        f"Distance: {total_distance:.1f}m  Time: {elapsed:.0f}s",
        f"{'PAUSED' if sim_paused else 'RUNNING'}",
    ]
    for i, line in enumerate(hud_lines):
        surf = font.render(line, True, COL_TEXT)
        screen.blit(surf, (10, 10 + i * 20))


# ────────────────────────── Flask API ─────────────────────────────────
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "simulator": "turtlebot3-ros2-demo"})


@app.route("/pose", methods=["GET"])
def get_pose():
    return jsonify({
        "x": round(robot_x, 4),
        "y": round(robot_y, 4),
        "yaw": round(robot_yaw, 4),
        "yaw_deg": round(math.degrees(robot_yaw), 2),
    })


@app.route("/scan", methods=["GET"])
def get_scan():
    """Return the latest LIDAR scan (360 range values)."""
    return jsonify({
        "ranges": lidar_ranges,
        "range_min": 0.12,
        "range_max": LIDAR_RANGE,
        "angle_min": 0.0,
        "angle_max": 2 * math.pi,
        "num_rays": LIDAR_RAYS,
    })


@app.route("/state", methods=["GET"])
def get_state():
    """Full state snapshot — pose, scan summary, waypoints, stats."""
    elapsed = time.time() - start_time
    reached = sum(waypoint_status)

    # Summarise LIDAR into 8 sectors (front, front-left, left, ... )
    sector_names = ["front", "front_left", "left", "rear_left",
                    "rear", "rear_right", "right", "front_right"]
    sector_size = LIDAR_RAYS // 8
    sectors = {}
    for i, name in enumerate(sector_names):
        start_idx = i * sector_size
        end_idx = start_idx + sector_size
        sector_ranges = lidar_ranges[start_idx:end_idx]
        sectors[name] = round(min(sector_ranges), 3) if sector_ranges else LIDAR_RANGE

    return jsonify({
        "x": round(robot_x, 4),
        "y": round(robot_y, 4),
        "yaw": round(robot_yaw, 4),
        "yaw_deg": round(math.degrees(robot_yaw), 2),
        "cmd_vx": round(cmd_vx, 3),
        "cmd_wz": round(cmd_wz, 3),
        "lidar_sectors": sectors,
        "lidar_min": round(min(lidar_ranges), 3),
        "waypoints": [
            {"x": wx, "y": wy, "reached": waypoint_status[i]}
            for i, (wx, wy) in enumerate(waypoints)
        ],
        "waypoints_reached": reached,
        "waypoints_total": len(waypoints),
        "collision_count": collision_count,
        "total_distance_m": round(total_distance, 2),
        "elapsed_s": round(elapsed, 1),
        "paused": sim_paused,
        "mission_complete": reached == len(waypoints),
    })


@app.route("/drive", methods=["POST"])
def drive():
    """Send velocity command: {vx, wz, duration?}."""
    global cmd_vx, cmd_wz, cmd_until
    data = request.get_json(force=True, silent=True) or {}
    with state_lock:
        cmd_vx = _clamp(float(data.get("vx", 0.0)), -MAX_VX, MAX_VX)
        cmd_wz = _clamp(float(data.get("wz", 0.0)), -MAX_WZ, MAX_WZ)
        duration = data.get("duration")
        cmd_until = (time.time() + float(duration)) if duration else None
    return jsonify({"ok": True, "vx": cmd_vx, "wz": cmd_wz})


@app.route("/stop", methods=["POST"])
def stop():
    global cmd_vx, cmd_wz, cmd_until
    with state_lock:
        cmd_vx = 0.0
        cmd_wz = 0.0
        cmd_until = None
    return jsonify({"ok": True})


@app.route("/reset", methods=["POST"])
def reset_robot():
    global robot_x, robot_y, robot_yaw, cmd_vx, cmd_wz, cmd_until
    global collision_count, total_distance, start_time, trail, waypoint_status
    with state_lock:
        robot_x = 1.0
        robot_y = 1.0
        robot_yaw = 0.0
        cmd_vx = 0.0
        cmd_wz = 0.0
        cmd_until = None
    collision_count = 0
    total_distance = 0.0
    start_time = time.time()
    trail.clear()
    waypoint_status = [False] * len(waypoints)
    return jsonify({"ok": True})


@app.route("/waypoints", methods=["GET"])
def get_waypoints():
    return jsonify({
        "waypoints": [
            {"x": wx, "y": wy, "reached": waypoint_status[i]}
            for i, (wx, wy) in enumerate(waypoints)
        ]
    })


@app.route("/waypoints/reach", methods=["POST"])
def reach_waypoint():
    """Mark a waypoint as reached: {index: int}."""
    data = request.get_json(force=True, silent=True) or {}
    idx = int(data.get("index", -1))
    if 0 <= idx < len(waypoints):
        waypoint_status[idx] = True
        return jsonify({"ok": True, "index": idx})
    return jsonify({"ok": False, "error": "invalid index"}), 400


@app.route("/pause", methods=["POST"])
def pause():
    global sim_paused
    sim_paused = True
    return jsonify({"ok": True, "paused": True})


@app.route("/resume", methods=["POST"])
def resume():
    global sim_paused
    sim_paused = False
    return jsonify({"ok": True, "paused": False})


# ──────────────────────── Main Loops ──────────────────────────────────
def simulation_loop() -> None:
    """Physics + LIDAR at SIM_HZ."""
    global lidar_ranges
    while sim_running:
        if not sim_paused:
            physics_step()
            lidar_ranges = cast_lidar()
        time.sleep(DT)


def run_gui() -> None:
    """Pygame render loop (blocks on main thread)."""
    global sim_running
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("TurtleBot3 — OmniLink ROS2 Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    while sim_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sim_running = False
                elif event.key == pygame.K_SPACE:
                    global sim_paused
                    sim_paused = not sim_paused

        render(screen, font)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main() -> None:
    global sim_running
    parser = argparse.ArgumentParser(description="TurtleBot3 2D Simulator")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    args = parser.parse_args()

    # Start physics in background
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()

    # Start Flask in background
    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=args.port, threaded=True),
        daemon=True,
    )
    flask_thread.start()
    print(f"[TurtleBot3 Sim] REST API on http://127.0.0.1:{args.port}")
    print(f"[TurtleBot3 Sim] Endpoints: /state /pose /scan /drive /stop /reset /waypoints")

    if args.headless:
        print("[TurtleBot3 Sim] Running headless — press Ctrl+C to stop")
        try:
            while sim_running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            sim_running = False
    else:
        run_gui()

    sim_running = False
    print("[TurtleBot3 Sim] Shutdown.")


if __name__ == "__main__":
    main()
