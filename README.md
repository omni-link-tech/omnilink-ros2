# OmniLink ROS2 Demo — TurtleBot3 Navigation

Lightweight 2D simulator that models a **TurtleBot3 Burger** with 360° LIDAR, differential-drive kinematics, and static obstacles. Integrates with OmniLink via the ToolRunner pattern for AI-driven waypoint navigation with obstacle avoidance.

---

## Features

- **2D physics** at 60 Hz with differential-drive kinematics (TurtleBot3 Burger specs)
- **360° LIDAR simulation** with raycasting against walls and obstacles
- **VFH-lite obstacle avoidance** — steers away from LIDAR-detected obstacles
- **Pure-pursuit waypoint navigation** with automatic waypoint marking
- **REST API** (Flask) for commanding the robot and reading state
- **Pygame GUI** with real-time LIDAR visualisation, trail, HUD, and waypoint markers
- **OmniLink ToolRunner** integration — 1 API call kickoff, local AI controls continuously
- **Headless mode** for CI/server environments

---

## Architecture

```
OmniLink Cloud (1 API call to kick off)
           │
           ▼
[TurtleBotRunner] (play_turtlebot.py)
  ├─ Polls /state every 50ms
  ├─ TurtleBotNavigator computes vx, wz
  ├─ POSTs /drive with velocity command
  ├─ POSTs /waypoints/reach when waypoint reached
  ├─ Periodic memory saves + agent reviews
  └─ Reports mission complete when all waypoints done
           │
           ▼
[Flask REST API] (turtlebot_sim.py)
           │
           ▼
[Pygame 2D Simulator]
  ├─ Differential-drive kinematics
  ├─ LIDAR raycasting (360 rays)
  ├─ Collision detection
  └─ Real-time rendering
```

---

## Project layout

```
omnilink-ros2/
├── README.md               — this file
├── requirements.txt        — Python dependencies
├── turtlebot_sim.py        — 2D simulator + Flask REST server
├── ros2_link/
│   ├── turtlebot_api.py    — HTTP client wrapper
│   ├── turtlebot_engine.py — LIDAR-aware navigation AI
│   └── play_turtlebot.py   — OmniLink ToolRunner integration
└── tests/
    └── test_ros2_demo.py   — unit tests
```

---

## Quick start

### 1. Install dependencies

```bash
cd omnilink-benchmarks/omnilink-ros2
pip install -r requirements.txt
```

### 2. Run the simulator

```bash
python -u turtlebot_sim.py
```

The Pygame window opens showing the arena with obstacles and waypoints. The REST API starts on `http://127.0.0.1:5000`.

### 3. Run the OmniLink agent

In a second terminal:

```bash
cd ros2_link
OMNI_KEY=your-key python -u play_turtlebot.py
```

The agent kicks off with 1 API call, then the local navigator takes over — steering through waypoints while avoiding obstacles.

### 4. Headless mode (no GUI)

```bash
python -u turtlebot_sim.py --headless
```

---

## REST API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health check |
| GET | `/pose` | Robot pose: x, y, yaw |
| GET | `/scan` | Full 360° LIDAR scan (360 floats) |
| GET | `/state` | Full state: pose + LIDAR sectors + waypoints + stats |
| POST | `/drive` | Velocity command: `{vx, wz, duration?}` |
| POST | `/stop` | Emergency stop |
| POST | `/reset` | Reset robot to start |
| GET | `/waypoints` | Waypoint list with reached status |
| POST | `/waypoints/reach` | Mark waypoint reached: `{index}` |
| POST | `/pause` | Pause simulation |
| POST | `/resume` | Resume simulation |

---

## ROS2 Mapping

This demo simulates the core ROS2 TurtleBot3 topics as REST endpoints:

| ROS2 Topic | Type | REST Equivalent |
|------------|------|-----------------|
| `/odom` | nav_msgs/Odometry | `GET /pose` |
| `/scan` | sensor_msgs/LaserScan | `GET /scan` |
| `/cmd_vel` | geometry_msgs/Twist | `POST /drive` |
| `/waypoints` | — | `GET /waypoints` |

This makes it straightforward to swap the Flask REST layer for actual ROS2 publishers/subscribers when deploying on a real TurtleBot3.

---

## Keyboard shortcuts (GUI mode)

| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| Escape | Quit |
