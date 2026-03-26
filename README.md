# AI-Based Real-Time Network Fault Detection System

A production-quality, full-stack distributed monitoring platform that detects network faults in real time using rule-based logic and machine learning. Multiple Python client nodes stream live metrics to a central FastAPI backend over WebSockets. Results are displayed on a React dashboard with live charts, alerts, and AI predictions.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Fault Types Detected](#3-fault-types-detected)
4. [Project Structure](#4-project-structure)
5. [Component Deep-Dive](#5-component-deep-dive)
6. [Prerequisites](#6-prerequisites)
7. [Step-by-Step Setup and Run Guide](#7-step-by-step-setup-and-run-guide)
   - [Step 1 – Clone and Prepare](#step-1--clone-and-prepare)
   - [Step 2 – Backend Setup](#step-2--backend-setup)
   - [Step 3 – Database Setup](#step-3--database-setup)
   - [Step 4 – ML Model Training](#step-4--ml-model-training)
   - [Step 5 – Start the Backend Server](#step-5--start-the-backend-server)
   - [Step 6 – Start Client Nodes](#step-6--start-client-nodes)
   - [Step 7 – Frontend Setup](#step-7--frontend-setup)
   - [Step 8 – Open the Dashboard](#step-8--open-the-dashboard)
8. [Running Across Multiple Machines](#8-running-across-multiple-machines)
9. [Environment Variables Reference](#9-environment-variables-reference)
10. [API Reference](#10-api-reference)
11. [WebSocket Message Protocol](#11-websocket-message-protocol)
12. [ML Model Details](#12-ml-model-details)
13. [Troubleshooting](#13-troubleshooting)
14. [Technology Stack](#14-technology-stack)

---

## 1. Project Overview

This system simulates and monitors a real multi-node network environment. Each node is a Python process running on a separate machine (or terminal). Nodes continuously measure their own network health — latency, packet loss, throughput, jitter — and stream that data to a central server via WebSocket.

The server applies two layers of fault detection:

- **Rule Engine** — threshold-based checks that fire immediately when a metric crosses a boundary (e.g., latency > 200 ms).
- **ML Engine** — a trained RandomForestClassifier that classifies the incoming metric vector into one of seven fault categories with a confidence score.

All events are persisted in SQLite (upgradeable to PostgreSQL) and broadcast in real time to any connected dashboard clients. The React dashboard renders live charts, node status cards, alert feeds, and AI prediction panels — all updating without a page refresh.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LOCAL NETWORK                               │
│                                                                     │
│  ┌──────────────┐   WebSocket    ┌─────────────────────────────┐   │
│  │  Node A      │ ─────────────► │                             │   │
│  │  (Machine 1) │                │     FastAPI Backend          │   │
│  └──────────────┘                │     server/main.py          │   │
│                                  │                             │   │
│  ┌──────────────┐   WebSocket    │  ┌──────────────────────┐  │   │
│  │  Node B      │ ─────────────► │  │  WebSocket Manager   │  │   │
│  │  (Machine 2) │                │  │  (node + dashboard)  │  │   │
│  └──────────────┘                │  └──────────┬───────────┘  │   │
│                                  │             │               │   │
│  ┌──────────────┐   WebSocket    │  ┌──────────▼───────────┐  │   │
│  │  Node C      │ ─────────────► │  │   Fault Detector     │  │   │
│  │  (Machine 3) │                │  │  Rule Engine + ML    │  │   │
│  └──────────────┘                │  └──────────┬───────────┘  │   │
│                                  │             │               │   │
│  ┌──────────────┐   HTTP/REST    │  ┌──────────▼───────────┐  │   │
│  │  React       │ ◄────────────► │  │  SQLite Database     │  │   │
│  │  Dashboard   │                │  │  (nodes/metrics/     │  │   │
│  │  (Browser)   │ ◄── WS ──────  │  │   faults/files)      │  │   │
│  └──────────────┘                │  └──────────────────────┘  │   │
│                                  └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Client Node
  │
  ├── Every 2s  → heartbeat message  ──────────────────────────────────►
  ├── Every 3s  → metrics message (latency, loss, throughput, jitter) ─►  Backend
  └── Every 30s → file upload (HTTP POST with SHA256 checksum) ─────────►
                                                                          │
                                                                          ├── Update heartbeat timer
                                                                          ├── Persist MetricLog to DB
                                                                          ├── Run Rule Engine → FaultEvents
                                                                          ├── Run ML Inference → prediction
                                                                          └── Broadcast to all Dashboard WS clients
                                                                                    │
                                                                                    ▼
                                                                          React Dashboard
                                                                          ├── Node status grid (live)
                                                                          ├── Latency / loss / throughput charts
                                                                          ├── Fault alert feed
                                                                          └── AI prediction panel
```

---

## 3. Fault Types Detected

| Fault | Trigger | Severity | Detection Method |
|---|---|---|---|
| `NODE_FAILURE` | No heartbeat for > 5 seconds | CRITICAL | Heartbeat monitor background task |
| `HIGH_LATENCY` | Latency > 200 ms | MEDIUM / CRITICAL | Rule engine + ML |
| `PACKET_LOSS` | Loss rate > 5% | MEDIUM / CRITICAL | Rule engine + ML |
| `THROUGHPUT_DEGRADATION` | Throughput < 1 Mbps | MEDIUM | Rule engine + ML |
| `NETWORK_CONGESTION` | 5 consecutive degraded samples | CRITICAL | Sliding window rule + ML |
| `DATA_CORRUPTION` | SHA256 checksum mismatch | CRITICAL | File transfer verification |
| `NORMAL` | All metrics in healthy range | LOW | ML classifier output |

---

## 4. Project Structure

```
iot-system/
│
├── server/                          # FastAPI backend
│   ├── main.py                      # App entrypoint, WS handlers, background tasks
│   ├── config.py                    # Settings and thresholds
│   ├── websocket_manager.py         # Connection registry (nodes + dashboards)
│   ├── fault_detector.py            # Rule-based fault engine
│   ├── ml_inference.py              # ML model loader and predictor
│   ├── requirements.txt             # Python dependencies
│   ├── routers/
│   │   ├── nodes.py                 # REST: /api/nodes
│   │   ├── metrics.py               # REST: /api/metrics
│   │   ├── alerts.py                # REST: /api/alerts
│   │   └── files.py                 # REST: /api/files/upload
│   └── uploads/                     # Received files (auto-created)
│
├── database/                        # SQLAlchemy ORM layer
│   ├── db.py                        # Async engine, session factory, init_db()
│   └── models.py                    # Node, MetricLog, FaultEvent, FileTransfer
│
├── client/                          # Python node agent
│   ├── node_client.py               # Main async client loop
│   ├── metrics_collector.py         # Latency, packet loss, throughput, jitter
│   └── requirements.txt             # Client dependencies
│
├── ml_model/                        # Machine learning pipeline
│   ├── train.py                     # Dataset generation + model training
│   ├── inference.py                 # Standalone inference utility
│   ├── models/                      # Saved model artifacts (auto-created)
│   │   ├── fault_classifier.pkl     # Trained RandomForestClassifier
│   │   └── scaler.pkl               # StandardScaler
│   └── requirements.txt
│
├── frontend/                        # React + Tailwind dashboard
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── hooks/useWebSocket.js    # WS connection hook
│   │   ├── components/
│   │   │   ├── NodeGrid.jsx         # Live node status cards
│   │   │   ├── MetricCharts.jsx     # Recharts line graphs
│   │   │   ├── AlertFeed.jsx        # Fault alert list
│   │   │   ├── LogPanel.jsx         # Raw event log
│   │   │   └── MLPanel.jsx          # AI prediction display
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
│
└── README.md                        # This file
```

---

## 5. Component Deep-Dive

### Backend (`server/`)

**`main.py`** is the FastAPI application. It exposes:
- `GET /` — health overview
- `GET /health` — liveness check with ML status
- `POST /api/ml/reload` — hot-reload ML model after retraining
- `WS /ws/node/{node_id}` — node data ingestion endpoint
- `WS /ws/dashboard` — dashboard subscription endpoint
- REST routers at `/api/nodes`, `/api/metrics`, `/api/alerts`, `/api/files`

A background `asyncio` task runs every second and checks whether each connected node has sent a heartbeat within the timeout window. If not, a `NODE_FAILURE` fault is generated automatically.

**`websocket_manager.py`** maintains two registries:
- `node_connections` — one WebSocket per node ID
- `dashboard_connections` — a list of all connected browser dashboard clients

Every message received from a node is processed and then fanned out via `broadcast_to_dashboards()` to every connected dashboard client simultaneously.

**`fault_detector.py`** implements the `RuleBasedFaultDetector` class. It maintains a per-node sliding window of the last 5 metric samples to detect sustained degradation (congestion). Each call to `.detect()` returns a list of `FaultEvent` objects, which are then persisted and broadcast.

**`ml_inference.py`** wraps the trained `RandomForestClassifier`. It accepts raw metric dicts, extracts the four feature values, optionally scales them, and returns the predicted fault type plus per-class probabilities. If the model file does not exist yet (before training), it silently skips inference and sets `ml_available: false` in the response payload.

---

### Database (`database/`)

Uses **SQLAlchemy 2.0 async** with **aiosqlite** driver (SQLite by default, swap `DATABASE_URL` to PostgreSQL with `asyncpg` for production).

Four tables:
- `nodes` — registered node metadata (hostname, IP, OS, last_seen)
- `metric_logs` — every metric sample from every node, timestamped
- `fault_events` — every detected fault with rule + ML attribution
- `file_transfers` — file upload records with checksum comparison results

---

### Client Node (`client/`)

Each client runs an `asyncio` event loop with three concurrent tasks:
1. **Heartbeat loop** — sends `{"type": "heartbeat"}` every 2 seconds
2. **Metrics loop** — collects real latency (socket timing to server), estimates packet loss from failed probes, measures throughput via timed data transfer, and computes jitter from latency variance. Sends all as `{"type": "metrics", "payload": {...}}` every 3 seconds
3. **File upload loop** — every 30 seconds, generates or selects a file, computes its SHA256 locally, and POSTs it to `/api/files/upload`. The server re-checksums and flags corruption

---

### ML Model (`ml_model/`)

**Training** (`train.py`) generates a labeled dataset by combining:
- Real metric logs exported from the running server's SQLite database (if available)
- Synthetic samples generated with controlled noise per fault class

It trains a `RandomForestClassifier` with 200 estimators, applies a `StandardScaler`, and saves both artifacts with `joblib`. It also prints a classification report showing per-class precision, recall, and F1 score.

**Features used:**
- `latency_ms` — round-trip time to server in milliseconds
- `packet_loss_rate` — fraction of failed transmissions (0.0 to 1.0)
- `throughput_mbps` — estimated data transfer rate
- `jitter_ms` — standard deviation of recent latency samples

**Classes (7):**
`NORMAL`, `HIGH_LATENCY`, `PACKET_LOSS`, `THROUGHPUT_DEGRADATION`, `NETWORK_CONGESTION`, `DATA_CORRUPTION`, `NODE_FAILURE`

---

### Frontend (`frontend/`)

Built with **React 18 + Vite + Tailwind CSS**. Connects to `ws://SERVER_IP:8000/ws/dashboard` on load and maintains the connection with automatic reconnection.

Panels:
- **Node Grid** — one card per node, green = online, red = offline, showing last-seen timestamp
- **Metric Charts** — Recharts `LineChart` components for latency, packet loss, and throughput, plotting the last 60 data points per selected node
- **Alert Feed** — scrolling list of fault events color-coded by severity (LOW=blue, MEDIUM=yellow, CRITICAL=red)
- **Log Panel** — raw event stream, newest first
- **AI Panel** — last ML prediction with fault type, confidence bar, and full class probability breakdown

---

## 6. Prerequisites

Install these on **every machine** that will participate.

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 or 3.12 | 3.10 minimum |
| Node.js | 18 or 20 | For frontend only |
| npm | 9+ | Comes with Node.js |
| Git | Any | For cloning |

**Check your versions:**
```bash
python3 --version
node --version
npm --version
```

**Network requirement:** All machines must be on the same LAN (or reachable via IP). Firewall must allow TCP on port `8000` (backend) and port `5173` (frontend dev server) from other machines.

---

## 7. Step-by-Step Setup and Run Guide

> Run these steps in order. Complete each step fully before moving to the next.

---

### Step 1 — Clone and Prepare

Run this on the **server machine** first, then repeat on each client machine.

```bash
# Clone the repository
git clone https://github.com/your-org/iot-fault-detection.git
cd iot-fault-detection

# Verify the structure
ls -la
# You should see: server/  client/  database/  ml_model/  frontend/  README.md
```

---

### Step 2 — Backend Setup

Do this on the **server machine only**.

```bash
# Navigate to the server directory
cd server

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install all Python dependencies
pip install -r requirements.txt

# Verify FastAPI installed correctly
python -c "import fastapi; print('FastAPI OK:', fastapi.__version__)"
```

---

### Step 3 — Database Setup

Still on the **server machine**, inside the `server/` directory with the venv active.

```bash
# The database/ folder sits one level up from server/
# Make sure you are in the project root
cd ..   # go back to iot-fault-detection/

# Create the database package structure if not already present
mkdir -p database

# The database initializes automatically when the server starts (init_db() is called in lifespan).
# However you can test it manually:
cd server
python -c "
import asyncio
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from database.db import init_db
asyncio.run(init_db())
print('Database tables created successfully')
"
```

You should see:
```
Database tables created successfully
```

A file `network_monitor.db` will appear in the `server/` directory.

---

### Step 4 — ML Model Training

Still on the **server machine**. The model must be trained before the server can use ML-based predictions. The server works without it (rule-based detection still runs), but training it first gives full functionality.

```bash
# From project root
cd ml_model

# Create virtual environment for ML (can reuse server venv if preferred)
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install ML dependencies
pip install -r requirements.txt

# Run training
python train.py
```

Expected output:
```
Generating synthetic training dataset...
Dataset size: 7000 samples
Class distribution:
  NORMAL                  1000
  HIGH_LATENCY            1000
  PACKET_LOSS             1000
  THROUGHPUT_DEGRADATION  1000
  NETWORK_CONGESTION      1000
  DATA_CORRUPTION         1000
  NODE_FAILURE            1000

Training RandomForestClassifier (200 estimators)...

Classification Report:
                         precision  recall  f1-score  support
NORMAL                      0.99    0.99      0.99      200
HIGH_LATENCY                0.98    0.97      0.97      200
PACKET_LOSS                 0.97    0.98      0.97      200
THROUGHPUT_DEGRADATION      0.98    0.98      0.98      200
NETWORK_CONGESTION          0.96    0.97      0.96      200
DATA_CORRUPTION             0.99    0.99      0.99      200
NODE_FAILURE                1.00    1.00      1.00      200

Model saved to: models/fault_classifier.pkl
Scaler saved to: models/scaler.pkl
Training complete.
```

Verify the model files exist:
```bash
ls models/
# fault_classifier.pkl    scaler.pkl
```

---

### Step 5 — Start the Backend Server

On the **server machine**, in a dedicated terminal window:

```bash
# From project root
cd server

# Activate venv if not already active
source venv/bin/activate

# Start the server (binds to all interfaces so clients can reach it)
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Expected startup output:
```
INFO:     Starting AI Network Fault Detection System v1.0.0
INFO:     Database initialized
INFO:     Heartbeat monitor started
INFO:     ML model loaded from ../ml_model/models/fault_classifier.pkl
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verify the server is running:**
```bash
# In a new terminal, on the server machine:
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000000",
  "ml_model_ready": true,
  "nodes_online": 0
}
```

**Find your server's local IP address** (you will need this for client machines):
```bash
# Linux/macOS
ip addr show | grep "inet " | grep -v 127.0.0.1
# or
hostname -I

# Windows
ipconfig
```

Note this IP — it will be referred to as `SERVER_IP` throughout the rest of this guide.
Example: `192.168.1.100`

---

### Step 6 — Start Client Nodes

Do this on **each client machine** (and optionally also on the server machine in separate terminals to simulate multiple nodes).

```bash
# Navigate to the client directory
cd client

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install client dependencies
pip install -r requirements.txt
```

**Start a node**, replacing `SERVER_IP` with the actual IP you found in Step 5:

```bash
# Basic usage — node ID is auto-generated from hostname if omitted
python node_client.py --server SERVER_IP --port 8000

# With a custom node ID and explicit settings
python node_client.py \
  --server 192.168.1.100 \
  --port 8000 \
  --node-id "node-machine-1" \
  --heartbeat-interval 2 \
  --metrics-interval 3 \
  --file-interval 30
```

Expected output on the client:
```
[2024-01-15 10:31:00] Node ID: node-machine-1
[2024-01-15 10:31:00] Connecting to ws://192.168.1.100:8000/ws/node/node-machine-1 ...
[2024-01-15 10:31:00] Connected! Acknowledged by server.
[2024-01-15 10:31:02] Heartbeat sent
[2024-01-15 10:31:03] Metrics sent | latency=12.4ms loss=0.00% throughput=45.2Mbps jitter=1.1ms
[2024-01-15 10:31:04] Heartbeat sent
[2024-01-15 10:31:06] Metrics sent | latency=13.1ms loss=0.00% throughput=44.8Mbps jitter=0.9ms
[2024-01-15 10:31:30] File uploaded | size=512KB checksum=a3f92... match=True
```

**Repeat on each additional machine**, using a different `--node-id` each time:

Machine 2:
```bash
python node_client.py --server 192.168.1.100 --port 8000 --node-id "node-machine-2"
```

Machine 3:
```bash
python node_client.py --server 192.168.1.100 --port 8000 --node-id "node-machine-3"
```

Machine 4 (simulate on the server itself in a new terminal):
```bash
python node_client.py --server 127.0.0.1 --port 8000 --node-id "node-local-4"
```

**Verify nodes are connected** — on the server machine:
```bash
curl http://localhost:8000/api/nodes/
```

```json
{
  "nodes": [
    {"node_id": "node-machine-1", "status": "online", "hostname": "machine1", ...},
    {"node_id": "node-machine-2", "status": "online", "hostname": "machine2", ...},
    {"node_id": "node-machine-3", "status": "online", "hostname": "machine3", ...}
  ],
  "total": 3
}
```

---

### Step 7 — Frontend Setup

Do this on the **server machine** (or any machine on the same network with Node.js).

```bash
# From project root
cd frontend

# Install npm dependencies
npm install

# Configure the server IP
# Open frontend/src/config.js (or set an environment variable)
# Change SERVER_IP to your actual server IP:
echo "VITE_SERVER_URL=http://192.168.1.100:8000" > .env.local
echo "VITE_WS_URL=ws://192.168.1.100:8000" >> .env.local
```

**Start the development server:**
```bash
npm run dev -- --host 0.0.0.0
```

Expected output:
```
  VITE v5.x.x  ready in 312 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.1.100:5173/
  ➜  press h + enter to show help
```

**For production build** (optional, to serve statically):
```bash
npm run build
# Outputs to frontend/dist/
# Serve with: npx serve dist -p 5173
```

---

### Step 8 — Open the Dashboard

Open a browser on **any machine on the network** and navigate to:

```
http://192.168.1.100:5173
```

Replace `192.168.1.100` with your server machine's IP.

The dashboard will:
1. Establish a WebSocket connection to `ws://SERVER_IP:8000/ws/dashboard`
2. Receive the initial state of all connected nodes
3. Begin streaming live metrics, heartbeats, and fault alerts

**What you should see:**
- Green node cards for every connected client
- Latency, packet loss, and throughput charts updating every 3 seconds
- Alert feed showing any detected faults
- AI panel showing ML predictions with confidence scores

**To simulate a node failure:** Stop one of the client processes with `Ctrl+C`. Within 5 seconds the server will generate a `NODE_FAILURE` CRITICAL alert and the dashboard will turn that node's card red.

---

## 8. Running Across Multiple Machines

### Network Checklist

Before starting, verify:

```bash
# From each client machine, ping the server:
ping 192.168.1.100

# From the server machine, confirm port 8000 is open:
# Linux:
ss -tlnp | grep 8000
# macOS:
lsof -i :8000

# Test HTTP connectivity from a client machine:
curl http://192.168.1.100:8000/health

# Test WebSocket connectivity from a client machine:
# Install wscat: npm install -g wscat
wscat -c ws://192.168.1.100:8000/ws/dashboard
```

### Firewall Rules

**Linux (ufw):**
```bash
sudo ufw allow 8000/tcp
sudo ufw allow 5173/tcp
sudo ufw reload
```

**Linux (firewalld):**
```bash
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=5173/tcp
sudo firewall-cmd --reload
```

**Windows Defender Firewall:**
```powershell
New-NetFirewallRule -DisplayName "IoT Backend" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
New-NetFirewallRule -DisplayName "IoT Frontend" -Direction Inbound -Protocol TCP -LocalPort 5173 -Action Allow
```

**macOS:**
```bash
# macOS typically does not block outbound connections.
# If prompted by the firewall, allow Python and Node.js.
```

### Recommended Machine Roles

| Machine | Role | Commands to run |
|---|---|---|
| Machine 1 (server) | Backend + Frontend | `python main.py` + `npm run dev` |
| Machine 2 | Client Node | `python node_client.py --server 192.168.1.100` |
| Machine 3 | Client Node | `python node_client.py --server 192.168.1.100` |
| Machine 4 | Client Node | `python node_client.py --server 192.168.1.100` |
| Any machine | Dashboard viewer | Open browser to `http://192.168.1.100:5173` |

---

## 9. Environment Variables Reference

Create a `.env` file in the `server/` directory to override defaults.

```ini
# server/.env

# Application
DEBUG=false

# Database (default: SQLite)
DATABASE_URL=sqlite+aiosqlite:///./network_monitor.db

# PostgreSQL alternative:
# DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/network_monitor

# Fault detection thresholds
LATENCY_THRESHOLD_MS=200.0
PACKET_LOSS_THRESHOLD=0.05
THROUGHPUT_MIN_MBPS=1.0
JITTER_THRESHOLD_MS=50.0

# Heartbeat timeout (seconds before node is marked failed)
WS_HEARTBEAT_TIMEOUT=5

# ML model paths (relative to server/ directory)
ML_MODEL_PATH=../ml_model/models/fault_classifier.pkl
ML_SCALER_PATH=../ml_model/models/scaler.pkl

# File upload directory
UPLOAD_DIR=./uploads

# Security
SECRET_KEY=change-me-in-production
```

Create a `.env.local` file in the `frontend/` directory:

```ini
# frontend/.env.local
VITE_SERVER_URL=http://192.168.1.100:8000
VITE_WS_URL=ws://192.168.1.100:8000
```

---

## 10. API Reference

All REST endpoints are available at `http://SERVER_IP:8000`. Interactive docs (Swagger UI) are at `http://SERVER_IP:8000/docs`.

### Nodes

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/nodes/` | List all nodes |
| GET | `/api/nodes/{node_id}` | Get one node |
| GET | `/api/nodes/{node_id}/metrics` | Metric history |
| GET | `/api/nodes/{node_id}/faults` | Fault history |
| GET | `/api/nodes/{node_id}/stats` | Aggregated stats |

### Metrics

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/metrics/latest` | Most recent metrics (all nodes) |
| GET | `/api/metrics/summary?hours=1` | Network-wide aggregates |
| GET | `/api/metrics/timeseries/{node_id}?minutes=10` | Chart data |

### Alerts

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/alerts/` | List alerts (filterable) |
| GET | `/api/alerts/summary?hours=24` | Counts by type and severity |
| GET | `/api/alerts/recent?limit=20` | Latest 20 alerts |

### Files

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/files/upload` | Upload file with checksum |
| GET | `/api/files/transfers` | List transfer records |

### System

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Server info |
| GET | `/health` | Liveness probe |
| POST | `/api/ml/reload` | Hot-reload ML model |
| GET | `/docs` | Swagger UI |

---

## 11. WebSocket Message Protocol

### Node → Server messages

**Heartbeat** (every 2 seconds):
```json
{
  "type": "heartbeat",
  "node_id": "node-machine-1",
  "timestamp": "2024-01-15T10:31:02.000Z"
}
```

**Metrics** (every 3 seconds):
```json
{
  "type": "metrics",
  "node_id": "node-machine-1",
  "payload": {
    "latency_ms": 12.4,
    "packet_loss_rate": 0.02,
    "throughput_mbps": 45.2,
    "jitter_ms": 1.1
  },
  "timestamp": "2024-01-15T10:31:03.000Z"
}
```

### Server → Dashboard messages

**Initial state** (sent once on dashboard connection):
```json
{
  "type": "initial_state",
  "nodes": [
    {"node_id": "node-machine-1", "status": "online", "hostname": "machine1", ...}
  ],
  "timestamp": "2024-01-15T10:31:00.000Z"
}
```

**Live metrics broadcast:**
```json
{
  "type": "metrics",
  "node_id": "node-machine-1",
  "payload": {
    "latency_ms": 12.4,
    "packet_loss_rate": 0.02,
    "throughput_mbps": 45.2,
    "jitter_ms": 1.1
  },
  "ml_prediction": {
    "ml_available": true,
    "fault_type": "NORMAL",
    "severity": "LOW",
    "confidence": 0.9812,
    "class_probabilities": {
      "NORMAL": 0.9812,
      "HIGH_LATENCY": 0.0088,
      "PACKET_LOSS": 0.0052,
      ...
    }
  },
  "timestamp": "2024-01-15T10:31:03.000Z"
}
```

**Fault alert broadcast:**
```json
{
  "type": "fault_alert",
  "node_id": "node-machine-2",
  "fault_type": "HIGH_LATENCY",
  "severity": "MEDIUM",
  "description": "Latency 312.5ms exceeds threshold 200.0ms",
  "source": "rule_engine",
  "ml_prediction": {
    "ml_available": true,
    "fault_type": "HIGH_LATENCY",
    "confidence": 0.9421
  },
  "timestamp": "2024-01-15T10:32:45.000Z"
}
```

**Node status change:**
```json
{
  "type": "node_status",
  "node_id": "node-machine-3",
  "status": "offline",
  "timestamp": "2024-01-15T10:35:00.000Z"
}
```

---

## 12. ML Model Details

### Algorithm
`RandomForestClassifier` from scikit-learn with 200 decision trees.

### Feature Engineering

| Feature | Unit | Normal Range | Fault Indicator |
|---|---|---|---|
| `latency_ms` | milliseconds | 1 – 50 | > 200 |
| `packet_loss_rate` | fraction 0–1 | 0 – 0.01 | > 0.05 |
| `throughput_mbps` | Mbps | 10 – 1000 | < 1.0 |
| `jitter_ms` | milliseconds | 0 – 10 | > 50 |

### Training Data

The training script generates 1000 synthetic samples per class (7000 total) with realistic noise and boundary conditions per fault type. If real metric logs exist in the database, those are included and weighted more heavily.

### Retraining Workflow

After the system has been running and accumulating real data:

```bash
# Export real logs from the database
cd server
python -c "
import asyncio, json, sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
# export_logs() function in ml_model/train.py handles this
"

# Retrain with real + synthetic data
cd ../ml_model
python train.py --use-db-logs --db-path ../server/network_monitor.db

# Hot-reload the model in the running server (no restart needed)
curl -X POST http://localhost:8000/api/ml/reload
```

---

## 13. Troubleshooting

### Client cannot connect to server

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Fix:**
- Confirm the server is running: `curl http://SERVER_IP:8000/health`
- Confirm the port is open on the server firewall (see Section 8)
- Confirm you are using the correct IP (not `localhost` on a different machine)
- Make sure both machines are on the same network: `ping SERVER_IP`

---

### `ModuleNotFoundError: No module named 'database'`

**Fix:** The server imports from `../database/`. Always run `python main.py` from inside the `server/` directory, not from the project root.

```bash
cd server
python main.py    # correct
```

---

### ML model not found warning

```
WARNING: ML model not found at ../ml_model/models/fault_classifier.pkl
```

**Fix:** Train the model first (Step 4). The server continues to work with rule-based detection only until the model is available.

---

### `aiosqlite` not found or import errors

```bash
pip install aiosqlite sqlalchemy[asyncio]
```

---

### Port 8000 already in use

```bash
# Find what is using port 8000
lsof -i :8000      # macOS/Linux
netstat -ano | findstr :8000   # Windows

# Kill the process or use a different port
uvicorn main:app --host 0.0.0.0 --port 8080
```

Then update clients and frontend to use port `8080`.

---

### Dashboard shows no data / WebSocket not connecting

- Open browser DevTools → Console — look for WebSocket errors
- Confirm the `VITE_WS_URL` in `frontend/.env.local` matches the actual server IP and port
- Check that the React dev server is running (`npm run dev`)
- Try `wscat -c ws://SERVER_IP:8000/ws/dashboard` to test the WS endpoint manually

---

### Nodes show as offline immediately after connecting

- The heartbeat interval on the client must be less than `WS_HEARTBEAT_TIMEOUT` on the server (default 5 seconds)
- Default client heartbeat interval is 2 seconds — this is safe
- If you changed `WS_HEARTBEAT_TIMEOUT` in `.env`, ensure it is greater than the client interval

---

## 14. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend framework | FastAPI 0.111 | Async REST + WebSocket server |
| ASGI server | Uvicorn | Production-grade async server |
| Database ORM | SQLAlchemy 2.0 (async) | Database access layer |
| Database | SQLite (aiosqlite) | Development storage |
| Database (prod) | PostgreSQL (asyncpg) | Production storage |
| ML framework | scikit-learn 1.4 | RandomForestClassifier |
| ML serialization | joblib | Model persistence |
| Numerics | NumPy | Feature arrays |
| Client networking | websockets, aiohttp | Async WS + HTTP client |
| Frontend framework | React 18 | Dashboard UI |
| Frontend build | Vite 5 | Fast dev server + bundler |
| Frontend styling | Tailwind CSS 3 | Utility-first CSS |
| Charting | Recharts | Live metric charts |
| Language | Python 3.11+, JavaScript ES2022 | Implementation languages |

---

## License

MIT License. See `LICENSE` file for details.

---

*Built as a reference implementation for distributed network monitoring with real-time ML-powered fault detection.*
