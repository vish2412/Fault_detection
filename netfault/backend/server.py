"""
NetFault - AI-Based Fault Detection in Distributed Networks
Backend: FastAPI + WebSocket + ML Pipeline
"""

import asyncio
import json
import random
import time
import threading
import math
from datetime import datetime
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = FastAPI(title="NetFault API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ────────────────────────────────────────────────────────────

NODES = {
    "NODE-A": {"region": "US-EAST", "role": "primary",   "status": "healthy"},
    "NODE-B": {"region": "EU-WEST", "role": "replica",   "status": "healthy"},
    "NODE-C": {"region": "AP-SOUTH","role": "compute",   "status": "healthy"},
    "NODE-D": {"region": "US-WEST", "role": "gateway",   "status": "healthy"},
    "NODE-E": {"region": "EU-CENTRAL","role":"cache",    "status": "healthy"},
    "NODE-F": {"region": "AP-EAST", "role": "replica",   "status": "healthy"},
}

node_metrics = {nid: deque(maxlen=200) for nid in NODES}
fault_log    = deque(maxlen=500)
active_faults = {}   # node_id -> fault_type
model_store  = {}    # node_id -> IsolationForest
scaler_store = {}    # node_id -> StandardScaler
model_trained = False
clients: list[WebSocket] = []
system_stats = {
    "total_packets": 0,
    "total_faults_detected": 0,
    "model_accuracy": 0.0,
    "uptime_start": time.time(),
}

# ─── Metric Generation ───────────────────────────────────────────────────────

def base_metrics(node_id: str) -> dict:
    """Generate realistic baseline metrics for a healthy node."""
    t = time.time()
    # Sinusoidal variation to simulate real network patterns
    wave = math.sin(t * 0.1 + hash(node_id) % 10)
    return {
        "rtt_ms":        max(1, 12 + wave * 4 + random.gauss(0, 1.5)),
        "packet_loss":   max(0, 0.3 + wave * 0.2 + random.gauss(0, 0.15)),
        "cpu_pct":       max(0, min(100, 28 + wave * 8 + random.gauss(0, 3))),
        "mem_pct":       max(0, min(100, 42 + wave * 5 + random.gauss(0, 2))),
        "throughput_mbps": max(0, 85 + wave * 20 + random.gauss(0, 5)),
        "error_rate":    max(0, 0.01 + random.gauss(0, 0.005)),
        "jitter_ms":     max(0, 2 + abs(random.gauss(0, 0.8))),
        "connections":   max(0, int(120 + wave * 30 + random.gauss(0, 10))),
    }

def apply_fault(metrics: dict, fault_type: str) -> dict:
    """Inject fault signatures into metrics."""
    m = metrics.copy()
    if fault_type == "packet_loss":
        m["packet_loss"]  = random.uniform(18, 45)
        m["error_rate"]   = random.uniform(0.08, 0.25)
        m["throughput_mbps"] = max(0, m["throughput_mbps"] * random.uniform(0.3, 0.6))
    elif fault_type == "high_latency":
        m["rtt_ms"]       = random.uniform(300, 900)
        m["jitter_ms"]    = random.uniform(50, 200)
        m["throughput_mbps"] = max(0, m["throughput_mbps"] * 0.5)
    elif fault_type == "cpu_exhaustion":
        m["cpu_pct"]      = random.uniform(88, 99)
        m["rtt_ms"]       = m["rtt_ms"] * random.uniform(2, 5)
        m["error_rate"]   = random.uniform(0.05, 0.15)
    elif fault_type == "memory_leak":
        m["mem_pct"]      = random.uniform(85, 99)
        m["cpu_pct"]      = min(100, m["cpu_pct"] + random.uniform(20, 40))
        m["throughput_mbps"] = max(0, m["throughput_mbps"] * 0.4)
    elif fault_type == "network_partition":
        m["packet_loss"]  = random.uniform(60, 95)
        m["rtt_ms"]       = random.uniform(500, 2000)
        m["connections"]  = max(0, int(m["connections"] * 0.1))
        m["throughput_mbps"] = max(0, m["throughput_mbps"] * 0.05)
    elif fault_type == "crash":
        m = {k: 0.0 for k in m}
        m["error_rate"] = 1.0
    elif fault_type == "bandwidth_saturation":
        m["throughput_mbps"] = max(0, m["throughput_mbps"] * random.uniform(0.05, 0.15))
        m["rtt_ms"]       = random.uniform(200, 600)
        m["packet_loss"]  = random.uniform(10, 30)
    return m

# ─── ML Pipeline ─────────────────────────────────────────────────────────────

FEATURES = ["rtt_ms","packet_loss","cpu_pct","mem_pct","throughput_mbps","error_rate","jitter_ms","connections"]

def extract_features(metrics: dict) -> np.ndarray:
    return np.array([metrics.get(f, 0) for f in FEATURES])

def generate_training_data(n_normal=800, n_fault=200):
    """Generate synthetic labelled training data."""
    X, y = [], []
    fault_types = ["packet_loss","high_latency","cpu_exhaustion","memory_leak","network_partition","bandwidth_saturation"]
    for _ in range(n_normal):
        m = base_metrics("NODE-A")
        X.append(extract_features(m))
        y.append(0)
    for _ in range(n_fault):
        ft = random.choice(fault_types)
        m = apply_fault(base_metrics("NODE-A"), ft)
        X.append(extract_features(m))
        y.append(1)
    return np.array(X), np.array(y)

def train_models():
    global model_trained, model_store, scaler_store
    print("[ML] Training Isolation Forest models...")
    X_normal = []
    for _ in range(1000):
        m = base_metrics("NODE-A")
        X_normal.append(extract_features(m))
    X_normal = np.array(X_normal)

    for node_id in NODES:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_normal)
        clf = IsolationForest(n_estimators=150, contamination=0.05, random_state=42, n_jobs=-1)
        clf.fit(X_scaled)
        model_store[node_id] = clf
        scaler_store[node_id] = scaler

    # Evaluate accuracy
    X_all, y_all = generate_training_data()
    scaler = scaler_store["NODE-A"]
    clf = model_store["NODE-A"]
    X_scaled = scaler.transform(X_all)
    preds = clf.predict(X_scaled)  # -1=anomaly, 1=normal
    preds_binary = (preds == -1).astype(int)
    acc = (preds_binary == y_all).mean()
    system_stats["model_accuracy"] = round(float(acc) * 100, 1)
    model_trained = True
    print(f"[ML] Models trained. Accuracy on test set: {acc:.1%}")

def detect_fault(node_id: str, metrics: dict) -> dict:
    """Run anomaly detection + fault classification."""
    if node_id not in model_store:
        return {"is_fault": False, "fault_type": "none", "confidence": 0, "score": 0}
    
    features = extract_features(metrics).reshape(1, -1)
    scaler = scaler_store[node_id]
    clf = model_store[node_id]
    X_scaled = scaler.transform(features)
    pred = clf.predict(X_scaled)[0]       # -1 = anomaly
    score = clf.score_samples(X_scaled)[0]
    
    is_fault = pred == -1
    confidence = max(0, min(100, int((1 - (score + 0.5)) * 100)))

    fault_type = "none"
    if is_fault:
        # Rule-based classifier on top of anomaly score
        m = metrics
        if m["packet_loss"] > 15 and m["throughput_mbps"] < 50:
            fault_type = "packet_loss"
        elif m["rtt_ms"] > 250:
            fault_type = "high_latency"
        elif m["cpu_pct"] > 85:
            fault_type = "cpu_exhaustion"
        elif m["mem_pct"] > 85:
            fault_type = "memory_leak"
        elif m["packet_loss"] > 50 and m["connections"] < 20:
            fault_type = "network_partition"
        elif m["throughput_mbps"] < 10 and m["rtt_ms"] > 150:
            fault_type = "bandwidth_saturation"
        elif all(v < 1 for v in [m["rtt_ms"], m["cpu_pct"], m["throughput_mbps"]]):
            fault_type = "crash"
        else:
            fault_type = "anomaly"

    return {
        "is_fault": is_fault,
        "fault_type": fault_type,
        "confidence": confidence,
        "score": round(float(score), 4),
    }

# ─── Node Simulation Loop ─────────────────────────────────────────────────────

async def simulate_nodes():
    """Background task: emit metrics for all nodes every 800ms."""
    global system_stats
    while True:
        snapshot = {}
        for node_id in NODES:
            m = base_metrics(node_id)
            fault = active_faults.get(node_id)
            if fault:
                m = apply_fault(m, fault)

            detection = detect_fault(node_id, m) if model_trained else {
                "is_fault": False, "fault_type": "none", "confidence": 0, "score": 0
            }

            status = "healthy"
            if detection["is_fault"]:
                status = "critical" if detection["confidence"] > 70 else "warning"
                if detection["fault_type"] == "crash":
                    status = "offline"

            NODES[node_id]["status"] = status
            system_stats["total_packets"] += random.randint(800, 1200)

            entry = {
                "node_id": node_id,
                "timestamp": time.time(),
                "ts_str": datetime.now().strftime("%H:%M:%S"),
                "metrics": {k: round(v, 2) for k, v in m.items()},
                "detection": detection,
                "status": status,
                "region": NODES[node_id]["region"],
                "role": NODES[node_id]["role"],
                "fault_injected": fault,
            }
            node_metrics[node_id].append(entry)
            snapshot[node_id] = entry

            if detection["is_fault"]:
                system_stats["total_faults_detected"] += 1
                fault_log.appendleft({
                    "id": f"F{int(time.time()*1000)%100000}",
                    "node_id": node_id,
                    "ts_str": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "fault_type": detection["fault_type"],
                    "confidence": detection["confidence"],
                    "score": detection["score"],
                    "metrics_snapshot": {k: round(v, 2) for k, v in m.items()},
                })

        payload = {
            "type": "metrics_update",
            "data": snapshot,
            "system": {
                **system_stats,
                "uptime_s": int(time.time() - system_stats["uptime_start"]),
                "model_trained": model_trained,
                "active_faults": dict(active_faults),
                "healthy_nodes": sum(1 for n in NODES.values() if n["status"] == "healthy"),
                "fault_count": len(active_faults),
            },
            "recent_faults": list(fault_log)[:20],
        }

        dead = []
        for ws in clients:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.remove(ws)

        await asyncio.sleep(0.8)

# ─── REST API ─────────────────────────────────────────────────────────────────

@app.get("/api/nodes")
def get_nodes():
    return {"nodes": NODES, "active_faults": dict(active_faults)}

@app.get("/api/history/{node_id}")
def get_history(node_id: str, limit: int = 60):
    if node_id not in node_metrics:
        return {"error": "Node not found"}
    return {"history": list(node_metrics[node_id])[-limit:]}

@app.get("/api/faults")
def get_faults(limit: int = 50):
    return {"faults": list(fault_log)[:limit]}

@app.post("/api/inject_fault")
async def inject_fault(payload: dict):
    node_id = payload.get("node_id")
    fault_type = payload.get("fault_type")
    if node_id not in NODES:
        return {"error": "Node not found"}
    if fault_type == "clear":
        active_faults.pop(node_id, None)
        NODES[node_id]["status"] = "healthy"
        return {"message": f"Fault cleared on {node_id}"}
    active_faults[node_id] = fault_type
    return {"message": f"Fault '{fault_type}' injected on {node_id}"}

@app.post("/api/inject_all")
async def inject_all(payload: dict):
    fault_type = payload.get("fault_type", "clear")
    for node_id in NODES:
        if fault_type == "clear":
            active_faults.pop(node_id, None)
            NODES[node_id]["status"] = "healthy"
        else:
            active_faults[node_id] = fault_type
    return {"message": f"Applied '{fault_type}' to all nodes"}

@app.get("/api/model/stats")
def model_stats():
    return {
        "trained": model_trained,
        "accuracy": system_stats["model_accuracy"],
        "nodes_monitored": len(NODES),
        "features": FEATURES,
        "algorithm": "Isolation Forest",
        "n_estimators": 150,
        "contamination": 0.05,
    }

@app.get("/api/stats")
def get_stats():
    node_statuses = {nid: NODES[nid]["status"] for nid in NODES}
    return {
        **system_stats,
        "uptime_s": int(time.time() - system_stats["uptime_start"]),
        "node_statuses": node_statuses,
        "active_faults": dict(active_faults),
        "model_trained": model_trained,
    }

# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    init = {
        "type": "init",
        "nodes": NODES,
        "history": {nid: list(node_metrics[nid])[-30:] for nid in NODES},
        "faults": list(fault_log)[:30],
        "system": system_stats,
    }
    await ws.send_text(json.dumps(init))
    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send a ping to keep alive
                await ws.send_text(json.dumps({"type": "ping"}))
    except (WebSocketDisconnect, Exception):
        if ws in clients:
            clients.remove(ws)

# ─── Frontend route (explicit, before StaticFiles mount) ─────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "../frontend/index.html"))

# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, train_models)
    asyncio.create_task(simulate_nodes())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, ws_ping_interval=20, ws_ping_timeout=60)
