#!/usr/bin/env python3
"""
=============================================================
  FAULT INJECTION CLIENT
  Project: AI-Based Fault Detection in Distributed Networks
=============================================================
  HOW TO USE (Quick Steps for Review):
  -------------------------------------
  1. Install dependency (if not already done):
       pip install requests

  2. Edit the CONFIG section below:
       - Set SERVER_URL to your server's IP/port
       - Set NODE_IDS to match your network nodes

  3. Run this file:
       python fault_injection_client.py

  4. Watch your dashboard — faults will appear in real time!
=============================================================
"""

import requests
import time
import random
import json
import threading
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIG — Edit these to match your setup
# ─────────────────────────────────────────────
SERVER_URL = "http://localhost:5000"          # Change to your server IP:PORT
FAULT_ENDPOINT = "/api/fault"                 # Your server's fault intake endpoint
HEARTBEAT_ENDPOINT = "/api/heartbeat"         # Optional: heartbeat/status endpoint
NODE_IDS = ["Node-1", "Node-2", "Node-3",
            "Node-4", "Node-5"]               # Must match your dashboard node names
INJECT_INTERVAL_SECONDS = 4                   # Time between fault injections
AUTO_STOP_AFTER = 60                          # Stop after N seconds (set 0 to run forever)
# ─────────────────────────────────────────────


# ── Fault Type Definitions ────────────────────
FAULT_TYPES = [
    {
        "type": "packet_loss",
        "severity": "high",
        "description": "Excessive packet loss detected on link",
        "metrics": {"packet_loss_pct": 85, "threshold": 10}
    },
    {
        "type": "latency_spike",
        "severity": "medium",
        "description": "Abnormal latency spike on node",
        "metrics": {"latency_ms": 950, "normal_ms": 20}
    },
    {
        "type": "node_failure",
        "severity": "critical",
        "description": "Node unresponsive — possible hardware failure",
        "metrics": {"uptime": 0, "last_seen_seconds_ago": 120}
    },
    {
        "type": "bandwidth_saturation",
        "severity": "high",
        "description": "Bandwidth utilization exceeded safe threshold",
        "metrics": {"utilization_pct": 98, "threshold_pct": 80}
    },
    {
        "type": "routing_loop",
        "severity": "critical",
        "description": "Routing loop detected — circular path identified",
        "metrics": {"hop_count": 32, "max_expected": 8}
    },
    {
        "type": "memory_overflow",
        "severity": "high",
        "description": "Node buffer overflow causing dropped connections",
        "metrics": {"memory_used_pct": 99, "threshold_pct": 85}
    },
    {
        "type": "link_flapping",
        "severity": "medium",
        "description": "Unstable link — repeated up/down transitions",
        "metrics": {"flap_count": 14, "window_seconds": 60}
    },
    {
        "type": "cpu_overload",
        "severity": "high",
        "description": "Node CPU saturated — processing delays expected",
        "metrics": {"cpu_pct": 97, "threshold_pct": 75}
    },
    {
        "type": "authentication_failure",
        "severity": "medium",
        "description": "Repeated auth failures — possible intrusion attempt",
        "metrics": {"failed_attempts": 23, "window_seconds": 30}
    },
    {
        "type": "network_partition",
        "severity": "critical",
        "description": "Network partition detected — nodes isolated",
        "metrics": {"isolated_nodes": 2, "total_nodes": len(NODE_IDS)}
    },
]


def build_fault_payload(node_id):
    """Construct a realistic fault payload for the given node."""
    fault = random.choice(FAULT_TYPES)
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "node_id": node_id,
        "fault_type": fault["type"],
        "severity": fault["severity"],
        "description": f"[{node_id}] {fault['description']}",
        "metrics": fault["metrics"],
        "source": "fault_injection_client",
        "ai_flag": True          # tells your AI module this is a detectable event
    }
    return payload


def send_fault(node_id):
    """Send one fault event to the server."""
    payload = build_fault_payload(node_id)
    try:
        response = requests.post(
            SERVER_URL + FAULT_ENDPOINT,
            json=payload,
            timeout=5
        )
        status = response.status_code
        indicator = "✅" if status == 200 else "⚠️"
        print(f"  {indicator} [{payload['timestamp']}] Fault sent → {node_id} | "
              f"Type: {payload['fault_type']} | Severity: {payload['severity'].upper()} | "
              f"HTTP {status}")
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Could not connect to server at {SERVER_URL}. "
              f"Check SERVER_URL in config.")
    except requests.exceptions.Timeout:
        print(f"  ⏱️  Request timed out for {node_id}.")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")


def send_heartbeat():
    """Optional: send a heartbeat so dashboard knows the client is alive."""
    try:
        requests.post(
            SERVER_URL + HEARTBEAT_ENDPOINT,
            json={"client": "fault_injection_client", "status": "active",
                  "timestamp": datetime.utcnow().isoformat() + "Z"},
            timeout=3
        )
    except Exception:
        pass   # Heartbeat failure is non-critical


def run_injection_loop():
    """Main loop — injects faults at regular intervals."""
    print("\n" + "="*60)
    print("  🚨 FAULT INJECTION CLIENT STARTED")
    print("="*60)
    print(f"  Server   : {SERVER_URL}")
    print(f"  Nodes    : {', '.join(NODE_IDS)}")
    print(f"  Interval : {INJECT_INTERVAL_SECONDS}s between faults")
    print(f"  Duration : {AUTO_STOP_AFTER}s" if AUTO_STOP_AFTER else "  Duration : Until Ctrl+C")
    print("="*60 + "\n")

    start_time = time.time()
    injection_count = 0

    try:
        while True:
            # Pick a random node for each fault
            target_node = random.choice(NODE_IDS)

            # Occasionally inject on multiple nodes (cascade simulation)
            if random.random() < 0.25:
                secondary_node = random.choice(
                    [n for n in NODE_IDS if n != target_node]
                )
                threading.Thread(
                    target=send_fault, args=(secondary_node,), daemon=True
                ).start()

            send_fault(target_node)
            send_heartbeat()
            injection_count += 1

            elapsed = time.time() - start_time
            if AUTO_STOP_AFTER and elapsed >= AUTO_STOP_AFTER:
                print(f"\n✅ Auto-stop reached. {injection_count} faults injected in "
                      f"{int(elapsed)}s.")
                break

            time.sleep(INJECT_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n🛑 Stopped by user. {injection_count} faults injected in "
              f"{int(elapsed)}s.")


if __name__ == "__main__":
    run_injection_loop()