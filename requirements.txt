"""
node_client.py
--------------
Main entry point for a monitoring client node.

Runs three concurrent async loops:
  1. Heartbeat loop   — sends heartbeat every --heartbeat-interval seconds
  2. Metrics loop     — collects and sends metrics every --metrics-interval seconds
  3. File upload loop — uploads a test file every --file-interval seconds

Usage:
    python node_client.py --server 192.168.1.100 --port 8000 --node-id node-A

All measurement is real (live socket probes). See metrics_collector.py.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import platform
import signal
import socket
import sys
import time
import uuid
from datetime import datetime
from typing import Optional

import aiohttp
import websockets
from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
    WebSocketException,
)

from metrics_collector import NetworkMetricsCollector, generate_test_file

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_local_ip() -> str:
    """Best-effort: get the machine's outbound LAN IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def build_node_id(custom: Optional[str]) -> str:
    if custom:
        return custom
    hostname = socket.gethostname().replace(" ", "-").lower()
    short_uuid = str(uuid.uuid4())[:8]
    return f"{hostname}-{short_uuid}"


def get_os_info() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


# ---------------------------------------------------------------------------
# NodeClient
# ---------------------------------------------------------------------------

class NodeClient:
    def __init__(
        self,
        server_host: str,
        server_port: int,
        node_id: str,
        heartbeat_interval: float,
        metrics_interval: float,
        file_interval: float,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.node_id = node_id
        self.heartbeat_interval = heartbeat_interval
        self.metrics_interval = metrics_interval
        self.file_interval = file_interval

        self.local_ip = get_local_ip()
        self.hostname = socket.gethostname()
        self.os_info = get_os_info()

        # WebSocket URL for data streaming
        self.ws_url = (
            f"ws://{server_host}:{server_port}/ws/node/{node_id}"
            f"?hostname={self.hostname}"
            f"&ip={self.local_ip}"
            f"&os_info={self.os_info.replace(' ', '+')}"
        )
        # HTTP URL for file uploads
        self.http_url = f"http://{server_host}:{server_port}"

        self.collector = NetworkMetricsCollector(server_host, server_port)

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._connected = False
        self._reconnect_delay = 2.0      # seconds, doubles on each failure
        self._max_reconnect_delay = 60.0

        # Stats
        self._heartbeats_sent = 0
        self._metrics_sent = 0
        self._files_uploaded = 0
        self._connection_attempts = 0

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    async def _send(self, payload: dict) -> bool:
        """Send a JSON message over the WebSocket. Returns False if disconnected."""
        if self._ws is None or not self._connected:
            return False
        try:
            await self._ws.send(json.dumps(payload, default=str))
            return True
        except (ConnectionClosedError, ConnectionClosedOK):
            self._connected = False
            return False
        except Exception as e:
            logger.warning("Send error: %s", e)
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # Loop: Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self):
        """Sends a heartbeat message every heartbeat_interval seconds."""
        while self._running:
            if self._connected:
                payload = {
                    "type": "heartbeat",
                    "node_id": self.node_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "uptime_s": time.monotonic(),
                }
                ok = await self._send(payload)
                if ok:
                    self._heartbeats_sent += 1
                    logger.debug("Heartbeat #%d sent", self._heartbeats_sent)
            await asyncio.sleep(self.heartbeat_interval)

    # ------------------------------------------------------------------
    # Loop: Metrics
    # ------------------------------------------------------------------

    async def _metrics_loop(self):
        """Collects and sends network metrics every metrics_interval seconds."""
        while self._running:
            if self._connected:
                try:
                    metrics = await self.collector.collect_all()

                    payload = {
                        "type": "metrics",
                        "node_id": self.node_id,
                        "payload": {
                            "latency_ms": metrics["latency_ms"],
                            "packet_loss_rate": metrics["packet_loss_rate"],
                            "throughput_mbps": metrics["throughput_mbps"],
                            "jitter_ms": metrics["jitter_ms"],
                        },
                        "system": metrics.get("system", {}),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    ok = await self._send(payload)
                    if ok:
                        self._metrics_sent += 1
                        logger.info(
                            "Metrics #%d | latency=%.1fms loss=%.2f%% "
                            "throughput=%.2fMbps jitter=%.1fms",
                            self._metrics_sent,
                            metrics["latency_ms"],
                            metrics["packet_loss_rate"] * 100,
                            metrics["throughput_mbps"],
                            metrics["jitter_ms"],
                        )

                except Exception as e:
                    logger.warning("Metrics collection error: %s", e)

            await asyncio.sleep(self.metrics_interval)

    # ------------------------------------------------------------------
    # Loop: File Upload
    # ------------------------------------------------------------------

    async def _file_upload_loop(self):
        """
        Generates a random file, computes SHA256 locally, then POSTs it
        to the server's /api/files/upload endpoint for checksum verification.
        """
        # Wait before first upload to let connection stabilize
        await asyncio.sleep(10)

        while self._running:
            if self._connected:
                try:
                    await self._upload_file()
                except Exception as e:
                    logger.warning("File upload error: %s", e)

            await asyncio.sleep(self.file_interval)

    async def _upload_file(self):
        """Generate, checksum, and upload a test file via HTTP multipart."""
        size = 256 * 1024  # 256 KB
        content, client_checksum = generate_test_file(size)
        filename = f"probe_{self.node_id}_{int(time.time())}.bin"

        url = f"{self.http_url}/api/files/upload"

        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field("node_id", self.node_id)
            form.add_field("client_checksum", client_checksum)
            form.add_field(
                "file",
                content,
                filename=filename,
                content_type="application/octet-stream",
            )

            async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    self._files_uploaded += 1
                    match = result.get("checksum_match", False)
                    corrupted = result.get("corruption_detected", False)

                    if corrupted:
                        logger.error(
                            "FILE UPLOAD #%d | CORRUPTION DETECTED! "
                            "client=%s server=%s",
                            self._files_uploaded,
                            client_checksum[:16],
                            result.get("server_checksum", "")[:16],
                        )
                    else:
                        logger.info(
                            "File upload #%d | size=%dKB checksum_match=%s",
                            self._files_uploaded,
                            size // 1024,
                            match,
                        )
                else:
                    body = await resp.text()
                    logger.warning("File upload failed: HTTP %d — %s", resp.status, body[:200])

    # ------------------------------------------------------------------
    # WebSocket receive loop
    # ------------------------------------------------------------------

    async def _receive_loop(self):
        """
        Listens for messages from the server (acks, commands, pong responses).
        Must run concurrently with send loops to keep the WS alive.
        """
        while self._running and self._ws is not None:
            try:
                raw = await self._ws.recv()
                msg = json.loads(raw)
                msg_type = msg.get("type", "unknown")

                if msg_type == "connection_ack":
                    logger.info(
                        "Server acknowledged connection | node_id=%s",
                        msg.get("node_id"),
                    )
                elif msg_type == "pong":
                    logger.debug("Pong received from server")
                elif msg_type == "command":
                    await self._handle_command(msg)
                else:
                    logger.debug("Server message: %s", msg_type)

            except (ConnectionClosedError, ConnectionClosedOK):
                self._connected = False
                break
            except Exception as e:
                logger.debug("Receive error: %s", e)
                break

    async def _handle_command(self, msg: dict):
        """Handle server-sent commands (e.g. force disconnect, config update)."""
        cmd = msg.get("command")
        if cmd == "disconnect":
            logger.info("Server requested disconnect")
            self._running = False
        elif cmd == "update_interval":
            new_interval = msg.get("metrics_interval")
            if new_interval:
                self.metrics_interval = float(new_interval)
                logger.info("Metrics interval updated to %.1fs", self.metrics_interval)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect_and_run(self):
        """
        Establishes the WebSocket connection and runs all loops concurrently.
        Returns when the connection is lost.
        """
        self._connection_attempts += 1
        logger.info(
            "Connecting [attempt %d] → %s",
            self._connection_attempts,
            self.ws_url,
        )

        try:
            async with websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
                max_size=10 * 1024 * 1024,  # 10 MB max message
            ) as ws:
                self._ws = ws
                self._connected = True
                self._reconnect_delay = 2.0  # reset backoff on success

                logger.info(
                    "Connected to server | node_id=%s host=%s ip=%s",
                    self.node_id,
                    self.server_host,
                    self.local_ip,
                )

                # Run all loops concurrently until any one exits
                await asyncio.gather(
                    self._receive_loop(),
                    self._heartbeat_loop(),
                    self._metrics_loop(),
                    return_exceptions=True,
                )

        except (ConnectionRefusedError, OSError) as e:
            logger.error("Connection refused: %s — is the server running?", e)
        except WebSocketException as e:
            logger.warning("WebSocket error: %s", e)
        except Exception as e:
            logger.error("Unexpected connection error: %s", e)
        finally:
            self._ws = None
            self._connected = False

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self):
        """
        Top-level coroutine. Connects, runs loops, and reconnects
        with exponential backoff on disconnection.
        File upload loop runs independently (uses HTTP, not WebSocket).
        """
        self._running = True

        logger.info("=" * 60)
        logger.info("  AI Network Fault Detection — Client Node")
        logger.info("  Node ID  : %s", self.node_id)
        logger.info("  Hostname : %s", self.hostname)
        logger.info("  Local IP : %s", self.local_ip)
        logger.info("  OS       : %s", self.os_info)
        logger.info("  Server   : %s:%d", self.server_host, self.server_port)
        logger.info("=" * 60)

        # File upload runs independently using HTTP; start it as a separate task
        file_task = asyncio.create_task(self._file_upload_loop())

        try:
            while self._running:
                await self._connect_and_run()

                if not self._running:
                    break

                logger.info(
                    "Disconnected. Reconnecting in %.1fs ...",
                    self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)

                # Exponential backoff, capped at max
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay,
                )
        finally:
            file_task.cancel()
            try:
                await file_task
            except asyncio.CancelledError:
                pass

        self._print_summary()

    def stop(self):
        """Signal the client to shut down gracefully."""
        self._running = False
        if self._ws:
            asyncio.create_task(self._ws.close())

    def _print_summary(self):
        logger.info("=" * 60)
        logger.info("  Session Summary")
        logger.info("  Node ID          : %s", self.node_id)
        logger.info("  Connection attempts: %d", self._connection_attempts)
        logger.info("  Heartbeats sent  : %d", self._heartbeats_sent)
        logger.info("  Metric batches   : %d", self._metrics_sent)
        logger.info("  Files uploaded   : %d", self._files_uploaded)
        logger.info("  Cumulative loss  : %.2f%%", self.collector.cumulative_loss_rate * 100)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IoT Network Fault Detection — Client Node Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server", "-s",
        default="127.0.0.1",
        help="Server IP address or hostname",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Server port",
    )
    parser.add_argument(
        "--node-id", "-n",
        default=None,
        help="Unique node identifier (auto-generated if omitted)",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=2.0,
        help="Seconds between heartbeat messages",
    )
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=3.0,
        help="Seconds between metric collection and send",
    )
    parser.add_argument(
        "--file-interval",
        type=float,
        default=30.0,
        help="Seconds between file upload probes",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    args = parse_args()

    # Apply log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    node_id = build_node_id(args.node_id)

    client = NodeClient(
        server_host=args.server,
        server_port=args.port,
        node_id=node_id,
        heartbeat_interval=args.heartbeat_interval,
        metrics_interval=args.metrics_interval,
        file_interval=args.file_interval,
    )

    # Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()

    def _shutdown():
        logger.info("Shutdown signal received. Stopping node client...")
        client.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            # Windows does not support add_signal_handler for all signals
            pass

    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client stopped by user.")
