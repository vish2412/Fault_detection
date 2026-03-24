#!/bin/bash
# NetFault - AI Fault Detection System - Quick Start
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   NETFAULT  ·  AI Fault Detection System     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")/backend"

echo "[1/3] Installing dependencies..."
pip install fastapi uvicorn websockets scikit-learn numpy joblib --quiet --break-system-packages

echo "[2/3] Starting server on http://localhost:8000"
echo "      → Dashboard: http://localhost:8000"
echo "      → API docs:  http://localhost:8000/docs"
echo ""
echo "[3/3] Training ML model in background (takes ~5s)..."
echo ""

python server.py
