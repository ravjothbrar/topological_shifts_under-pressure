#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  Topological Shifts Under Pressure — quick-start script
#
#  Usage:
#    chmod +x start.sh
#    ./start.sh                  # uses default model (Qwen3-4B)
#    ./start.sh --share          # creates a public Gradio link
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment…"
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing / updating dependencies…"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "  ┌──────────────────────────────────────────────────┐"
echo "  │  Topological Shifts Under Pressure               │"
echo "  │                                                  │"
echo "  │  Open http://localhost:7860 in your browser      │"
echo "  │  Press Ctrl+C to stop                            │"
echo "  └──────────────────────────────────────────────────┘"
echo ""

python app.py "$@"
