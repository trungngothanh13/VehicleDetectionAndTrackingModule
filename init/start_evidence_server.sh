#!/bin/bash
# Start the Evidence HTTP file server in background on port 8080
INIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$INIT_DIR/.." && pwd)"

# Always run from project root so output/evidence is correctly located
cd "$ROOT_DIR"

# Check if already running
if pgrep -f "evidence_server.py" > /dev/null; then
    echo "📦 [Evidence Server] Already running on port 8080 (PID: $(pgrep -f 'evidence_server.py'))"
    exit 0
fi

echo "📦 [Evidence Server] Starting on port 8080 in background..."
nohup python3 "$INIT_DIR/evidence_server.py" --port 8080 --evidence-dir "$ROOT_DIR/output/evidence" > "$INIT_DIR/evidence_server.log" 2>&1 &
sleep 1

if pgrep -f "evidence_server.py" > /dev/null; then
    echo "✅ [Evidence Server] Running successfully (PID: $(pgrep -f 'evidence_server.py'))"
else
    echo "❌ [Evidence Server] Failed to start. Check $INIT_DIR/evidence_server.log"
    cat "$INIT_DIR/evidence_server.log"
fi
