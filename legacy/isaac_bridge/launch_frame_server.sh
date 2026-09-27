#!/bin/bash
# AECS-SDC Isaac Sim 5.0 Frame Server Launcher
ISAAC_ROOT="$HOME/.local/share/ov/pkg/isaac_sim-5.0.0"

echo "=== AECS-SDC Isaac Sim 5.0 Frame Server ==="
echo "Isaac root: $ISAAC_ROOT"
echo "Frame server will be available at http://localhost:8002"
echo ""

"$ISAAC_ROOT/python.sh" ~/aecs-sdc/isaac_bridge/frame_server.py "$@"
