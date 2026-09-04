#!/bin/bash
# Usage: ./set_scenario.sh nominal | low_light | occlusion | motion
SCENARIO=${1:?Usage: $0 nominal|low_light|occlusion|motion}
curl -s -X PUT "http://localhost:8000/scenario?name=$SCENARIO" | python3 -m json.tool
echo ""
curl -s http://localhost:8000/status | python3 -c "
import sys,json; s=json.load(sys.stdin)
print(f'current_scenario : {s[\"current_scenario\"]}')
print(f'total_frames     : {s[\"total_frames\"]}')
print(f'harvest_rate     : {s[\"harvest_rate_pct\"]}%')
print(f'tau              : {s[\"current_tau\"]}')
"
