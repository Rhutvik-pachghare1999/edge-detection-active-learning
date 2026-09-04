#!/bin/bash
set -e
API="http://localhost:8000"
DURATION=120

# Reset database
rm -f ~/aecs-sdc/data/events.db
echo "Database cleared. Starting experiment..."
sleep 1

run_scenario() {
    local name=$1
    local label=$2
    local action=$3

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  SCENARIO: $label"
    echo "  ACTION:   $action"
    echo "════════════════════════════════════════════════════"

    curl -s -X PUT "$API/scenario?name=$name" > /dev/null
    echo "  Scenario set. Starting in 3s..."
    sleep 3

    for ((i=DURATION; i>=0; i--)); do
        mins=$(printf "%02d" $((i/60)))
        secs=$(printf "%02d" $((i%60)))
        printf "\r  [%s] %s — %s:%s remaining   " "$label" "$action" "$mins" "$secs"
        sleep 1
    done
    echo ""
    echo "  DONE — NEXT SCENARIO"
}

run_scenario "nominal"   "NOMINAL"   "Normal lighting, nothing blocking the camera"
run_scenario "low_light" "LOW_LIGHT" "Turn off the room lights now"
run_scenario "occlusion" "OCCLUSION" "Hold an object close in front of the camera"
run_scenario "motion"    "MOTION"    "Shake the drone/laptop while recording"

echo ""
echo "════════════════════════════════════════════════════"
echo "  EXPERIMENT COMPLETE — RESULTS"
echo "════════════════════════════════════════════════════"
curl -s "$API/results" | python3 -m json.tool
