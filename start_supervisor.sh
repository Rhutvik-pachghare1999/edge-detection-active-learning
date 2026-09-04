#!/bin/bash
source ~/venvs/aecs-supervisor/bin/activate
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
cd ~/aecs-sdc/supervisor
echo "Starting AECS-SDC Supervisor on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level info
