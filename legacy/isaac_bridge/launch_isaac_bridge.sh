#!/bin/bash
# AECS-SDC Isaac Sim ROS2 Bridge Launcher
#
# Isaac Sim 4.5 only supports ROS2 Humble internally.
# This script overrides ROS_DISTRO and LD_LIBRARY_PATH to use
# Isaac Sim's bundled Humble libs instead of the system Jazzy install.

ISAAC_ROOT=~/isaacsim
BRIDGE_EXT="$ISAAC_ROOT/exts/isaacsim.ros2.bridge"
HUMBLE_LIBS="$BRIDGE_EXT/humble/lib"

echo "=== AECS-SDC Isaac Sim ROS2 Bridge ==="
echo "Isaac root:   $ISAAC_ROOT"
echo "Humble libs:  $HUMBLE_LIBS"
echo ""

# Override to Humble so the bridge extension accepts the distro
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="$HUMBLE_LIBS:${LD_LIBRARY_PATH}"

# Unset Jazzy-specific vars that would conflict
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH

echo "ROS_DISTRO overridden to: $ROS_DISTRO (using Isaac bundled Humble libs)"
echo "Starting Isaac Sim..."
echo ""

"$ISAAC_ROOT/python.sh" ~/aecs-sdc/isaac_bridge/isaac_ros2_bridge.py "$@"
