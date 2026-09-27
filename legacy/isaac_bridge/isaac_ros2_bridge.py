"""
AECS-SDC Isaac Sim ROS2 Camera Bridge
Publishes warehouse camera on /warehouse_cam/image_raw at 10 Hz.

Run via: bash ~/aecs-sdc/isaac_bridge/launch_isaac_bridge.sh
Do NOT run directly with python — use the launch script which sets env vars.
"""

from isaacsim import SimulationApp

CAMERA_PATH = "/World/WarehouseCam"
GRAPH_PATH = "/ROS2_WarehouseCam"
WAREHOUSE_USD = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
TOPIC = "warehouse_cam/image_raw"
FRAME_ID = "warehouse_cam"
PUBLISH_EVERY_N_FRAMES = 3  # ~10 Hz at 30 Hz sim

simulation_app = SimulationApp({"renderer": "RaytracedLighting", "headless": False})

import sys
import carb
import omni
import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import extensions, stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom

# Enable ROS2 bridge (uses bundled Humble via env vars set in launch script)
extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

sim = SimulationContext(stage_units_in_meters=1.0)

# Load warehouse environment
assets_root = get_assets_root_path()
if assets_root is None:
    carb.log_error("Cannot find Isaac Sim assets root. Check Nucleus connection.")
    simulation_app.close()
    sys.exit(1)

stage.add_reference_to_stage(assets_root + WAREHOUSE_USD, "/background")

# Create camera prim — positioned above warehouse floor looking down at 45°
cam_prim = UsdGeom.Camera(
    omni.usd.get_context().get_stage().DefinePrim(CAMERA_PATH, "Camera")
)
xform = UsdGeom.XformCommonAPI(cam_prim)
xform.SetTranslate(Gf.Vec3d(0, 0, 4.0))       # 4m above floor
xform.SetRotate((60, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)  # 60° tilt
cam_prim.GetHorizontalApertureAttr().Set(21)
cam_prim.GetVerticalApertureAttr().Set(16)
cam_prim.GetFocalLengthAttr().Set(24)
cam_prim.GetProjectionAttr().Set("perspective")

simulation_app.update()

# Build OmniGraph: OnTick → Viewport → RenderProduct → SetCamera → ROS2CameraHelper
keys = og.Controller.Keys
og.Controller.edit(
    {
        "graph_path": GRAPH_PATH,
        "evaluator_name": "push",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
    },
    {
        keys.CREATE_NODES: [
            ("OnTick",           "omni.graph.action.OnTick"),
            ("createViewport",   "isaacsim.core.nodes.IsaacCreateViewport"),
            ("getRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
            ("setCamera",        "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
            ("cameraRgb",        "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick",                        "createViewport.inputs:execIn"),
            ("createViewport.outputs:execOut",             "getRenderProduct.inputs:execIn"),
            ("createViewport.outputs:viewport",            "getRenderProduct.inputs:viewport"),
            ("getRenderProduct.outputs:execOut",           "setCamera.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
            ("setCamera.outputs:execOut",                  "cameraRgb.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "cameraRgb.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("createViewport.inputs:viewportId",  0),
            ("cameraRgb.inputs:frameId",          FRAME_ID),
            ("cameraRgb.inputs:topicName",        TOPIC),
            ("cameraRgb.inputs:type",             "rgb"),
            ("setCamera.inputs:cameraPrim",       [usdrt.Sdf.Path(CAMERA_PATH)]),
        ],
    },
)

# Evaluate once to register the ROS2 publishers in the SDG pipeline
og.Controller.evaluate_sync(og.get_graph_by_path(GRAPH_PATH))
simulation_app.update()

# Set publish rate via IsaacSimulationGate
try:
    from omni.kit.viewport.utility import get_active_viewport
    import omni.syntheticdata
    import omni.syntheticdata._syntheticdata as sd

    vp = get_active_viewport()
    if vp:
        rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
        gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv + "IsaacSimulationGate", vp.get_render_product_path()
        )
        og.Controller.attribute(gate_path + ".inputs:step").set(PUBLISH_EVERY_N_FRAMES)
        print(f"Camera gate set: publish every {PUBLISH_EVERY_N_FRAMES} frames (~10 Hz)")
except Exception as e:
    carb.log_warn(f"Could not set simulation gate step: {e}")

sim.initialize_physics()
sim.play()

print(f"Publishing on ROS2 topic: /{TOPIC}")
print("Isaac Sim running. Press Ctrl+C to stop.")

while simulation_app.is_running():
    sim.step(render=True)

sim.stop()
simulation_app.close()
