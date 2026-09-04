"""
AECS-SDC Isaac Sim 5.0 Frame Server
- Runs Isaac Sim headless, renders a warehouse-like scene
- Serves JPEG frames via HTTP GET /frame at ~10 Hz
- Supports 3 scenarios: nominal, low_light, occlusion
- Scenario switched via PUT /scenario?name=<scenario>
- Run via: bash ~/aecs-sdc/isaac_bridge/launch_frame_server.sh
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import io
import threading
import time
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.sensors.camera import Camera

# ── Scene setup ──────────────────────────────────────────────────────────────

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Warehouse objects — forklifts represented as coloured cuboids
_objects = []
for i, (pos, col, scale) in enumerate([
    ([2.0,  1.0, 0.5], [0.8, 0.4, 0.1], [0.8, 1.6, 1.0]),   # forklift A
    ([-2.0, -1.0, 0.5], [0.1, 0.4, 0.8], [0.8, 1.6, 1.0]),  # forklift B
    ([0.0,  3.0, 0.3], [0.2, 0.7, 0.2], [1.2, 0.6, 0.6]),   # pallet
    ([1.5, -2.5, 0.4], [0.7, 0.7, 0.1], [0.5, 0.5, 0.8]),   # box stack
]):
    _objects.append(world.scene.add(DynamicCuboid(
        prim_path=f"/World/obj_{i}",
        name=f"obj_{i}",
        position=np.array(pos),
        scale=np.array(scale),
        size=1.0,
        color=np.array(col),
    )))

# Occlusion panel — toggled for occlusion scenario
_occluder = world.scene.add(VisualCuboid(
    prim_path="/World/occluder",
    name="occluder",
    position=np.array([0.0, 0.0, 2.5]),
    scale=np.array([6.0, 6.0, 0.05]),
    size=1.0,
    color=np.array([0.15, 0.15, 0.15]),
))

# Camera — top-down, 4m above floor, slight tilt
camera = Camera(
    prim_path="/World/cam",
    position=np.array([0.0, 0.0, 5.0]),
    frequency=30,
    resolution=(640, 480),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 85, 0]), degrees=True),
)

world.reset()
camera.initialize()

# Warm up renderer
for _ in range(60):
    world.step(render=True)

# ── Scenario state ────────────────────────────────────────────────────────────

_scenario = "nominal"
_scenario_lock = threading.Lock()

# Access stage for lighting control
import omni.usd
from pxr import Sdf

def _get_or_create_dome_light():
    stage = omni.usd.get_context().get_stage()
    path = "/World/DomeLight"
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        prim = stage.DefinePrim(path, "DomeLight")
    # Use inputs:intensity attribute (Isaac Sim 5.0 style)
    attr = prim.GetAttribute("inputs:intensity")
    if not attr.IsValid():
        attr = prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float)
    attr.Set(1000.0)
    return prim

_dome_prim = _get_or_create_dome_light()

def _set_dome_intensity(value: float):
    _dome_prim.GetAttribute("inputs:intensity").Set(value)

def _apply_scenario(name: str):
    """Mutate scene to match scenario."""
    if name == "nominal":
        _set_dome_intensity(1000.0)
        _occluder.set_visibility(False)
    elif name == "low_light":
        _set_dome_intensity(80.0)
        _occluder.set_visibility(False)
    elif name == "occlusion":
        _set_dome_intensity(1000.0)
        _occluder.set_visibility(True)

# ── Shared frame buffer ───────────────────────────────────────────────────────

_frame_lock = threading.Lock()
_jpeg_bytes: bytes = b""

def _sim_loop():
    """Runs in main thread — steps sim and updates _jpeg_bytes."""
    global _jpeg_bytes, _scenario
    last_scenario = None
    while simulation_app.is_running():
        with _scenario_lock:
            current = _scenario
        if current != last_scenario:
            _apply_scenario(current)
            last_scenario = current

        world.step(render=True)
        rgba = camera.get_rgba()
        if rgba is not None and rgba.size > 0:
            bgr = cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with _frame_lock:
                _jpeg_bytes = buf.tobytes()

# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress access logs

    def do_GET(self):
        if self.path == "/frame":
            with _frame_lock:
                data = _jpeg_bytes
            if not data:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            with _scenario_lock:
                s = _scenario
            self.wfile.write(f"ok scenario={s}".encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_PUT(self):
        parsed = urlparse(self.path)
        if parsed.path == "/scenario":
            qs = parse_qs(parsed.query)
            name = qs.get("name", ["nominal"])[0]
            if name in ("nominal", "low_light", "occlusion"):
                with _scenario_lock:
                    global _scenario
                    _scenario = name
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(f'{{"scenario":"{name}"}}'.encode())
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


PORT = 8002
server = HTTPServer(("0.0.0.0", PORT), Handler)
server_thread = threading.Thread(target=server.serve_forever, daemon=True)
server_thread.start()
print(f"Frame server listening on http://0.0.0.0:{PORT}")
print("GET  /frame          → latest JPEG")
print("GET  /health         → status")
print("PUT  /scenario?name= → nominal | low_light | occlusion")

# ── Main sim loop (must run on main thread for Isaac Sim) ─────────────────────
_sim_loop()

server.shutdown()
simulation_app.close()
