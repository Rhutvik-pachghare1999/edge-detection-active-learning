"""
Subscribes to Isaac Sim ROS2 camera topic and feeds frames into an asyncio queue.
Falls back to a simulated frame generator when ROS2/Isaac Sim is unavailable.
"""
import sys
import asyncio
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("WARNING: rclpy not available — using simulated frames")

sys.path.insert(0, '../supervisor')
from config import ROS2_TOPIC

frame_queue: asyncio.Queue = asyncio.Queue(maxsize=10)


class CameraSubscriber:
    def __init__(self, topic: str):
        rclpy.init()
        self.node = rclpy.create_node("aecs_camera_subscriber")
        self.node.create_subscription(Image, topic, self._callback, 10)
        print(f"Subscribed to {topic}")

    def _callback(self, msg: "Image"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        try:
            frame_queue.put_nowait(arr.copy())
        except asyncio.QueueFull:
            pass

    def spin(self):
        rclpy.spin(self.node)


class SimulatedFrameGenerator:
    async def run(self):
        import cv2
        print("Using SIMULATED frames (Isaac Sim not connected)")
        fid = 0
        while True:
            frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            if fid % 30 < 10:
                cv2.rectangle(frame, (100, 100), (300, 300), (200, 50, 50), -1)
            try:
                frame_queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass
            fid += 1
            await asyncio.sleep(0.1)


def get_frame_source():
    if ROS2_AVAILABLE:
        try:
            return CameraSubscriber(ROS2_TOPIC)
        except Exception as e:
            print(f"ROS2 init failed: {e} — falling back to simulation")
    return SimulatedFrameGenerator()
