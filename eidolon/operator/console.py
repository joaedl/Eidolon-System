"""Remote operator console for teleoperation."""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import structlog
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from ..teleop.webrtc_client import TeleopClient, TeleopConfig
from ..common.config import get_config

logger = structlog.get_logger(__name__)


class OperatorMode(Enum):
    """Operator control mode."""
    MANUAL = "manual"
    ASSISTED = "assisted"
    SUPERVISED = "supervised"


@dataclass
class RobotStatus:
    """Robot status information."""
    robot_id: str
    status: str
    battery_level: float
    safety_status: str
    current_pose: Dict[str, float]
    velocity: List[float]
    joint_positions: List[float]
    last_seen: float


@dataclass
class SafetyOverlay:
    """Safety overlay information."""
    velocity_limit: float
    force_limit: float
    safety_zone_violation: bool
    emergency_stop: bool
    proximity_warnings: List[Dict[str, Any]]


class OperatorConsole:
    """Main operator console."""
    
    def __init__(self):
        self.config = get_config()
        self.teleop_client: Optional[TeleopClient] = None
        self.current_session: Optional[str] = None
        self.robot_status: Optional[RobotStatus] = None
        self.safety_overlay: Optional[SafetyOverlay] = None
        self.operator_mode = OperatorMode.MANUAL
        self.running = False
        
        # Video processing
        self.video_frames: List[np.ndarray] = []
        self.max_video_frames = 10
        
        # Control state
        self.control_active = False
        self.last_command_time = 0.0
        self.command_rate = 10.0  # Hz
    
    async def start(self):
        """Start the operator console."""
        self.running = True
        logger.info("Operator console started")
    
    async def stop(self):
        """Stop the operator console."""
        self.running = False
        
        if self.teleop_client:
            await self.teleop_client.stop()
        
        logger.info("Operator console stopped")
    
    async def connect_to_robot(self, session_id: str, robot_id: str, operator_id: str, tenant_id: str):
        """Connect to a robot for teleoperation."""
        try:
            # Create teleop configuration
            config = TeleopConfig(
                session_id=session_id,
                robot_id=robot_id,
                operator_id=operator_id,
                tenant_id=tenant_id,
                signaling_url=f"ws://localhost:8001/ws/operator/{session_id}",
                turn_servers=[],
                stun_servers=[{"urls": "stun:stun.l.google.com:19302"}]
            )
            
            # Create and start teleop client
            self.teleop_client = TeleopClient(config)
            await self.teleop_client.start()
            
            self.current_session = session_id
            logger.info("Connected to robot", session_id=session_id, robot_id=robot_id)
            
        except Exception as e:
            logger.error("Failed to connect to robot", error=str(e))
            raise
    
    async def disconnect_from_robot(self):
        """Disconnect from current robot."""
        if self.teleop_client:
            await self.teleop_client.stop()
            self.teleop_client = None
        
        self.current_session = None
        self.robot_status = None
        logger.info("Disconnected from robot")
    
    async def send_velocity_command(self, linear: List[float], angular: List[float]):
        """Send velocity command to robot."""
        if not self.teleop_client or not self.teleop_client.is_connected():
            logger.warning("Not connected to robot")
            return
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_command_time < 1.0 / self.command_rate:
            return
        
        await self.teleop_client.send_velocity_command(linear, angular)
        self.last_command_time = current_time
        
        logger.debug("Velocity command sent", linear=linear, angular=angular)
    
    async def send_position_command(self, position: List[float], orientation: List[float]):
        """Send position command to robot."""
        if not self.teleop_client or not self.teleop_client.is_connected():
            logger.warning("Not connected to robot")
            return
        
        await self.teleop_client.send_position_command(position, orientation)
        logger.debug("Position command sent", position=position, orientation=orientation)
    
    async def send_emergency_stop(self):
        """Send emergency stop command."""
        if not self.teleop_client:
            logger.warning("Not connected to robot")
            return
        
        await self.teleop_client.send_emergency_stop()
        logger.warning("Emergency stop sent")
    
    def set_operator_mode(self, mode: OperatorMode):
        """Set operator control mode."""
        self.operator_mode = mode
        logger.info("Operator mode changed", mode=mode.value)
    
    def get_robot_status(self) -> Optional[RobotStatus]:
        """Get current robot status."""
        return self.robot_status
    
    def get_safety_overlay(self) -> Optional[SafetyOverlay]:
        """Get safety overlay information."""
        return self.safety_overlay
    
    def is_connected(self) -> bool:
        """Check if connected to robot."""
        return self.teleop_client is not None and self.teleop_client.is_connected()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status."""
        return {
            "connected": self.is_connected(),
            "session_id": self.current_session,
            "operator_mode": self.operator_mode.value,
            "control_active": self.control_active
        }


class JoystickController:
    """Joystick controller for teleoperation."""
    
    def __init__(self, console: OperatorConsole):
        self.console = console
        self.linear_velocity = [0.0, 0.0, 0.0]
        self.angular_velocity = [0.0, 0.0, 0.0]
        self.max_linear_velocity = 1.0
        self.max_angular_velocity = 1.0
        self.deadzone = 0.1
    
    def update_joystick_input(self, x: float, y: float, z: float, rx: float, ry: float, rz: float):
        """Update joystick input."""
        # Apply deadzone
        if abs(x) < self.deadzone:
            x = 0.0
        if abs(y) < self.deadzone:
            y = 0.0
        if abs(z) < self.deadzone:
            z = 0.0
        if abs(rx) < self.deadzone:
            rx = 0.0
        if abs(ry) < self.deadzone:
            ry = 0.0
        if abs(rz) < self.deadzone:
            rz = 0.0
        
        # Scale to max velocity
        self.linear_velocity = [
            x * self.max_linear_velocity,
            y * self.max_linear_velocity,
            z * self.max_linear_velocity
        ]
        
        self.angular_velocity = [
            rx * self.max_angular_velocity,
            ry * self.max_angular_velocity,
            rz * self.max_angular_velocity
        ]
    
    async def send_velocity_command(self):
        """Send velocity command to robot."""
        if any(abs(v) > 0.0 for v in self.linear_velocity + self.angular_velocity):
            await self.console.send_velocity_command(self.linear_velocity, self.angular_velocity)


class VideoProcessor:
    """Video processing for operator console."""
    
    def __init__(self):
        self.frames: List[np.ndarray] = []
        self.max_frames = 10
        self.current_frame: Optional[np.ndarray] = None
    
    def process_frame(self, frame_data: bytes) -> np.ndarray:
        """Process incoming video frame."""
        try:
            # Decode frame
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.current_frame = frame
                self.frames.append(frame)
                
                # Keep only recent frames
                if len(self.frames) > self.max_frames:
                    self.frames.pop(0)
            
            return frame
            
        except Exception as e:
            logger.error("Error processing video frame", error=str(e))
            return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current video frame."""
        return self.current_frame
    
    def add_safety_overlay(self, frame: np.ndarray, safety_info: SafetyOverlay) -> np.ndarray:
        """Add safety overlay to frame."""
        if frame is None:
            return frame
        
        overlay = frame.copy()
        
        # Add velocity limit indicator
        if safety_info.velocity_limit < 1.0:
            cv2.putText(overlay, f"VEL LIMIT: {safety_info.velocity_limit:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add emergency stop indicator
        if safety_info.emergency_stop:
            cv2.putText(overlay, "EMERGENCY STOP", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Add safety zone violation
        if safety_info.safety_zone_violation:
            cv2.putText(overlay, "SAFETY ZONE VIOLATION", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return overlay


# FastAPI app for operator console
app = FastAPI(title="Operator Console", version="0.1.0")

# Global console instance
console = OperatorConsole()
joystick_controller = JoystickController(console)
video_processor = VideoProcessor()


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    await console.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    await console.stop()


@app.get("/")
async def get_console():
    """Get operator console HTML."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eidolon Operator Console</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .status { background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .video-container { position: relative; width: 640px; height: 480px; border: 1px solid #ccc; }
            .controls { margin: 20px 0; }
            .joystick { width: 200px; height: 200px; border: 2px solid #333; border-radius: 50%; position: relative; }
            .emergency-stop { background: red; color: white; padding: 20px; font-size: 24px; border: none; border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Eidolon Operator Console</h1>
            
            <div class="status">
                <h3>Connection Status</h3>
                <p id="connection-status">Disconnected</p>
            </div>
            
            <div class="video-container">
                <video id="robot-video" width="640" height="480" autoplay></video>
            </div>
            
            <div class="controls">
                <h3>Controls</h3>
                <div class="joystick" id="joystick"></div>
                <button class="emergency-stop" onclick="emergencyStop()">EMERGENCY STOP</button>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket('ws://localhost:8002/ws');
            
            ws.onopen = function() {
                console.log('Connected to operator console');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                console.log('Received:', data);
            };
            
            function emergencyStop() {
                ws.send(JSON.stringify({type: 'emergency_stop'}));
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for operator console."""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "connect":
                await console.connect_to_robot(
                    message["session_id"],
                    message["robot_id"],
                    message["operator_id"],
                    message["tenant_id"]
                )
            
            elif message["type"] == "velocity_command":
                await joystick_controller.send_velocity_command()
            
            elif message["type"] == "emergency_stop":
                await console.send_emergency_stop()
            
            # Send status update
            status = console.get_connection_status()
            await websocket.send_text(json.dumps({
                "type": "status_update",
                "data": status
            }))
            
    except WebSocketDisconnect:
        logger.info("Operator console disconnected")
    except Exception as e:
        logger.error("Operator console error", error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
