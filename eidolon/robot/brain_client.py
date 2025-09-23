"""Brain Client for robot-cloud communication."""

import asyncio
import time
import grpc
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import structlog

from ..common.config import get_config
from ..common.security import SecurityManager
from .safety import SafetyController, SafetyLimits
from .control import ControlManager, ControlMode, ControlCommand
from .perception import PerceptionPipeline, ObjectDetection, Affordance

logger = structlog.get_logger(__name__)


@dataclass
class TelemetryData:
    """Telemetry data to send to cloud."""
    device_id: str
    timestamp: float
    metrics: Dict[str, float]
    keyframe_jpeg: Optional[bytes] = None
    rosbag_chunk_url: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, str] = None


class BrainClient:
    """Brain client for robot-cloud communication."""
    
    def __init__(self, robot_id: str, tenant_id: str):
        self.robot_id = robot_id
        self.tenant_id = tenant_id
        self.config = get_config()
        self.security_manager = SecurityManager(self.config)
        
        # Robot subsystems
        self.safety_controller: Optional[SafetyController] = None
        self.control_manager: Optional[ControlManager] = None
        self.perception_pipeline: Optional[PerceptionPipeline] = None
        
        # Communication
        self.grpc_channel: Optional[grpc.aio.Channel] = None
        self.connected = False
        self.session_token: Optional[str] = None
        
        # Telemetry
        self.telemetry_queue: List[TelemetryData] = []
        self.telemetry_batch_size = 10
        self.telemetry_interval = 1.0  # seconds
        
        # Tasks
        self.running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._telemetry_task: Optional[asyncio.Task] = None
        self._subgoal_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the brain client."""
        logger.info("Initializing brain client", robot_id=self.robot_id)
        
        # Initialize robot subsystems
        await self._initialize_subsystems()
        
        # Initialize communication
        await self._initialize_communication()
        
        logger.info("Brain client initialized")
    
    async def _initialize_subsystems(self):
        """Initialize robot subsystems."""
        # Initialize safety controller
        safety_limits = SafetyLimits(
            max_velocity=self.config.robot.max_velocity,
            max_acceleration=self.config.robot.max_acceleration,
            safety_zone_radius=self.config.robot.safety_zone_radius
        )
        self.safety_controller = SafetyController(safety_limits)
        await self.safety_controller.start()
        
        # Initialize control manager
        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.control_manager = ControlManager(joint_names)
        await self.control_manager.start()
        
        # Initialize perception pipeline
        self.perception_pipeline = PerceptionPipeline()
        await self.perception_pipeline.start()
        await self.perception_pipeline.add_camera("main_camera")
        
        logger.info("Robot subsystems initialized")
    
    async def _initialize_communication(self):
        """Initialize communication with cloud."""
        try:
            # Get device identity
            device_identity = self.security_manager.get_device_identity()
            if not device_identity:
                raise RuntimeError("Device identity not available")
            
            # Create secure channel
            credentials = grpc.ssl_channel_credentials()
            self.grpc_channel = grpc.aio.secure_channel(
                f"{self.config.cloud.host}:{self.config.cloud.port}",
                credentials
            )
            
            # Register device
            await self._register_device()
            
            self.connected = True
            logger.info("Connected to cloud", robot_id=self.robot_id)
            
        except Exception as e:
            logger.error("Failed to initialize communication", error=str(e))
            raise
    
    async def _register_device(self):
        """Register device with cloud."""
        # This would use the actual gRPC service
        # For now, create a session token
        self.session_token = self.security_manager.create_device_token(
            self.robot_id, self.tenant_id
        )
        logger.info("Device registered with cloud")
    
    async def start(self):
        """Start the brain client."""
        self.running = True
        
        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._telemetry_task = asyncio.create_task(self._telemetry_loop())
        self._subgoal_task = asyncio.create_task(self._subgoal_loop())
        
        logger.info("Brain client started")
    
    async def stop(self):
        """Stop the brain client."""
        self.running = False
        
        # Cancel background tasks
        for task in [self._heartbeat_task, self._telemetry_task, self._subgoal_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop subsystems
        if self.safety_controller:
            await self.safety_controller.stop()
        if self.control_manager:
            await self.control_manager.stop()
        if self.perception_pipeline:
            await self.perception_pipeline.stop()
        
        # Close communication
        if self.grpc_channel:
            await self.grpc_channel.close()
        
        logger.info("Brain client stopped")
    
    async def _heartbeat_loop(self):
        """Send heartbeat to cloud."""
        while self.running:
            try:
                if self.connected and self.session_token:
                    await self._send_heartbeat()
                await asyncio.sleep(5.0)  # 5-second heartbeat
            except Exception as e:
                logger.error("Heartbeat error", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _send_heartbeat(self):
        """Send heartbeat to cloud."""
        # This would use the actual gRPC service
        # For now, just log
        logger.debug("Sending heartbeat", robot_id=self.robot_id)
    
    async def _telemetry_loop(self):
        """Send telemetry data to cloud."""
        while self.running:
            try:
                if self.connected:
                    await self._collect_and_send_telemetry()
                await asyncio.sleep(self.telemetry_interval)
            except Exception as e:
                logger.error("Telemetry error", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _collect_and_send_telemetry(self):
        """Collect and send telemetry data."""
        # Collect metrics from subsystems
        metrics = await self._collect_metrics()
        
        # Get latest perception data
        detections = self.perception_pipeline.get_latest_detections() if self.perception_pipeline else []
        affordances = self.perception_pipeline.get_latest_affordances() if self.perception_pipeline else []
        
        # Create telemetry data
        telemetry = TelemetryData(
            device_id=self.robot_id,
            timestamp=time.time(),
            metrics=metrics,
            metadata={
                "detections_count": str(len(detections)),
                "affordances_count": str(len(affordances)),
                "safety_state": self.safety_controller.status.state.value if self.safety_controller else "unknown",
                "control_mode": self.control_manager.control_mode.value if self.control_manager else "unknown"
            }
        )
        
        # Add to queue
        self.telemetry_queue.append(telemetry)
        
        # Send batch if queue is full
        if len(self.telemetry_queue) >= self.telemetry_batch_size:
            await self._send_telemetry_batch()
    
    async def _collect_metrics(self) -> Dict[str, float]:
        """Collect metrics from robot subsystems."""
        metrics = {}
        
        # Safety metrics
        if self.safety_controller:
            safety_status = self.safety_controller.get_status()
            metrics.update({
                "safety_emergency_stop": float(safety_status.emergency_stop),
                "safety_communication_lost": float(safety_status.communication_lost),
                "safety_violations_count": float(len(safety_status.violations))
            })
        
        # Control metrics
        if self.control_manager:
            joint_states = self.control_manager.get_joint_states()
            if joint_states:
                metrics.update({
                    "joint_count": float(len(joint_states)),
                    "avg_joint_velocity": float(np.mean([js.velocity for js in joint_states])),
                    "max_joint_velocity": float(max([js.velocity for js in joint_states]))
                })
        
        # System metrics (simplified)
        metrics.update({
            "cpu_usage": 0.5,  # Would be actual CPU usage
            "memory_usage": 0.3,  # Would be actual memory usage
            "disk_usage": 0.2,  # Would be actual disk usage
            "network_usage": 0.1  # Would be actual network usage
        })
        
        return metrics
    
    async def _send_telemetry_batch(self):
        """Send telemetry batch to cloud."""
        if not self.telemetry_queue:
            return
        
        # This would use the actual gRPC streaming service
        # For now, just log and clear queue
        logger.debug("Sending telemetry batch", count=len(self.telemetry_queue))
        self.telemetry_queue.clear()
    
    async def _subgoal_loop(self):
        """Request and process subgoals from cloud."""
        while self.running:
            try:
                if self.connected and self.control_manager:
                    await self._request_subgoal()
                await asyncio.sleep(1.0)  # 1-second subgoal requests
            except Exception as e:
                logger.error("Subgoal error", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _request_subgoal(self):
        """Request subgoal from cloud planner."""
        # This would use the actual gRPC service
        # For now, just log
        logger.debug("Requesting subgoal", robot_id=self.robot_id)
    
    def add_control_command(self, command: ControlCommand):
        """Add a control command to the robot."""
        if self.control_manager:
            self.control_manager.add_command(command)
            logger.debug("Control command added", type=command.command_type)
    
    def set_control_mode(self, mode: ControlMode):
        """Set the robot control mode."""
        if self.control_manager:
            self.control_manager.set_control_mode(mode)
            logger.info("Control mode set", mode=mode.value)
    
    def get_robot_status(self) -> Dict[str, Any]:
        """Get current robot status."""
        status = {
            "robot_id": self.robot_id,
            "tenant_id": self.tenant_id,
            "connected": self.connected,
            "running": self.running
        }
        
        if self.safety_controller:
            safety_status = self.safety_controller.get_status()
            status.update({
                "safety_state": safety_status.state.value,
                "emergency_stop": safety_status.emergency_stop,
                "communication_lost": safety_status.communication_lost
            })
        
        if self.control_manager:
            status.update({
                "control_mode": self.control_manager.control_mode.value,
                "ready": self.control_manager.is_ready()
            })
        
        return status
    
    async def present_robot_config(self, capabilities: Dict[str, Any]):
        """Present robot configuration and capabilities to the cloud server."""
        if not self.connected:
            logger.warning("Not connected to cloud server, cannot present config")
            return
        
        try:
            # Create device info with robot capabilities
            device_info = {
                "device_id": self.robot_id,
                "device_type": "robot",
                "hardware_version": capabilities.get("robot_info", {}).get("version", "1.0.0"),
                "firmware_version": "1.0.0",
                "capabilities": capabilities,
                "status": "online",
                "last_seen": time.time()
            }
            
            # Send device registration with capabilities
            await self._send_device_registration(device_info)
            
            logger.info("Robot configuration presented to cloud server", 
                       robot_id=self.robot_id,
                       joint_count=capabilities.get("joints", {}).get("count", 0),
                       camera_count=capabilities.get("cameras", {}).get("count", 0),
                       sensor_count=capabilities.get("sensors", {}).get("count", 0))
            
        except Exception as e:
            logger.error("Failed to present robot config to cloud server", error=str(e))
    
    async def _send_device_registration(self, device_info: Dict[str, Any]):
        """Send device registration to cloud server."""
        # This would be implemented with actual gRPC call
        # For now, just log the registration
        logger.info("Device registration sent", device_id=device_info["device_id"])
