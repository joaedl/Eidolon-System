"""Main robot system coordinator."""

import asyncio
import time
import signal
import sys
from typing import Dict, Any, Optional, List
import structlog

from .config import RobotConfig, RobotConfigManager
from .hardware import HardwareManager
from .safety import SafetyController, SafetyLimits
from .control import ControlManager, ControlMode, ControlCommand
from .perception import PerceptionPipeline
from .brain_client import BrainClient

logger = structlog.get_logger(__name__)


class RobotSystem:
    """Main robot system coordinator."""
    
    def __init__(self, config_name: str = "default", config_dir: str = "config/robots"):
        self.config_manager = RobotConfigManager(config_dir)
        self.robot_config: Optional[RobotConfig] = None
        self.hardware_manager: Optional[HardwareManager] = None
        self.safety_controller: Optional[SafetyController] = None
        self.control_manager: Optional[ControlManager] = None
        self.perception_pipeline: Optional[PerceptionPipeline] = None
        self.brain_client: Optional[BrainClient] = None
        
        self.running = False
        self.config_name = config_name
        
        # System status
        self.system_status = {
            "initialized": False,
            "hardware_ready": False,
            "safety_ok": True,
            "control_active": False,
            "perception_active": False,
            "cloud_connected": False
        }
    
    async def initialize(self):
        """Initialize the robot system."""
        logger.info("Initializing robot system", config=self.config_name)
        
        try:
            # Load robot configuration
            self.robot_config = self.config_manager.load_config(self.config_name)
            logger.info("Robot config loaded", robot_id=self.robot_config.robot_id)
            
            # Initialize hardware manager
            self.hardware_manager = HardwareManager(self.robot_config)
            await self.hardware_manager.initialize()
            self.system_status["hardware_ready"] = True
            
            # Initialize safety controller
            safety_limits = SafetyLimits(
                max_velocity=max(joint.max_velocity for arm in self.robot_config.arms for joint in arm.joints),
                max_acceleration=2.0,
                safety_zone_radius=2.0
            )
            self.safety_controller = SafetyController(safety_limits)
            await self.safety_controller.start()
            
            # Initialize control manager
            all_joint_names = []
            for arm in self.robot_config.arms:
                all_joint_names.extend([joint.name for joint in arm.joints])
            if self.robot_config.head:
                all_joint_names.extend([joint.name for joint in self.robot_config.head.joints])
            
            self.control_manager = ControlManager(all_joint_names)
            await self.control_manager.start()
            
            # Initialize perception pipeline
            self.perception_pipeline = PerceptionPipeline()
            await self.perception_pipeline.start()
            
            # Add cameras to perception pipeline
            if self.robot_config.head:
                for camera_config in self.robot_config.head.cameras:
                    await self.perception_pipeline.add_camera(
                        camera_config.camera_id,
                        camera_config.width,
                        camera_config.height
                    )
            
            # Initialize brain client if cloud enabled
            if self.robot_config.cloud_enabled:
                self.brain_client = BrainClient(
                    self.robot_config.robot_id,
                    "tenant-001"  # This should come from config
                )
                await self.brain_client.initialize()
                await self.brain_client.start()
                
                # Present robot configuration to cloud server
                if self.hardware_manager:
                    capabilities = self.hardware_manager.get_robot_capabilities()
                    await self.brain_client.present_robot_config(capabilities)
                
                self.system_status["cloud_connected"] = True
            
            self.system_status["initialized"] = True
            logger.info("Robot system initialized successfully", robot_id=self.robot_config.robot_id)
            
        except Exception as e:
            logger.error("Failed to initialize robot system", error=str(e))
            raise
    
    async def start(self):
        """Start the robot system."""
        if not self.system_status["initialized"]:
            await self.initialize()
        
        self.running = True
        logger.info("Robot system started", robot_id=self.robot_config.robot_id)
    
    async def stop(self):
        """Stop the robot system."""
        self.running = False
        
        # Stop all subsystems
        if self.brain_client:
            await self.brain_client.stop()
        
        if self.perception_pipeline:
            await self.perception_pipeline.stop()
        
        if self.control_manager:
            await self.control_manager.stop()
        
        if self.safety_controller:
            await self.safety_controller.stop()
        
        if self.hardware_manager:
            await self.hardware_manager.shutdown()
        
        logger.info("Robot system stopped", robot_id=self.robot_config.robot_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        status = self.system_status.copy()
        
        if self.hardware_manager:
            status["hardware_ready"] = self.hardware_manager.is_hardware_ready()
            status["joint_count"] = len(self.hardware_manager.motor_controllers)
            status["camera_count"] = len(self.hardware_manager.camera_interfaces)
            status["sensor_count"] = len(self.hardware_manager.sensor_interfaces)
        
        if self.safety_controller:
            safety_status = self.safety_controller.get_status()
            status["safety_ok"] = not safety_status.emergency_stop
            status["safety_state"] = safety_status.state.value
        
        if self.control_manager:
            status["control_active"] = self.control_manager.is_ready()
            status["control_mode"] = self.control_manager.control_mode.value
        
        if self.perception_pipeline:
            status["perception_active"] = True
            detections = self.perception_pipeline.get_latest_detections()
            status["detection_count"] = len(detections)
        
        return status
    
    def get_joint_states(self) -> List[Dict[str, Any]]:
        """Get all joint states."""
        if not self.hardware_manager:
            return []
        
        joint_states = self.hardware_manager.get_joint_states()
        return [
            {
                "name": state.name,
                "position": state.position,
                "velocity": state.velocity,
                "torque": state.torque,
                "temperature": state.temperature,
                "timestamp": state.timestamp
            }
            for state in joint_states
        ]
    
    def get_camera_frames(self) -> List[Dict[str, Any]]:
        """Get all camera frames."""
        if not self.hardware_manager:
            return []
        
        frames = []
        for camera_id in self.hardware_manager.get_camera_ids():
            # This would be async in real implementation
            frame = asyncio.create_task(self.hardware_manager.get_camera_frame(camera_id))
            if frame:
                frames.append({
                    "camera_id": camera_id,
                    "timestamp": frame.timestamp,
                    "frame_id": frame.frame_id,
                    "has_depth": frame.depth is not None
                })
        
        return frames
    
    def get_sensor_data(self) -> List[Dict[str, Any]]:
        """Get all sensor data."""
        if not self.hardware_manager:
            return []
        
        sensor_data = []
        for sensor_id in self.hardware_manager.get_sensor_ids():
            # This would be async in real implementation
            data = asyncio.create_task(self.hardware_manager.get_sensor_data(sensor_id))
            if data:
                sensor_data.append({
                    "sensor_id": sensor_id,
                    "data": data.data,
                    "timestamp": data.timestamp
                })
        
        return sensor_data
    
    async def move_joint(self, joint_name: str, position: float):
        """Move a specific joint to position."""
        if not self.hardware_manager:
            logger.warning("Hardware not initialized")
            return
        
        await self.hardware_manager.set_joint_position(joint_name, position)
        logger.debug("Joint moved", joint=joint_name, position=position)
    
    async def move_arm(self, arm_id: str, positions: List[float]):
        """Move all joints in an arm."""
        if not self.hardware_manager:
            logger.warning("Hardware not initialized")
            return
        
        joint_names = self.hardware_manager.get_arm_joints(arm_id)
        if len(positions) != len(joint_names):
            logger.error("Position count mismatch", arm=arm_id, expected=len(joint_names), got=len(positions))
            return
        
        for joint_name, position in zip(joint_names, positions):
            await self.hardware_manager.set_joint_position(joint_name, position)
        
        logger.debug("Arm moved", arm=arm_id, positions=positions)
    
    async def move_head(self, positions: List[float]):
        """Move head joints."""
        if not self.hardware_manager:
            logger.warning("Hardware not initialized")
            return
        
        joint_names = self.hardware_manager.get_head_joints()
        if len(positions) != len(joint_names):
            logger.error("Position count mismatch", expected=len(joint_names), got=len(positions))
            return
        
        for joint_name, position in zip(joint_names, positions):
            await self.hardware_manager.set_joint_position(joint_name, position)
        
        logger.debug("Head moved", positions=positions)
    
    async def emergency_stop(self):
        """Emergency stop all motors."""
        if self.safety_controller:
            self.safety_controller.set_emergency_stop(True)
        
        if self.control_manager:
            self.control_manager.clear_command_queue()
        
        logger.critical("Emergency stop activated")
    
    async def home_robot(self):
        """Move robot to home position."""
        if not self.hardware_manager:
            logger.warning("Hardware not initialized")
            return
        
        # Move all joints to home position
        for controller in self.hardware_manager.motor_controllers.values():
            await controller.set_position(controller.joint_config.home_position)
        
        logger.info("Robot moved to home position")


async def main():
    """Main entry point for robot system."""
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get config name from command line or environment
    config_name = "default"
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    
    # Create robot system
    robot_system = RobotSystem(config_name)
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        asyncio.create_task(robot_system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start robot system
        await robot_system.start()
        
        # Keep running until shutdown
        while robot_system.running:
            await asyncio.sleep(1.0)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error("Robot system error", error=str(e))
        sys.exit(1)
    finally:
        # Ensure clean shutdown
        await robot_system.stop()


if __name__ == "__main__":
    asyncio.run(main())
