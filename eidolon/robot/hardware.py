"""Hardware abstraction layer for robot components."""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import structlog
import numpy as np
import cv2

from .config import RobotConfig, JointConfig, CameraConfig, SensorConfig, MotorType, CameraType, SensorType
from .feetech import FeetechServoBus, FeetechConfig

logger = structlog.get_logger(__name__)


@dataclass
class JointState:
    """Joint state information."""
    name: str
    position: float
    velocity: float
    torque: float
    temperature: float
    timestamp: float


@dataclass
class CameraFrame:
    """Camera frame data."""
    camera_id: str
    image: np.ndarray
    depth: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame_id: int = 0


@dataclass
class SensorData:
    """Sensor data."""
    sensor_id: str
    data: Dict[str, float]
    timestamp: float = 0.0


class MotorController:
    """Abstract motor controller interface."""
    
    def __init__(self, joint_config: JointConfig):
        self.joint_config = joint_config
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.current_torque = 0.0
        self.current_temperature = 25.0
        self.enabled = False
        self.target_position = 0.0
        self.target_velocity = 0.0
        self.target_torque = 0.0


class FeetechServoController(MotorController):
    """Feetech servo motor controller for STS3215 servos."""
    
    def __init__(self, joint_config: JointConfig, servo_bus: Optional[FeetechServoBus] = None):
        super().__init__(joint_config)
        self.servo_bus = servo_bus
        self.resolution = 4096  # STS3215 resolution
        self.servo_id = getattr(joint_config, 'servo_id', None)
        
    async def initialize(self):
        """Initialize the Feetech servo controller."""
        logger.info("Initializing Feetech servo controller", joint=self.joint_config.name, servo_id=self.servo_id)
        
        if self.servo_bus and self.servo_id is not None:
            # Set angle limits
            min_angle = self.joint_config.min_position
            max_angle = self.joint_config.max_position
            if hasattr(self.servo_bus, 'set_angle_limits'):
                self.servo_bus.set_angle_limits(self.servo_id, min_angle, max_angle)
        
        self.enabled = True
        logger.info("Feetech servo controller initialized", joint=self.joint_config.name)
    
    async def enable(self):
        """Enable the servo motor."""
        if self.servo_bus and self.servo_id is not None:
            if hasattr(self.servo_bus, 'enable_torque'):
                success = self.servo_bus.enable_torque(self.servo_id, True)
                if success:
                    self.enabled = True
                    logger.debug("Feetech servo enabled", joint=self.joint_config.name, servo_id=self.servo_id)
                else:
                    logger.warning("Failed to enable Feetech servo", joint=self.joint_config.name, servo_id=self.servo_id)
            else:
                self.enabled = True
        else:
            self.enabled = True
            logger.debug("Feetech servo enabled (mock)", joint=self.joint_config.name)
    
    async def disable(self):
        """Disable the servo motor."""
        if self.servo_bus and self.servo_id is not None:
            if hasattr(self.servo_bus, 'enable_torque'):
                success = self.servo_bus.enable_torque(self.servo_id, False)
                if success:
                    self.enabled = False
                    logger.debug("Feetech servo disabled", joint=self.joint_config.name, servo_id=self.servo_id)
                else:
                    logger.warning("Failed to disable Feetech servo", joint=self.joint_config.name, servo_id=self.servo_id)
            else:
                self.enabled = False
        else:
            self.enabled = False
            logger.debug("Feetech servo disabled (mock)", joint=self.joint_config.name)
    
    async def set_position(self, position: float):
        """Set target position in degrees."""
        if not self.enabled:
            logger.warning("Servo not enabled", joint=self.joint_config.name)
            return
        
        # Clamp to limits
        position = np.clip(position, self.joint_config.min_position, self.joint_config.max_position)
        self.target_position = position
        
        # Apply offset and direction if available
        adjusted_position = position
        if hasattr(self.joint_config, 'offset'):
            adjusted_position += self.joint_config.offset
        if hasattr(self.joint_config, 'direction'):
            adjusted_position *= self.joint_config.direction
        
        # Send to Feetech servo
        if self.servo_bus and self.servo_id is not None:
            if hasattr(self.servo_bus, 'set_position'):
                success = self.servo_bus.set_position(self.servo_id, adjusted_position)
                if not success:
                    logger.warning("Failed to set servo position", joint=self.joint_config.name, servo_id=self.servo_id)
        
        # Update current position for simulation
        await self._update_position()
    
    async def set_velocity(self, velocity: float):
        """Set target velocity in degrees/second."""
        if not self.enabled:
            logger.warning("Servo not enabled", joint=self.joint_config.name)
            return
        
        # Clamp to limits
        velocity = np.clip(velocity, -self.joint_config.max_velocity, self.joint_config.max_velocity)
        self.target_velocity = velocity
        
        # Convert to Feetech speed format (0-1023, where 0 = max speed)
        if self.servo_bus and self.servo_id is not None:
            if hasattr(self.servo_bus, 'set_speed'):
                speed = int((self.joint_config.max_velocity - abs(velocity)) / self.joint_config.max_velocity * 1023)
                success = self.servo_bus.set_speed(self.servo_id, speed)
                if not success:
                    logger.warning("Failed to set servo speed", joint=self.joint_config.name, servo_id=self.servo_id)
    
    async def set_torque(self, torque: float):
        """Set target torque as percentage of maximum."""
        if not self.enabled:
            logger.warning("Servo not enabled", joint=self.joint_config.name)
            return
        
        # Clamp to limits
        torque = np.clip(torque, -self.joint_config.max_torque, self.joint_config.max_torque)
        self.target_torque = torque
        
        # Convert to Feetech torque format
        if self.servo_bus and self.servo_id is not None:
            if hasattr(self.servo_bus, 'set_torque_limit'):
                torque_percent = int(abs(torque) / self.joint_config.max_torque * 100)
                success = self.servo_bus.set_torque_limit(self.servo_id, torque_percent)
                if not success:
                    logger.warning("Failed to set servo torque", joint=self.joint_config.name, servo_id=self.servo_id)
    
    async def _update_position(self):
        """Update position based on target."""
        # Read actual position from servo if available
        if self.servo_bus and self.servo_id is not None:
            if hasattr(self.servo_bus, 'read_position'):
                position = self.servo_bus.read_position(self.servo_id)
                if position is not None:
                    # Apply inverse offset and direction
                    adjusted_position = position
                    if hasattr(self.joint_config, 'direction'):
                        adjusted_position /= self.joint_config.direction
                    if hasattr(self.joint_config, 'offset'):
                        adjusted_position -= self.joint_config.offset
                    
                    self.current_position = adjusted_position
                    return
        
        # Fallback to simulation
        error = self.target_position - self.current_position
        if abs(error) > 0.01:  # 0.01 degree threshold
            # Move towards target
            direction = 1 if error > 0 else -1
            self.current_position += direction * 0.1  # 0.1 degree/step
            self.current_velocity = direction * 0.1
        else:
            self.current_velocity = 0.0
    
    def get_state(self) -> JointState:
        """Get current joint state."""
        return JointState(
            name=self.joint_config.name,
            position=self.current_position,
            velocity=self.current_velocity,
            torque=self.current_torque,
            temperature=self.current_temperature,
            timestamp=time.time()
        )
    
    async def enable(self):
        """Enable the motor."""
        self.enabled = True
        logger.debug("Motor enabled", joint=self.joint_config.name)
    
    async def disable(self):
        """Disable the motor."""
        self.enabled = False
        logger.debug("Motor disabled", joint=self.joint_config.name)
    
    async def set_position(self, position: float):
        """Set target position."""
        if not self.enabled:
            logger.warning("Motor not enabled", joint=self.joint_config.name)
            return
        
        # Clamp to limits
        position = np.clip(position, self.joint_config.min_position, self.joint_config.max_position)
        self.target_position = position
        
        # Simulate position control
        await self._update_position()
    
    async def set_velocity(self, velocity: float):
        """Set target velocity."""
        if not self.enabled:
            logger.warning("Motor not enabled", joint=self.joint_config.name)
            return
        
        # Clamp to limits
        velocity = np.clip(velocity, -self.joint_config.max_velocity, self.joint_config.max_velocity)
        self.target_velocity = velocity
    
    async def set_torque(self, torque: float):
        """Set target torque."""
        if not self.enabled:
            logger.warning("Motor not enabled", joint=self.joint_config.name)
            return
        
        # Clamp to limits
        torque = np.clip(torque, -self.joint_config.max_torque, self.joint_config.max_torque)
        self.target_torque = torque
    
    async def _update_position(self):
        """Update position based on target."""
        # Simple position control simulation
        error = self.target_position - self.current_position
        if abs(error) > 0.01:  # 0.01 rad threshold
            # Move towards target
            direction = 1 if error > 0 else -1
            self.current_position += direction * 0.1  # 0.1 rad/step
            self.current_velocity = direction * 0.1
        else:
            self.current_velocity = 0.0
    
    def get_state(self) -> JointState:
        """Get current joint state."""
        return JointState(
            name=self.joint_config.name,
            position=self.current_position,
            velocity=self.current_velocity,
            torque=self.current_torque,
            temperature=self.current_temperature,
            timestamp=time.time()
        )


class CameraInterface:
    """Camera interface for different camera types."""
    
    def __init__(self, camera_config: CameraConfig):
        self.camera_config = camera_config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[CameraFrame] = None
    
    async def initialize(self):
        """Initialize the camera."""
        logger.info("Initializing camera", camera=self.camera_config.name)
        
        # Try to open camera
        self.cap = cv2.VideoCapture(self.camera_config.device_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_config.name}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
        
        # Start capture thread
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
    
    def _capture_loop(self):
        """Camera capture loop."""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_count += 1
                    
                    # Process frame based on camera type
                    if self.camera_config.camera_type == CameraType.RGBD:
                        # Split RGB and depth channels
                        rgb = frame[:, :, :3]
                        depth = frame[:, :, 3] if frame.shape[2] > 3 else None
                        
                        self._latest_frame = CameraFrame(
                            camera_id=self.camera_config.camera_id,
                            image=rgb,
                            depth=depth,
                            timestamp=time.time(),
                            frame_id=self.frame_count
                        )
                    else:
                        self._latest_frame = CameraFrame(
                            camera_id=self.camera_config.camera_id,
                            image=frame,
                            timestamp=time.time(),
                            frame_id=self.frame_count
                        )
                
                time.sleep(1.0 / self.camera_config.fps)
            except Exception as e:
                logger.error("Camera capture error", camera=self.camera_config.name, error=str(e))
                time.sleep(0.1)
    
    async def get_frame(self) -> Optional[CameraFrame]:
        """Get latest camera frame."""
        return self._latest_frame
    
    async def stop(self):
        """Stop camera capture."""
        self.running = False
        if self._capture_thread:
            self._capture_thread.join()
        if self.cap:
            self.cap.release()
        logger.info("Camera stopped", camera=self.camera_config.name)


class SensorInterface:
    """Sensor interface for different sensor types."""
    
    def __init__(self, sensor_config: SensorConfig):
        self.sensor_config = sensor_config
        self.running = False
        self._latest_data: Optional[SensorData] = None
        self._update_thread: Optional[threading.Thread] = None
    
    async def initialize(self):
        """Initialize the sensor."""
        logger.info("Initializing sensor", sensor=self.sensor_config.name)
        
        # Start update thread
        self.running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
    
    def _update_loop(self):
        """Sensor update loop."""
        while self.running:
            try:
                # Simulate sensor data based on type
                if self.sensor_config.sensor_type == SensorType.IMU:
                    data = {
                        "linear_acceleration": [np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(-9.81, 0.1)],
                        "angular_velocity": [np.random.normal(0, 0.01), np.random.normal(0, 0.01), np.random.normal(0, 0.01)],
                        "orientation": [0, 0, 0, 1]  # quaternion
                    }
                elif self.sensor_config.sensor_type == SensorType.FORCE_TORQUE:
                    data = {
                        "force": [np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(0, 0.1)],
                        "torque": [np.random.normal(0, 0.01), np.random.normal(0, 0.01), np.random.normal(0, 0.01)]
                    }
                else:
                    data = {"value": np.random.normal(0, 0.1)}
                
                self._latest_data = SensorData(
                    sensor_id=self.sensor_config.sensor_id,
                    data=data,
                    timestamp=time.time()
                )
                
                time.sleep(1.0 / self.sensor_config.update_rate)
            except Exception as e:
                logger.error("Sensor update error", sensor=self.sensor_config.name, error=str(e))
                time.sleep(0.1)
    
    async def get_data(self) -> Optional[SensorData]:
        """Get latest sensor data."""
        return self._latest_data
    
    async def stop(self):
        """Stop sensor updates."""
        self.running = False
        if self._update_thread:
            self._update_thread.join()
        logger.info("Sensor stopped", sensor=self.sensor_config.name)


class HardwareManager:
    """Hardware manager for robot components."""
    
    def __init__(self, robot_config: RobotConfig):
        self.robot_config = robot_config
        self.motor_controllers: Dict[str, MotorController] = {}
        self.camera_interfaces: Dict[str, CameraInterface] = {}
        self.sensor_interfaces: Dict[str, SensorInterface] = {}
        self.servo_bus: Optional[FeetechServoBus] = None
        self.running = False
    
    async def initialize(self):
        """Initialize all hardware components."""
        logger.info("Initializing hardware components", robot=self.robot_config.robot_id)
        
        # Initialize servo bus if configured
        if hasattr(self.robot_config, 'servo_bus') and self.robot_config.servo_bus:
            servo_config = FeetechConfig(
                port=self.robot_config.servo_bus.get('port', '/dev/ttyUSB0'),
                baudrate=self.robot_config.servo_bus.get('baudrate', 1000000),
                mock=self.robot_config.servo_bus.get('mock', False)
            )
            self.servo_bus = FeetechServoBus(servo_config)
            if not self.servo_bus.connect():
                logger.warning("Failed to connect to servo bus, using mock mode")
                self.servo_bus = None
        
        # Initialize motor controllers
        for arm in self.robot_config.arms:
            for joint in arm.joints:
                # Check if this is a Feetech servo
                if hasattr(joint, 'servo_id') and joint.servo_id is not None:
                    controller = FeetechServoController(joint, self.servo_bus)
                else:
                    controller = MotorController(joint)
                await controller.initialize()
                self.motor_controllers[joint.name] = controller
        
        if self.robot_config.head:
            for joint in self.robot_config.head.joints:
                if hasattr(joint, 'servo_id') and joint.servo_id is not None:
                    controller = FeetechServoController(joint, self.servo_bus)
                else:
                    controller = MotorController(joint)
                await controller.initialize()
                self.motor_controllers[joint.name] = controller
        
        # Initialize cameras
        if self.robot_config.head:
            for camera_config in self.robot_config.head.cameras:
                camera = CameraInterface(camera_config)
                await camera.initialize()
                self.camera_interfaces[camera_config.camera_id] = camera
        
        # Initialize sensors
        for sensor_config in self.robot_config.base_sensors:
            sensor = SensorInterface(sensor_config)
            await sensor.initialize()
            self.sensor_interfaces[sensor_config.sensor_id] = sensor
        
        self.running = True
        logger.info("Hardware initialization completed", robot=self.robot_config.robot_id)
    
    async def shutdown(self):
        """Shutdown all hardware components."""
        logger.info("Shutting down hardware components", robot=self.robot_config.robot_id)
        
        # Stop all components
        for camera in self.camera_interfaces.values():
            await camera.stop()
        
        for sensor in self.sensor_interfaces.values():
            await sensor.stop()
        
        for controller in self.motor_controllers.values():
            await controller.disable()
        
        self.running = False
        logger.info("Hardware shutdown completed", robot=self.robot_config.robot_id)
    
    def get_joint_states(self) -> List[JointState]:
        """Get all joint states."""
        return [controller.get_state() for controller in self.motor_controllers.values()]
    
    def get_joint_state(self, joint_name: str) -> Optional[JointState]:
        """Get specific joint state."""
        controller = self.motor_controllers.get(joint_name)
        return controller.get_state() if controller else None
    
    async def set_joint_position(self, joint_name: str, position: float):
        """Set joint position."""
        controller = self.motor_controllers.get(joint_name)
        if controller:
            await controller.set_position(position)
        else:
            logger.warning("Joint not found", joint=joint_name)
    
    async def set_joint_velocity(self, joint_name: str, velocity: float):
        """Set joint velocity."""
        controller = self.motor_controllers.get(joint_name)
        if controller:
            await controller.set_velocity(velocity)
        else:
            logger.warning("Joint not found", joint=joint_name)
    
    async def set_joint_torque(self, joint_name: str, torque: float):
        """Set joint torque."""
        controller = self.motor_controllers.get(joint_name)
        if controller:
            await controller.set_torque(torque)
        else:
            logger.warning("Joint not found", joint=joint_name)
    
    async def get_camera_frame(self, camera_id: str) -> Optional[CameraFrame]:
        """Get camera frame."""
        camera = self.camera_interfaces.get(camera_id)
        return await camera.get_frame() if camera else None
    
    async def get_sensor_data(self, sensor_id: str) -> Optional[SensorData]:
        """Get sensor data."""
        sensor = self.sensor_interfaces.get(sensor_id)
        return await sensor.get_data() if sensor else None
    
    def get_arm_joints(self, arm_id: str) -> List[str]:
        """Get joint names for an arm."""
        for arm in self.robot_config.arms:
            if arm.arm_id == arm_id:
                return [joint.name for joint in arm.joints]
        return []
    
    def get_head_joints(self) -> List[str]:
        """Get head joint names."""
        if self.robot_config.head:
            return [joint.name for joint in self.robot_config.head.joints]
        return []
    
    def get_camera_ids(self) -> List[str]:
        """Get all camera IDs."""
        return list(self.camera_interfaces.keys())
    
    def get_sensor_ids(self) -> List[str]:
        """Get all sensor IDs."""
        return list(self.sensor_interfaces.keys())
    
    def is_hardware_ready(self) -> bool:
        """Check if hardware is ready."""
        return self.running and len(self.motor_controllers) > 0
    
    def get_robot_capabilities(self) -> Dict[str, Any]:
        """Get robot capabilities for presentation to cloud server."""
        capabilities = {
            "robot_info": {
                "name": getattr(self.robot_config, 'name', 'Unknown Robot'),
                "type": getattr(self.robot_config, 'type', 'unknown'),
                "version": getattr(self.robot_config, 'version', '1.0.0'),
                "description": getattr(self.robot_config, 'description', ''),
                "hardware_ready": self.is_hardware_ready()
            },
            "joints": {
                "count": len(self.motor_controllers),
                "names": list(self.motor_controllers.keys()),
                "types": {}
            },
            "cameras": {
                "count": len(self.camera_interfaces),
                "ids": list(self.camera_interfaces.keys()),
                "types": {}
            },
            "sensors": {
                "count": len(self.sensor_interfaces),
                "ids": list(self.sensor_interfaces.keys()),
                "types": {}
            },
            "capabilities": {
                "low_level": getattr(self.robot_config, 'capabilities', {}).get('low_level', []),
                "high_level": getattr(self.robot_config, 'capabilities', {}).get('high_level', []),
                "perception": getattr(self.robot_config, 'capabilities', {}).get('perception', []),
                "safety": getattr(self.robot_config, 'capabilities', {}).get('safety', [])
            },
            "control_modes": getattr(self.robot_config, 'control', {}).get('modes', ['position']),
            "safety_features": {
                "emergency_stop": getattr(self.robot_config, 'safety', {}).get('emergency_stop', {}),
                "joint_limits": getattr(self.robot_config, 'safety', {}).get('joint_limits', {}),
                "collision_detection": getattr(self.robot_config, 'safety', {}).get('collision_detection', {})
            },
            "kinematics": {
                "arm_lengths": getattr(self.robot_config, 'kinematics', {}).get('arm_lengths', {}),
                "workspace": getattr(self.robot_config, 'kinematics', {}).get('workspace', {}),
                "base_frame": getattr(self.robot_config, 'kinematics', {}).get('base_frame', 'base_link')
            }
        }
        
        # Add joint details
        for joint_name, controller in self.motor_controllers.items():
            joint_config = controller.joint_config
            capabilities["joints"]["types"][joint_name] = {
                "type": getattr(joint_config, 'joint_type', 'revolute'),
                "position_limits": [joint_config.min_position, joint_config.max_position],
                "velocity_limit": joint_config.max_velocity,
                "torque_limit": joint_config.max_torque,
                "servo_id": getattr(joint_config, 'servo_id', None),
                "model": getattr(joint_config, 'model', 'unknown')
            }
        
        # Add camera details
        for camera_id, camera in self.camera_interfaces.items():
            camera_config = camera.camera_config
            capabilities["cameras"]["types"][camera_id] = {
                "type": camera_config.camera_type.value,
                "resolution": [camera_config.width, camera_config.height],
                "fps": camera_config.fps,
                "position": getattr(camera_config, 'position', 'unknown')
            }
        
        # Add sensor details
        for sensor_id, sensor in self.sensor_interfaces.items():
            sensor_config = sensor.sensor_config
            capabilities["sensors"]["types"][sensor_id] = {
                "type": sensor_config.sensor_type.value,
                "update_rate": sensor_config.update_rate,
                "position": getattr(sensor_config, 'position', 'unknown')
            }
        
        return capabilities
