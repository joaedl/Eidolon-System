"""Robot configuration system for different hardware setups."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


class MotorType(Enum):
    """Motor control types."""
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    IMPEDANCE = "impedance"


class CameraType(Enum):
    """Camera types."""
    RGB = "rgb"
    RGBD = "rgbd"
    STEREO = "stereo"
    MONOCULAR = "monocular"


class SensorType(Enum):
    """Sensor types."""
    IMU = "imu"
    FORCE_TORQUE = "force_torque"
    TOUCH = "touch"
    PROXIMITY = "proximity"
    LIDAR = "lidar"
    SONAR = "sonar"


@dataclass
class JointConfig:
    """Joint configuration."""
    name: str
    joint_id: int
    motor_type: MotorType
    min_position: float = -3.14
    max_position: float = 3.14
    max_velocity: float = 1.0
    max_torque: float = 10.0
    gear_ratio: float = 1.0
    encoder_resolution: int = 4096
    home_position: float = 0.0
    safety_limits: Dict[str, float] = field(default_factory=dict)
    # Feetech servo specific fields
    servo_id: Optional[int] = None
    joint_type: str = "revolute"
    model: str = "unknown"
    offset: float = 0.0
    direction: int = 1


@dataclass
class CameraConfig:
    """Camera configuration."""
    name: str
    camera_id: str
    camera_type: CameraType
    device_path: str
    width: int = 640
    height: int = 480
    fps: int = 30
    depth_range: tuple = (0.1, 10.0)  # min, max depth in meters
    calibration_file: Optional[str] = None
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    orientation: Dict[str, float] = field(default_factory=lambda: {"rx": 0, "ry": 0, "rz": 0})


@dataclass
class SensorConfig:
    """Sensor configuration."""
    name: str
    sensor_id: str
    sensor_type: SensorType
    device_path: str
    update_rate: float = 100.0  # Hz
    calibration_file: Optional[str] = None
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    orientation: Dict[str, float] = field(default_factory=lambda: {"rx": 0, "ry": 0, "rz": 0})


@dataclass
class ArmConfig:
    """Arm configuration."""
    name: str
    arm_id: str
    joints: List[JointConfig]
    end_effector: Optional[str] = None
    base_position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    workspace_limits: Dict[str, float] = field(default_factory=dict)


@dataclass
class HeadConfig:
    """Head configuration."""
    name: str
    head_id: str
    joints: List[JointConfig]
    cameras: List[CameraConfig]
    base_position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})


@dataclass
class RobotConfig:
    """Complete robot configuration."""
    robot_id: str
    robot_name: str
    robot_type: str = "humanoid"
    manufacturer: str = "custom"
    model: str = "v1.0"
    
    # Hardware components
    arms: List[ArmConfig] = field(default_factory=list)
    head: Optional[HeadConfig] = None
    base_sensors: List[SensorConfig] = field(default_factory=list)
    
    # Communication
    cloud_enabled: bool = True
    local_processing: bool = True
    teleop_enabled: bool = True
    
    # Safety
    emergency_stop_enabled: bool = True
    collision_detection: bool = True
    workspace_limits: Dict[str, float] = field(default_factory=dict)
    
    # Performance
    control_frequency: float = 100.0  # Hz
    perception_frequency: float = 30.0  # Hz
    safety_frequency: float = 1000.0  # Hz
    
    # Network
    local_ip: str = "192.168.1.100"
    cloud_endpoint: str = "controller.eidolon.cloud:443"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "robot_id": self.robot_id,
            "robot_name": self.robot_name,
            "robot_type": self.robot_type,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "arms": [
                {
                    "name": arm.name,
                    "arm_id": arm.arm_id,
                    "joints": [
                        {
                            "name": joint.name,
                            "joint_id": joint.joint_id,
                            "motor_type": joint.motor_type.value,
                            "min_position": joint.min_position,
                            "max_position": joint.max_position,
                            "max_velocity": joint.max_velocity,
                            "max_torque": joint.max_torque,
                            "gear_ratio": joint.gear_ratio,
                            "encoder_resolution": joint.encoder_resolution,
                            "home_position": joint.home_position,
                            "safety_limits": joint.safety_limits
                        } for joint in arm.joints
                    ],
                    "end_effector": arm.end_effector,
                    "base_position": arm.base_position,
                    "workspace_limits": arm.workspace_limits
                } for arm in self.arms
            ],
            "head": {
                "name": self.head.name,
                "head_id": self.head.head_id,
                "joints": [
                    {
                        "name": joint.name,
                        "joint_id": joint.joint_id,
                        "motor_type": joint.motor_type.value,
                        "min_position": joint.min_position,
                        "max_position": joint.max_position,
                        "max_velocity": joint.max_velocity,
                        "max_torque": joint.max_torque,
                        "gear_ratio": joint.gear_ratio,
                        "encoder_resolution": joint.encoder_resolution,
                        "home_position": joint.home_position,
                        "safety_limits": joint.safety_limits
                    } for joint in self.head.joints
                ],
                "cameras": [
                    {
                        "name": camera.name,
                        "camera_id": camera.camera_id,
                        "camera_type": camera.camera_type.value,
                        "device_path": camera.device_path,
                        "width": camera.width,
                        "height": camera.height,
                        "fps": camera.fps,
                        "depth_range": camera.depth_range,
                        "calibration_file": camera.calibration_file,
                        "position": camera.position,
                        "orientation": camera.orientation
                    } for camera in self.head.cameras
                ],
                "base_position": self.head.base_position
            } if self.head else None,
            "base_sensors": [
                {
                    "name": sensor.name,
                    "sensor_id": sensor.sensor_id,
                    "sensor_type": sensor.sensor_type.value,
                    "device_path": sensor.device_path,
                    "update_rate": sensor.update_rate,
                    "calibration_file": sensor.calibration_file,
                    "position": sensor.position,
                    "orientation": sensor.orientation
                } for sensor in self.base_sensors
            ],
            "cloud_enabled": self.cloud_enabled,
            "local_processing": self.local_processing,
            "teleop_enabled": self.teleop_enabled,
            "emergency_stop_enabled": self.emergency_stop_enabled,
            "collision_detection": self.collision_detection,
            "workspace_limits": self.workspace_limits,
            "control_frequency": self.control_frequency,
            "perception_frequency": self.perception_frequency,
            "safety_frequency": self.safety_frequency,
            "local_ip": self.local_ip,
            "cloud_endpoint": self.cloud_endpoint
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RobotConfig':
        """Create from dictionary."""
        # Parse arms
        arms = []
        for arm_data in data.get("arms", []):
            joints = []
            for joint_data in arm_data.get("joints", []):
                joint = JointConfig(
                    name=joint_data["name"],
                    joint_id=joint_data["joint_id"],
                    motor_type=MotorType(joint_data["motor_type"]),
                    min_position=joint_data.get("min_position", -3.14),
                    max_position=joint_data.get("max_position", 3.14),
                    max_velocity=joint_data.get("max_velocity", 1.0),
                    max_torque=joint_data.get("max_torque", 10.0),
                    gear_ratio=joint_data.get("gear_ratio", 1.0),
                    encoder_resolution=joint_data.get("encoder_resolution", 4096),
                    home_position=joint_data.get("home_position", 0.0),
                    safety_limits=joint_data.get("safety_limits", {})
                )
                joints.append(joint)
            
            arm = ArmConfig(
                name=arm_data["name"],
                arm_id=arm_data["arm_id"],
                joints=joints,
                end_effector=arm_data.get("end_effector"),
                base_position=arm_data.get("base_position", {"x": 0, "y": 0, "z": 0}),
                workspace_limits=arm_data.get("workspace_limits", {})
            )
            arms.append(arm)
        
        # Parse head
        head = None
        if "head" in data and data["head"]:
            head_data = data["head"]
            head_joints = []
            for joint_data in head_data.get("joints", []):
                joint = JointConfig(
                    name=joint_data["name"],
                    joint_id=joint_data["joint_id"],
                    motor_type=MotorType(joint_data["motor_type"]),
                    min_position=joint_data.get("min_position", -3.14),
                    max_position=joint_data.get("max_position", 3.14),
                    max_velocity=joint_data.get("max_velocity", 1.0),
                    max_torque=joint_data.get("max_torque", 10.0),
                    gear_ratio=joint_data.get("gear_ratio", 1.0),
                    encoder_resolution=joint_data.get("encoder_resolution", 4096),
                    home_position=joint_data.get("home_position", 0.0),
                    safety_limits=joint_data.get("safety_limits", {})
                )
                head_joints.append(joint)
            
            head_cameras = []
            for camera_data in head_data.get("cameras", []):
                camera = CameraConfig(
                    name=camera_data["name"],
                    camera_id=camera_data["camera_id"],
                    camera_type=CameraType(camera_data["camera_type"]),
                    device_path=camera_data["device_path"],
                    width=camera_data.get("width", 640),
                    height=camera_data.get("height", 480),
                    fps=camera_data.get("fps", 30),
                    depth_range=tuple(camera_data.get("depth_range", [0.1, 10.0])),
                    calibration_file=camera_data.get("calibration_file"),
                    position=camera_data.get("position", {"x": 0, "y": 0, "z": 0}),
                    orientation=camera_data.get("orientation", {"rx": 0, "ry": 0, "rz": 0})
                )
                head_cameras.append(camera)
            
            head = HeadConfig(
                name=head_data["name"],
                head_id=head_data["head_id"],
                joints=head_joints,
                cameras=head_cameras,
                base_position=head_data.get("base_position", {"x": 0, "y": 0, "z": 0})
            )
        
        # Parse base sensors
        base_sensors = []
        for sensor_data in data.get("base_sensors", []):
            sensor = SensorConfig(
                name=sensor_data["name"],
                sensor_id=sensor_data["sensor_id"],
                sensor_type=SensorType(sensor_data["sensor_type"]),
                device_path=sensor_data["device_path"],
                update_rate=sensor_data.get("update_rate", 100.0),
                calibration_file=sensor_data.get("calibration_file"),
                position=sensor_data.get("position", {"x": 0, "y": 0, "z": 0}),
                orientation=sensor_data.get("orientation", {"rx": 0, "ry": 0, "rz": 0})
            )
            base_sensors.append(sensor)
        
        return cls(
            robot_id=data["robot_id"],
            robot_name=data["robot_name"],
            robot_type=data.get("robot_type", "humanoid"),
            manufacturer=data.get("manufacturer", "custom"),
            model=data.get("model", "v1.0"),
            arms=arms,
            head=head,
            base_sensors=base_sensors,
            cloud_enabled=data.get("cloud_enabled", True),
            local_processing=data.get("local_processing", True),
            teleop_enabled=data.get("teleop_enabled", True),
            emergency_stop_enabled=data.get("emergency_stop_enabled", True),
            collision_detection=data.get("collision_detection", True),
            workspace_limits=data.get("workspace_limits", {}),
            control_frequency=data.get("control_frequency", 100.0),
            perception_frequency=data.get("perception_frequency", 30.0),
            safety_frequency=data.get("safety_frequency", 1000.0),
            local_ip=data.get("local_ip", "192.168.1.100"),
            cloud_endpoint=data.get("cloud_endpoint", "controller.eidolon.cloud:443")
        )


class RobotConfigManager:
    """Robot configuration manager."""
    
    def __init__(self, config_dir: str = "config/robots"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.available_configs: Dict[str, RobotConfig] = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load all available robot configurations."""
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                config = self.load_config(config_file.stem)
                self.available_configs[config_file.stem] = config
                logger.info("Loaded robot config", name=config_file.stem)
            except Exception as e:
                logger.error("Failed to load config", file=config_file, error=str(e))
    
    def load_config(self, config_name: str) -> RobotConfig:
        """Load robot configuration by name."""
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Check if this is a new format YAML config
        if 'robot' in data:
            return self._load_yaml_config(data)
        else:
            return RobotConfig.from_dict(data)
    
    def _load_yaml_config(self, data: Dict[str, Any]) -> RobotConfig:
        """Load new YAML format configuration."""
        robot_data = data['robot']
        
        # Extract robot info
        robot_id = robot_data.get('name', 'unknown').lower().replace(' ', '_')
        robot_name = robot_data.get('name', 'Unknown Robot')
        robot_type = robot_data.get('type', 'humanoid')
        version = robot_data.get('version', '1.0.0')
        description = robot_data.get('description', '')
        
        # Extract servo bus configuration
        servo_bus_config = robot_data.get('hardware', {}).get('servo_bus', {})
        
        # Extract motors configuration
        motors_config = robot_data.get('hardware', {}).get('motors', {})
        
        # Create joint configurations
        joints = []
        for joint_name, joint_data in motors_config.items():
            joint = JointConfig(
                name=joint_name,
                joint_id=joint_data.get('id', 0),
                motor_type=MotorType.POSITION,  # Default to position control
                min_position=joint_data.get('position_limits', [-180.0, 180.0])[0],
                max_position=joint_data.get('position_limits', [-180.0, 180.0])[1],
                max_velocity=joint_data.get('velocity_limit', 100.0),
                max_torque=joint_data.get('torque_limit', 100.0),
                gear_ratio=joint_data.get('gear_ratio', 1.0),
                home_position=0.0,
                servo_id=joint_data.get('id'),
                joint_type=joint_data.get('joint_type', 'revolute'),
                model=joint_data.get('model', 'STS3215'),
                offset=joint_data.get('offset', 0.0),
                direction=joint_data.get('direction', 1)
            )
            joints.append(joint)
        
        # Group joints into arms
        left_arm_joints = [j for j in joints if j.name.startswith('left_')]
        right_arm_joints = [j for j in joints if j.name.startswith('right_')]
        head_joints = [j for j in joints if j.name.startswith('head_')]
        
        # Create arm configurations
        arms = []
        if left_arm_joints:
            arms.append(ArmConfig("left_arm", "left", left_arm_joints, "left_hand"))
        if right_arm_joints:
            arms.append(ArmConfig("right_arm", "right", right_arm_joints, "right_hand"))
        
        # Create head configuration
        head = None
        if head_joints:
            # Extract cameras
            cameras_config = robot_data.get('cameras', {})
            cameras = []
            for camera_name, camera_data in cameras_config.items():
                if camera_data.get('enabled', True):  # Skip disabled cameras
                    camera = CameraConfig(
                        name=camera_name,
                        camera_id=camera_data.get('name', camera_name),
                        camera_type=CameraType(camera_data.get('type', 'rgb')),
                        device_path=f"/dev/video{camera_data.get('device_id', 0)}",
                        width=camera_data.get('resolution', [640, 480])[0],
                        height=camera_data.get('resolution', [640, 480])[1],
                        fps=camera_data.get('fps', 30),
                        position={
                            "x": camera_data.get('mount_offset', [0, 0, 0])[0],
                            "y": camera_data.get('mount_offset', [0, 0, 0])[1],
                            "z": camera_data.get('mount_offset', [0, 0, 0])[2]
                        }
                    )
                    cameras.append(camera)
            
            head = HeadConfig("head", "head", head_joints, cameras)
        
        # Extract base sensors
        sensors_config = robot_data.get('sensors', {})
        base_sensors = []
        for sensor_name, sensor_data in sensors_config.items():
            if sensor_data.get('enabled', True):  # Skip disabled sensors
                sensor = SensorConfig(
                    name=sensor_name,
                    sensor_id=sensor_data.get('name', sensor_name),
                    sensor_type=SensorType(sensor_data.get('type', 'imu')),
                    device_path=sensor_data.get('device_id', '/dev/ttyUSB0'),
                    update_rate=sensor_data.get('update_rate', 100.0),
                    position={
                        "x": sensor_data.get('mount_offset', [0, 0, 0])[0],
                        "y": sensor_data.get('mount_offset', [0, 0, 0])[1],
                        "z": sensor_data.get('mount_offset', [0, 0, 0])[2]
                    }
                )
                base_sensors.append(sensor)
        
        # Extract capabilities
        capabilities = robot_data.get('capabilities', {})
        
        # Extract safety configuration
        safety_config = robot_data.get('safety', {})
        
        # Extract control configuration
        control_config = robot_data.get('control', {})
        
        # Extract kinematics
        kinematics = robot_data.get('kinematics', {})
        
        # Create RobotConfig
        config = RobotConfig(
            robot_id=robot_id,
            robot_name=robot_name,
            robot_type=robot_type,
            manufacturer="LeRobot",
            model="SO100",
            arms=arms,
            head=head,
            base_sensors=base_sensors,
            cloud_enabled=True,
            local_processing=True,
            teleop_enabled=True,
            emergency_stop_enabled=safety_config.get('emergency_stop', {}).get('hardware_estop', True),
            collision_detection=safety_config.get('collision_detection', {}).get('enabled', True),
            workspace_limits=safety_config.get('workspace_limits', {}),
            control_frequency=robot_data.get('software', {}).get('control_loop', {}).get('frequency', 100.0),
            perception_frequency=30.0,
            safety_frequency=1000.0,
            local_ip=robot_data.get('communication', {}).get('local_network', {}).get('multicast_address', '192.168.1.100'),
            cloud_endpoint=robot_data.get('communication', {}).get('cloud_server', {}).get('host', 'controller.eidolon.cloud:443')
        )
        
        # Add additional attributes for the new format
        config.name = robot_name
        config.version = version
        config.description = description
        config.servo_bus = servo_bus_config
        config.capabilities = capabilities
        config.safety = safety_config
        config.control = control_config
        config.kinematics = kinematics
        
        return config
    
    def save_config(self, config: RobotConfig, config_name: str):
        """Save robot configuration."""
        config_file = self.config_dir / f"{config_name}.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
        
        self.available_configs[config_name] = config
        logger.info("Saved robot config", name=config_name)
    
    def list_configs(self) -> List[str]:
        """List available configurations."""
        return list(self.available_configs.keys())
    
    def get_config(self, config_name: str) -> Optional[RobotConfig]:
        """Get configuration by name."""
        return self.available_configs.get(config_name)
    
    def create_default_humanoid_config(self, robot_id: str, robot_name: str) -> RobotConfig:
        """Create default humanoid robot configuration."""
        # Left arm joints
        left_arm_joints = [
            JointConfig("left_shoulder_pitch", 0, MotorType.POSITION, -1.57, 1.57, 2.0, 20.0),
            JointConfig("left_shoulder_roll", 1, MotorType.POSITION, -1.57, 1.57, 2.0, 20.0),
            JointConfig("left_shoulder_yaw", 2, MotorType.POSITION, -1.57, 1.57, 2.0, 20.0),
            JointConfig("left_elbow", 3, MotorType.POSITION, -2.0, 2.0, 3.0, 15.0),
            JointConfig("left_wrist_pitch", 4, MotorType.POSITION, -1.57, 1.57, 3.0, 10.0),
            JointConfig("left_wrist_roll", 5, MotorType.POSITION, -3.14, 3.14, 3.0, 10.0),
        ]
        
        # Right arm joints
        right_arm_joints = [
            JointConfig("right_shoulder_pitch", 6, MotorType.POSITION, -1.57, 1.57, 2.0, 20.0),
            JointConfig("right_shoulder_roll", 7, MotorType.POSITION, -1.57, 1.57, 2.0, 20.0),
            JointConfig("right_shoulder_yaw", 8, MotorType.POSITION, -1.57, 1.57, 2.0, 20.0),
            JointConfig("right_elbow", 9, MotorType.POSITION, -2.0, 2.0, 3.0, 15.0),
            JointConfig("right_wrist_pitch", 10, MotorType.POSITION, -1.57, 1.57, 3.0, 10.0),
            JointConfig("right_wrist_roll", 11, MotorType.POSITION, -3.14, 3.14, 3.0, 10.0),
        ]
        
        # Head joints
        head_joints = [
            JointConfig("head_pan", 12, MotorType.POSITION, -1.57, 1.57, 2.0, 5.0),
            JointConfig("head_tilt", 13, MotorType.POSITION, -0.5, 0.5, 2.0, 5.0),
        ]
        
        # Cameras
        cameras = [
            CameraConfig("main_camera", "cam_0", CameraType.RGB, "/dev/video0", 640, 480, 30),
            CameraConfig("depth_camera", "cam_1", CameraType.RGBD, "/dev/video1", 640, 480, 30),
        ]
        
        # Base sensors
        base_sensors = [
            SensorConfig("imu", "imu_0", SensorType.IMU, "/dev/ttyUSB0", 100.0),
            SensorConfig("force_torque_left", "ft_left", SensorType.FORCE_TORQUE, "/dev/ttyUSB1", 100.0),
            SensorConfig("force_torque_right", "ft_right", SensorType.FORCE_TORQUE, "/dev/ttyUSB2", 100.0),
        ]
        
        # Create arms
        left_arm = ArmConfig("left_arm", "left", left_arm_joints, "left_hand")
        right_arm = ArmConfig("right_arm", "right", right_arm_joints, "right_hand")
        
        # Create head
        head = HeadConfig("head", "head", head_joints, cameras)
        
        return RobotConfig(
            robot_id=robot_id,
            robot_name=robot_name,
            robot_type="humanoid",
            arms=[left_arm, right_arm],
            head=head,
            base_sensors=base_sensors,
            workspace_limits={
                "x_min": -1.0, "x_max": 1.0,
                "y_min": -1.0, "y_max": 1.0,
                "z_min": 0.0, "z_max": 2.0
            }
        )
