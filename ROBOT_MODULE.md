# 🤖 Eidolon Robot Module

The enhanced robot module provides a flexible, configurable system for controlling humanoid robots with various hardware configurations. It supports both local processing and cloud connectivity, with comprehensive teleoperation capabilities.

## 🏗️ Architecture

### Core Components

1. **Configuration System** (`config.py`)
   - Flexible robot configuration management
   - Support for different hardware setups
   - YAML-based configuration files

2. **Hardware Abstraction** (`hardware.py`)
   - Motor controllers (position, velocity, torque)
   - Camera interfaces (RGB, RGB-D, stereo)
   - Sensor interfaces (IMU, force-torque, touch)
   - Hardware-agnostic abstraction layer

3. **Robot System** (`robot_system.py`)
   - Main system coordinator
   - Hardware initialization and management
   - Safety and control integration

4. **Teleoperation Interface** (`teleop_interface.py`)
   - Real-time teleoperation control
   - Multiple control modes
   - Safety validation

5. **Safety System** (`safety.py`)
   - Hardware safety chains
   - Emergency stop handling
   - Collision detection

6. **Control System** (`control.py`)
   - Real-time motor control
   - Trajectory execution
   - Control mode management

7. **Perception Pipeline** (`perception.py`)
   - Computer vision processing
   - Object detection and tracking
   - Multi-camera support

8. **Brain Client** (`brain_client.py`)
   - Cloud connectivity
   - Telemetry streaming
   - Subgoal processing

## 🚀 Quick Start

### 1. Setup Robot Configuration

```bash
# Create a new robot configuration
python scripts/setup_robot.py --create

# List available configurations
python scripts/setup_robot.py --list
```

### 2. Run Robot System

```bash
# Run with default configuration
python scripts/run_robot_enhanced.py

# Run with specific configuration
python scripts/run_robot_enhanced.py --config my_robot

# Run in local-only mode (no cloud)
python scripts/run_robot_enhanced.py --local-only

# Run with debug logging
python scripts/run_robot_enhanced.py --debug
```

### 3. Configuration Examples

#### Basic Humanoid Robot
```yaml
robot_id: "robot-001"
robot_name: "Warehouse Robot 1"
robot_type: "humanoid"

arms:
  - name: "left_arm"
    arm_id: "left"
    joints:
      - name: "left_shoulder_pitch"
        joint_id: 0
        motor_type: "position"
        min_position: -1.57
        max_position: 1.57
        max_velocity: 2.0
        max_torque: 20.0
        home_position: 0.0

head:
  name: "head"
  head_id: "head"
  joints:
    - name: "head_pan"
      joint_id: 12
      motor_type: "position"
  cameras:
    - name: "main_camera"
      camera_id: "cam_0"
      camera_type: "rgb"
      device_path: "/dev/video0"
      width: 640
      height: 480
```

#### Dual-Arm Manipulator
```yaml
robot_id: "manipulator-001"
robot_name: "Dual-Arm Manipulator"
robot_type: "manipulator"

arms:
  - name: "left_arm"
    arm_id: "left"
    joints: [6 joints]
  - name: "right_arm"
    arm_id: "right"
    joints: [6 joints]

# No head, no cameras
head: null
```

## 🔧 Hardware Support

### Motor Controllers
- **Position Control**: Precise joint positioning
- **Velocity Control**: Smooth motion control
- **Torque Control**: Force-based control
- **Impedance Control**: Compliant control

### Camera Systems
- **RGB Cameras**: Standard color cameras
- **RGB-D Cameras**: Depth-enabled cameras
- **Stereo Cameras**: 3D vision systems
- **Multiple Cameras**: Head-mounted camera arrays

### Sensors
- **IMU**: Inertial measurement units
- **Force-Torque**: End-effector force sensing
- **Touch**: Tactile sensors
- **Proximity**: Distance sensors
- **LIDAR**: 3D scanning

## 🎮 Teleoperation Modes

### Control Modes
1. **Position Control**: Direct joint positioning
2. **Velocity Control**: Joint velocity commands
3. **Torque Control**: Force-based control
4. **Pose Control**: End-effector positioning

### Safety Features
- **Velocity Limits**: Maximum joint velocities
- **Workspace Limits**: Safe operating zones
- **Collision Detection**: Real-time collision avoidance
- **Emergency Stop**: Immediate safety shutdown

## 📡 Communication

### Local Processing
- **Real-time Control**: 100Hz control loops
- **Perception**: 30Hz vision processing
- **Safety**: 1000Hz safety monitoring

### Cloud Connectivity
- **Telemetry Streaming**: Real-time data upload
- **Subgoal Processing**: Cloud-based planning
- **Remote Monitoring**: Status and diagnostics

### Teleoperation
- **WebRTC**: Low-latency video/control
- **P2P Communication**: Direct robot-operator connection
- **TURN Relay**: NAT traversal support

## 🛠️ Configuration Management

### Robot Configuration Files
Located in `config/robots/`, each robot has a YAML configuration file:

```yaml
# Basic robot information
robot_id: "robot-001"
robot_name: "My Robot"
robot_type: "humanoid"

# Hardware configuration
arms: [...]
head: {...}
base_sensors: [...]

# System settings
cloud_enabled: true
local_processing: true
teleop_enabled: true

# Performance tuning
control_frequency: 100.0
perception_frequency: 30.0
safety_frequency: 1000.0
```

### Environment Variables
```bash
# Robot identification
export ROBOT_ID="robot-001"
export ROBOT_TENANT_ID="tenant-001"

# System configuration
export CLOUD_ENABLED="true"
export TELEOP_ENABLED="true"
export SAFETY_ENABLED="true"

# Debug settings
export LOG_LEVEL="DEBUG"
```

## 🔒 Security Features

### Device Security
- **mTLS Authentication**: Mutual TLS for all communications
- **Device Certificates**: X.509 certificate-based identity
- **Secure Boot**: Verified firmware loading
- **Hardware Security**: TPM/secure element support

### Network Security
- **Encrypted Communications**: All data encrypted in transit
- **Access Control**: Role-based permissions
- **Audit Logging**: Complete action tracking
- **Intrusion Detection**: Real-time threat monitoring

## 📊 Monitoring and Diagnostics

### System Status
```python
# Get robot system status
status = robot_system.get_system_status()
print(f"Hardware ready: {status['hardware_ready']}")
print(f"Safety OK: {status['safety_ok']}")
print(f"Control active: {status['control_active']}")
print(f"Cloud connected: {status['cloud_connected']}")
```

### Joint States
```python
# Get all joint states
joint_states = robot_system.get_joint_states()
for joint in joint_states:
    print(f"{joint['name']}: {joint['position']:.3f} rad")
```

### Camera Feeds
```python
# Get camera frames
frames = robot_system.get_camera_frames()
for frame in frames:
    print(f"Camera {frame['camera_id']}: {frame['frame_id']}")
```

## 🧪 Testing and Development

### Simulation Mode
```bash
# Run in simulation mode (no hardware)
python scripts/run_robot_enhanced.py --config simulation
```

### Debug Mode
```bash
# Enable debug logging
python scripts/run_robot_enhanced.py --debug
```

### Hardware Testing
```python
# Test individual components
from eidolon.robot.hardware import HardwareManager
from eidolon.robot.config import RobotConfigManager

config_manager = RobotConfigManager()
config = config_manager.load_config("default")
hardware = HardwareManager(config)

await hardware.initialize()
joint_states = hardware.get_joint_states()
```

## 🔄 Deployment Options

### 1. Local Processing Only
- All processing on robot computer
- No cloud connectivity
- Offline operation

### 2. Hybrid Processing
- Local real-time control
- Cloud-based planning
- Teleoperation support

### 3. Cloud-First
- Minimal local processing
- Cloud-based intelligence
- Remote operation

## 📚 API Reference

### RobotSystem Class
```python
class RobotSystem:
    def __init__(self, config_name: str, config_dir: str)
    async def initialize(self)
    async def start(self)
    async def stop(self)
    def get_system_status(self) -> Dict[str, Any]
    def get_joint_states(self) -> List[Dict[str, Any]]
    async def move_joint(self, joint_name: str, position: float)
    async def move_arm(self, arm_id: str, positions: List[float])
    async def emergency_stop(self)
```

### HardwareManager Class
```python
class HardwareManager:
    def __init__(self, robot_config: RobotConfig)
    async def initialize(self)
    async def shutdown(self)
    def get_joint_states(self) -> List[JointState]
    async def set_joint_position(self, joint_name: str, position: float)
    async def get_camera_frame(self, camera_id: str) -> CameraFrame
    async def get_sensor_data(self, sensor_id: str) -> SensorData
```

### TeleopInterface Class
```python
class TeleopInterface:
    def __init__(self, robot_config, hardware_manager, control_manager)
    async def start(self)
    async def stop(self)
    def set_teleop_mode(self, mode: TeleopMode)
    def set_active_arm(self, arm_id: str)
    async def process_command(self, command: TeleopCommand)
    def get_teleop_status(self) -> Dict[str, Any]
```

## 🚨 Troubleshooting

### Common Issues

1. **Hardware Not Detected**
   - Check device paths in configuration
   - Verify USB/serial connections
   - Check permissions on device files

2. **Camera Issues**
   - Verify camera device paths
   - Check camera permissions
   - Test with `v4l2-ctl` for USB cameras

3. **Motor Control Problems**
   - Check motor connections
   - Verify joint limits
   - Test with low-level motor commands

4. **Network Connectivity**
   - Check firewall settings
   - Verify cloud endpoint accessibility
   - Test with `ping` and `telnet`

### Debug Commands
```bash
# Check hardware devices
ls -la /dev/video*
ls -la /dev/ttyUSB*

# Test camera
v4l2-ctl --list-devices
ffplay /dev/video0

# Check network connectivity
ping controller.eidolon.cloud
telnet controller.eidolon.cloud 443
```

## 📈 Performance Optimization

### Control Loop Tuning
- **Control Frequency**: Adjust based on hardware capabilities
- **Safety Frequency**: Higher for critical safety systems
- **Perception Frequency**: Balance between performance and CPU usage

### Network Optimization
- **Telemetry Batching**: Reduce network overhead
- **Video Compression**: Optimize bandwidth usage
- **Local Processing**: Minimize cloud dependencies

### Hardware Optimization
- **Motor Tuning**: Optimize PID parameters
- **Camera Settings**: Balance resolution and frame rate
- **Sensor Fusion**: Combine multiple sensor inputs

This enhanced robot module provides a comprehensive foundation for controlling humanoid robots with flexible hardware configurations, robust safety systems, and seamless cloud connectivity.
