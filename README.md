# Eidolon Robot Fleet Management System

A comprehensive robot fleet management system that enables teleoperation, autonomous operation, and cloud-based intelligence for large robot fleets.

## Architecture Overview

The system consists of several interconnected modules:

1. **Robot (Edge)** - Real-time control, safety, and local intelligence
2. **Cloud Server** - Multi-tenant orchestration and intelligence
3. **Remote Operator** - Human-in-the-loop teleoperation
4. **Customer Dashboard** - Fleet management web interface
5. **Teleop Gateway** - WebRTC signaling and NAT traversal
6. **Data Lake** - Storage, model registry, and MLOps
7. **Monitoring** - Security, audit, and observability

## 🤖 Enhanced Robot Module

The robot module has been significantly enhanced to support flexible hardware configurations for humanoid robots:

### **Key Features**
- **Flexible Configuration System**: YAML-based configurations for different robot setups
- **Hardware Abstraction Layer**: Support for various motor types, cameras, and sensors
- **Multiple Control Modes**: Position, velocity, torque, and impedance control
- **Comprehensive Safety**: Hardware safety chains, emergency stops, collision detection
- **Teleoperation Support**: Real-time control with WebRTC and P2P communication
- **Local & Cloud Processing**: Flexible deployment options

### **Hardware Support**
- **Dual-arm humanoid robots** with 6-DOF arms
- **Head with pan/tilt** and multiple cameras (RGB, RGB-D, stereo)
- **Various motor types**: Position, velocity, torque control
- **Comprehensive sensors**: IMU, force-torque, touch, proximity, LIDAR
- **Flexible deployment**: Raspberry Pi, industrial computers, edge devices

### **Quick Robot Setup**
```bash
# Create robot configuration
python scripts/setup_robot.py --create

# Run robot system
python scripts/run_robot_enhanced.py --config my_robot

# Run in local-only mode
python scripts/run_robot_enhanced.py --local-only
```

### **Configuration Examples**
- **Humanoid Robot**: Dual arms, head with cameras, IMU sensors
- **Dual-Arm Manipulator**: Two 6-DOF arms for assembly tasks
- **Mobile Robot**: Base sensors, navigation cameras
- **Custom Configurations**: Flexible setup for any robot type

For detailed robot module documentation, see [ROBOT_MODULE.md](ROBOT_MODULE.md).

## Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- ROS2 (for robot edge components)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd eidolon-system

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Start infrastructure services
docker-compose up -d

# Run the system
python -m eidolon.robot.main  # Robot edge
python -m eidolon.cloud.main  # Cloud server
python -m eidolon.operator.main  # Operator console
```

## Security Features

- mTLS authentication for all services
- Device attestation with TPM support
- End-to-end encryption for teleoperation
- Role-based access control (RBAC)
- Immutable audit logging
- Hardware safety chains independent of network

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy .
```

## License

MIT License - see LICENSE file for details.
