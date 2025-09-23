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
