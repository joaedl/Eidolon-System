#!/usr/bin/env python3
"""
Test script for LeRobot SO100 and Eidolon001 robot configurations and hardware integration.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eidolon.robot.config import RobotConfigManager
from eidolon.robot.hardware import HardwareManager
from eidolon.robot.feetech import FeetechServoBus, FeetechConfig
import structlog

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

logger = structlog.get_logger(__name__)


async def test_config_loading():
    """Test loading robot configurations."""
    print("🔧 Testing robot configuration loading...")
    
    try:
        config_manager = RobotConfigManager("config/robots")
        
        # Test all LeRobot and Eidolon configurations
        configs_to_test = ["lerobot_so100", "lerobot_so100_dual", "eidolon001"]
        
        configs = {}
        for config_name in configs_to_test:
            print(f"\n📋 Testing {config_name} configuration...")
            config = configs[config_name] = config_manager.load_config(config_name)
            
            print(f"✅ Configuration loaded successfully!")
            print(f"   Robot: {config.robot_name} ({config.robot_type})")
            print(f"   Version: {getattr(config, 'version', 'Unknown')}")
            print(f"   Description: {getattr(config, 'description', 'No description')}")
            
            # Print hardware information
            print(f"\n🤖 Hardware Configuration:")
            print(f"   Arms: {len(config.arms)}")
            for arm in config.arms:
                print(f"     - {arm.name}: {len(arm.joints)} joints")
                for joint in arm.joints:
                    servo_id = getattr(joint, 'servo_id', 'N/A')
                    print(f"       * {joint.name} (ID: {servo_id}, Model: {getattr(joint, 'model', 'Unknown')})")
            
            if config.head:
                print(f"   Head: {len(config.head.joints)} joints, {len(config.head.cameras)} cameras")
                for joint in config.head.joints:
                    servo_id = getattr(joint, 'servo_id', 'N/A')
                    print(f"     - {joint.name} (ID: {servo_id})")
                for camera in config.head.cameras:
                    print(f"     - {camera.name} ({camera.camera_type.value})")
            
            print(f"   Base Sensors: {len(config.base_sensors)}")
            for sensor in config.base_sensors:
                print(f"     - {sensor.name} ({sensor.sensor_type.value})")
            
            # Print servo bus configuration
            if hasattr(config, 'servo_bus'):
                servo_bus = config.servo_bus
                print(f"\n🔌 Servo Bus Configuration:")
                print(f"   Port: {servo_bus.get('port', 'N/A')}")
                print(f"   Baudrate: {servo_bus.get('baudrate', 'N/A')}")
                print(f"   Mock Mode: {servo_bus.get('mock', False)}")
            
            # Print capabilities
            if hasattr(config, 'capabilities'):
                capabilities = config.capabilities
                print(f"\n🎯 Robot Capabilities:")
                print(f"   Low-level: {len(capabilities.get('low_level', []))} functions")
                print(f"   High-level: {len(capabilities.get('high_level', []))} functions")
                print(f"   Perception: {len(capabilities.get('perception', []))} capabilities")
                print(f"   Safety: {len(capabilities.get('safety', []))} features")
        
        # Return the single arm config for further testing
        return configs.get("lerobot_so100")
        
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return None


async def test_hardware_manager(config):
    """Test hardware manager with the configuration."""
    print("\n🔧 Testing Hardware Manager...")
    
    try:
        hardware_manager = HardwareManager(config)
        await hardware_manager.initialize()
        
        print("✅ Hardware manager initialized successfully!")
        
        # Test servo bus connection
        if hardware_manager.servo_bus:
            print(f"🔌 Servo bus connected: {hardware_manager.servo_bus.is_connected}")
            if hardware_manager.servo_bus.is_connected:
                print("   Testing servo communication...")
                # Test reading from first servo
                test_id = 1
                position = hardware_manager.servo_bus.read_position(test_id)
                if position is not None:
                    print(f"   ✅ Successfully read position from servo {test_id}: {position:.2f}°")
                else:
                    print(f"   ⚠️  Could not read position from servo {test_id}")
        
        # Test motor controllers
        print(f"\n🎮 Motor Controllers:")
        print(f"   Total controllers: {len(hardware_manager.motor_controllers)}")
        
        for name, controller in hardware_manager.motor_controllers.items():
            servo_id = getattr(controller.joint_config, 'servo_id', 'N/A')
            controller_type = type(controller).__name__
            print(f"   - {name}: {controller_type} (Servo ID: {servo_id})")
        
        # Test camera interfaces
        print(f"\n📷 Camera Interfaces:")
        print(f"   Total cameras: {len(hardware_manager.camera_interfaces)}")
        
        for camera_id, camera in hardware_manager.camera_interfaces.items():
            print(f"   - {camera_id}: {camera.camera_config.camera_type.value}")
        
        # Test sensor interfaces
        print(f"\n📊 Sensor Interfaces:")
        print(f"   Total sensors: {len(hardware_manager.sensor_interfaces)}")
        
        for sensor_id, sensor in hardware_manager.sensor_interfaces.items():
            print(f"   - {sensor_id}: {sensor.sensor_config.sensor_type.value}")
        
        # Test robot capabilities presentation
        print(f"\n📋 Robot Capabilities:")
        capabilities = hardware_manager.get_robot_capabilities()
        
        robot_info = capabilities.get("robot_info", {})
        print(f"   Name: {robot_info.get('name', 'Unknown')}")
        print(f"   Type: {robot_info.get('type', 'Unknown')}")
        print(f"   Hardware Ready: {robot_info.get('hardware_ready', False)}")
        
        joints = capabilities.get("joints", {})
        print(f"   Joints: {joints.get('count', 0)}")
        
        cameras = capabilities.get("cameras", {})
        print(f"   Cameras: {cameras.get('count', 0)}")
        
        sensors = capabilities.get("sensors", {})
        print(f"   Sensors: {sensors.get('count', 0)}")
        
        # Test joint control
        print(f"\n🎯 Testing Joint Control:")
        if hardware_manager.motor_controllers:
            first_joint = list(hardware_manager.motor_controllers.keys())[0]
            controller = hardware_manager.motor_controllers[first_joint]
            
            print(f"   Testing joint: {first_joint}")
            
            # Enable the joint
            await controller.enable()
            print(f"   ✅ Joint enabled")
            
            # Set position
            await controller.set_position(0.0)
            print(f"   ✅ Position set to 0.0°")
            
            # Read position
            state = controller.get_state()
            print(f"   ✅ Current position: {state.position:.2f}°")
            
            # Disable the joint
            await controller.disable()
            print(f"   ✅ Joint disabled")
        
        await hardware_manager.shutdown()
        print("✅ Hardware manager test completed successfully!")
        
    except Exception as e:
        print(f"❌ Hardware manager test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_feetech_connection():
    """Test Feetech servo bus connection."""
    print("\n🔌 Testing Feetech Servo Bus Connection...")
    
    try:
        # Test with mock mode first
        config = FeetechConfig(port="/dev/ttyUSB0", mock=True)
        bus = FeetechServoBus(config)
        
        if bus.connect():
            print("✅ Mock Feetech bus connected successfully!")
            
            # Test reading positions
            print("   Testing position reading...")
            for servo_id in range(1, 7):  # Test first 6 servos
                position = bus.read_position(servo_id)
                if position is not None:
                    print(f"   ✅ Servo {servo_id}: {position:.2f}°")
                else:
                    print(f"   ⚠️  Servo {servo_id}: No response")
            
            # Test setting positions
            print("   Testing position setting...")
            for servo_id in range(1, 4):  # Test first 3 servos
                success = bus.set_position(servo_id, 45.0)
                if success:
                    print(f"   ✅ Servo {servo_id}: Position set to 45°")
                else:
                    print(f"   ⚠️  Servo {servo_id}: Failed to set position")
            
            bus.disconnect()
            print("✅ Mock Feetech bus test completed!")
        else:
            print("❌ Failed to connect to mock Feetech bus")
        
        # Test real connection (if available)
        print("\n   Testing real hardware connection...")
        real_config = FeetechConfig(port="/dev/ttyUSB0", mock=False)
        real_bus = FeetechServoBus(real_config)
        
        if real_bus.connect():
            print("✅ Real Feetech bus connected!")
            real_bus.disconnect()
        else:
            print("⚠️  Real hardware not available (this is normal if no servos connected)")
        
    except Exception as e:
        print(f"❌ Feetech connection test failed: {e}")


async def main():
    """Main test function."""
    print("🚀 Robot Configuration Test")
    print("=" * 50)
    
    # Test configuration loading
    config = await test_config_loading()
    if not config:
        print("❌ Configuration test failed, aborting")
        return
    
    # Test hardware manager
    await test_hardware_manager(config)
    
    # Test Feetech connection
    await test_feetech_connection()
    
    print("\n🎉 All tests completed!")
    print("\nTo run the robot system with different configurations:")
    print("   # Single arm (6 servos, 1 camera):")
    print("   python scripts/run_robot_enhanced.py --config lerobot_so100")
    print("   # Dual arm (12 servos, 2 cameras):")
    print("   python scripts/run_robot_enhanced.py --config lerobot_so100_dual")
    print("   # Eidolon001 (16 servos, 1 camera, 2 grippers):")
    print("   python scripts/run_robot_enhanced.py --config eidolon001")


if __name__ == "__main__":
    asyncio.run(main())
