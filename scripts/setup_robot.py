#!/usr/bin/env python3
"""Setup script for robot configuration."""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eidolon.robot.config import RobotConfigManager, RobotConfig

def create_robot_config():
    """Interactive robot configuration creation."""
    print("🤖 Eidolon Robot Configuration Setup")
    print("=" * 50)
    
    # Get basic robot information
    robot_id = input("Enter robot ID (e.g., robot-001): ").strip()
    if not robot_id:
        robot_id = "robot-001"
    
    robot_name = input("Enter robot name (e.g., 'Warehouse Robot 1'): ").strip()
    if not robot_name:
        robot_name = f"Robot {robot_id}"
    
    robot_type = input("Enter robot type (humanoid/manipulator/mobile) [humanoid]: ").strip()
    if not robot_type:
        robot_type = "humanoid"
    
    # Create config manager
    config_manager = RobotConfigManager()
    
    # Create default configuration
    if robot_type == "humanoid":
        config = config_manager.create_default_humanoid_config(robot_id, robot_name)
    else:
        print(f"Creating basic {robot_type} configuration...")
        # For now, use humanoid as template
        config = config_manager.create_default_humanoid_config(robot_id, robot_name)
        config.robot_type = robot_type
    
    # Configure arms
    print("\n📋 Configuring Arms")
    print("-" * 20)
    
    num_arms = int(input("Number of arms [2]: ") or "2")
    config.arms = []
    
    for i in range(num_arms):
        arm_name = f"arm_{i+1}" if num_arms > 1 else "arm"
        arm_id = f"arm_{i+1}" if num_arms > 1 else "arm"
        
        print(f"\nConfiguring {arm_name}:")
        num_joints = int(input(f"Number of joints for {arm_name} [6]: ") or "6")
        
        joints = []
        for j in range(num_joints):
            joint_name = input(f"Joint {j+1} name (e.g., shoulder_pitch): ").strip()
            if not joint_name:
                joint_name = f"joint_{j+1}"
            
            motor_type = input(f"Motor type for {joint_name} (position/velocity/torque) [position]: ").strip()
            if not motor_type:
                motor_type = "position"
            
            from eidolon.robot.config import JointConfig, MotorType
            joint = JointConfig(
                name=joint_name,
                joint_id=j,
                motor_type=MotorType(motor_type),
                min_position=float(input(f"Min position for {joint_name} [-3.14]: ") or "-3.14"),
                max_position=float(input(f"Max position for {joint_name} [3.14]: ") or "3.14"),
                max_velocity=float(input(f"Max velocity for {joint_name} [2.0]: ") or "2.0"),
                max_torque=float(input(f"Max torque for {joint_name} [10.0]: ") or "10.0"),
                home_position=float(input(f"Home position for {joint_name} [0.0]: ") or "0.0")
            )
            joints.append(joint)
        
        from eidolon.robot.config import ArmConfig
        arm = ArmConfig(
            name=arm_name,
            arm_id=arm_id,
            joints=joints
        )
        config.arms.append(arm)
    
    # Configure head
    print("\n📋 Configuring Head")
    print("-" * 20)
    
    has_head = input("Does the robot have a head? (y/n) [y]: ").strip().lower()
    if has_head in ['y', 'yes', '']:
        num_head_joints = int(input("Number of head joints [2]: ") or "2")
        
        head_joints = []
        for j in range(num_head_joints):
            joint_name = input(f"Head joint {j+1} name (e.g., pan): ").strip()
            if not joint_name:
                joint_name = f"head_joint_{j+1}"
            
            from eidolon.robot.config import JointConfig, MotorType
            joint = JointConfig(
                name=joint_name,
                joint_id=len(config.arms) * 6 + j,  # Offset joint IDs
                motor_type=MotorType.POSITION,
                min_position=float(input(f"Min position for {joint_name} [-1.57]: ") or "-1.57"),
                max_position=float(input(f"Max position for {joint_name} [1.57]: ") or "1.57"),
                max_velocity=float(input(f"Max velocity for {joint_name} [2.0]: ") or "2.0"),
                max_torque=float(input(f"Max torque for {joint_name} [5.0]: ") or "5.0"),
                home_position=float(input(f"Home position for {joint_name} [0.0]: ") or "0.0")
            )
            head_joints.append(joint)
        
        # Configure head cameras
        num_cameras = int(input("Number of head cameras [1]: ") or "1")
        cameras = []
        
        for c in range(num_cameras):
            camera_name = input(f"Camera {c+1} name: ").strip()
            if not camera_name:
                camera_name = f"camera_{c+1}"
            
            camera_type = input(f"Camera type (rgb/rgbd) [rgb]: ").strip()
            if not camera_type:
                camera_type = "rgb"
            
            device_path = input(f"Device path (e.g., /dev/video0): ").strip()
            if not device_path:
                device_path = f"/dev/video{c}"
            
            from eidolon.robot.config import CameraConfig, CameraType
            camera = CameraConfig(
                name=camera_name,
                camera_id=f"cam_{c}",
                camera_type=CameraType(camera_type),
                device_path=device_path,
                width=int(input(f"Camera width [640]: ") or "640"),
                height=int(input(f"Camera height [480]: ") or "480"),
                fps=int(input(f"Camera FPS [30]: ") or "30")
            )
            cameras.append(camera)
        
        from eidolon.robot.config import HeadConfig
        config.head = HeadConfig(
            name="head",
            head_id="head",
            joints=head_joints,
            cameras=cameras
        )
    
    # Configure base sensors
    print("\n📋 Configuring Base Sensors")
    print("-" * 20)
    
    num_sensors = int(input("Number of base sensors [0]: ") or "0")
    base_sensors = []
    
    for s in range(num_sensors):
        sensor_name = input(f"Sensor {s+1} name: ").strip()
        if not sensor_name:
            sensor_name = f"sensor_{s+1}"
        
        sensor_type = input(f"Sensor type (imu/force_torque/touch/proximity) [imu]: ").strip()
        if not sensor_type:
            sensor_type = "imu"
        
        device_path = input(f"Device path (e.g., /dev/ttyUSB0): ").strip()
        if not device_path:
            device_path = f"/dev/ttyUSB{s}"
        
        from eidolon.robot.config import SensorConfig, SensorType
        sensor = SensorConfig(
            name=sensor_name,
            sensor_id=f"sensor_{s}",
            sensor_type=SensorType(sensor_type),
            device_path=device_path,
            update_rate=float(input(f"Update rate (Hz) [100]: ") or "100")
        )
        base_sensors.append(sensor)
    
    config.base_sensors = base_sensors
    
    # System configuration
    print("\n📋 System Configuration")
    print("-" * 20)
    
    config.cloud_enabled = input("Enable cloud connection? (y/n) [y]: ").strip().lower() in ['y', 'yes', '']
    config.local_processing = input("Enable local processing? (y/n) [y]: ").strip().lower() in ['y', 'yes', '']
    config.teleop_enabled = input("Enable teleoperation? (y/n) [y]: ").strip().lower() in ['y', 'yes', '']
    
    # Save configuration
    config_name = input(f"\nSave configuration as [default]: ").strip()
    if not config_name:
        config_name = "default"
    
    config_manager.save_config(config, config_name)
    print(f"\n✅ Robot configuration saved as '{config_name}'")
    print(f"📁 Configuration file: config/robots/{config_name}.yaml")
    
    return config_name

def list_configs():
    """List available robot configurations."""
    config_manager = RobotConfigManager()
    configs = config_manager.list_configs()
    
    print("🤖 Available Robot Configurations")
    print("=" * 40)
    
    if not configs:
        print("No configurations found.")
        return
    
    for config_name in configs:
        try:
            config = config_manager.get_config(config_name)
            print(f"📋 {config_name}")
            print(f"   Robot ID: {config.robot_id}")
            print(f"   Name: {config.robot_name}")
            print(f"   Type: {config.robot_type}")
            print(f"   Arms: {len(config.arms)}")
            print(f"   Head: {'Yes' if config.head else 'No'}")
            print(f"   Cameras: {len(config.head.cameras) if config.head else 0}")
            print(f"   Sensors: {len(config.base_sensors)}")
            print()
        except Exception as e:
            print(f"❌ Error loading {config_name}: {e}")

def main():
    """Main setup script."""
    parser = argparse.ArgumentParser(description="Robot configuration setup")
    parser.add_argument("--create", action="store_true", help="Create new robot configuration")
    parser.add_argument("--list", action="store_true", help="List available configurations")
    parser.add_argument("--config", help="Configuration name to use")
    
    args = parser.parse_args()
    
    if args.list:
        list_configs()
    elif args.create:
        create_robot_config()
    else:
        print("🤖 Eidolon Robot Setup")
        print("=" * 30)
        print("Use --create to create a new configuration")
        print("Use --list to see available configurations")
        print("Use --config <name> to use a specific configuration")

if __name__ == "__main__":
    main()
