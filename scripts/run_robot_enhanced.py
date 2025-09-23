#!/usr/bin/env python3
"""Enhanced robot runner with configuration support."""

import sys
import os
import argparse
import asyncio

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eidolon.robot.robot_system import RobotSystem

async def main():
    """Main entry point for enhanced robot system."""
    parser = argparse.ArgumentParser(description="Run Eidolon Robot System")
    parser.add_argument("--config", default="default", help="Robot configuration name")
    parser.add_argument("--config-dir", default="config/robots", help="Configuration directory")
    parser.add_argument("--robot-id", help="Override robot ID")
    parser.add_argument("--local-only", action="store_true", help="Run in local-only mode (no cloud)")
    parser.add_argument("--no-teleop", action="store_true", help="Disable teleoperation")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.robot_id:
        os.environ["ROBOT_ID"] = args.robot_id
    
    if args.local_only:
        os.environ["CLOUD_ENABLED"] = "false"
    
    if args.no_teleop:
        os.environ["TELEOP_ENABLED"] = "false"
    
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
    
    print(f"🤖 Starting Eidolon Robot System")
    print(f"📋 Configuration: {args.config}")
    print(f"📁 Config directory: {args.config_dir}")
    
    # Create robot system
    robot_system = RobotSystem(args.config, args.config_dir)
    
    try:
        # Start robot system
        await robot_system.start()
        
        # Print system status
        status = robot_system.get_system_status()
        print(f"✅ Robot system started")
        print(f"   Hardware ready: {status['hardware_ready']}")
        print(f"   Safety OK: {status['safety_ok']}")
        print(f"   Control active: {status['control_active']}")
        print(f"   Cloud connected: {status['cloud_connected']}")
        
        # Keep running until shutdown
        while robot_system.running:
            await asyncio.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
    except Exception as e:
        print(f"❌ Robot system error: {e}")
        sys.exit(1)
    finally:
        # Ensure clean shutdown
        await robot_system.stop()
        print("✅ Robot system stopped")

if __name__ == "__main__":
    asyncio.run(main())
