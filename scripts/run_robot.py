#!/usr/bin/env python3
"""Run a robot edge instance."""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eidolon.robot.main import main

if __name__ == "__main__":
    # Set robot configuration via environment variables
    os.environ.setdefault("ROBOT_ID", "robot-001")
    os.environ.setdefault("ROBOT_TENANT_ID", "tenant-001")
    os.environ.setdefault("SAFETY_ENABLED", "true")
    os.environ.setdefault("AUTONOMOUS_MODE", "true")
    
    asyncio.run(main())
