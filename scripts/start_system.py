#!/usr/bin/env python3
"""Start the Eidolon system components."""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

async def start_component(name: str, module: str, port: int = None):
    """Start a system component."""
    print(f"Starting {name}...")
    
    cmd = [sys.executable, "-m", module]
    if port:
        cmd.extend(["--port", str(port)])
    
    process = subprocess.Popen(cmd)
    
    # Wait a moment for startup
    await asyncio.sleep(2)
    
    if process.poll() is None:
        print(f"{name} started successfully (PID: {process.pid})")
        return process
    else:
        print(f"Failed to start {name}")
        return None

async def main():
    """Start all system components."""
    print("Starting Eidolon Robot Fleet Management System...")
    
    # Start infrastructure services
    print("Starting infrastructure services...")
    subprocess.run(["docker-compose", "up", "-d"], check=True)
    
    # Wait for services to be ready
    print("Waiting for infrastructure services...")
    await asyncio.sleep(10)
    
    # Start system components
    components = [
        ("Cloud Server", "eidolon.cloud.main", 8000),
        ("Teleop Gateway", "eidolon.teleop.signaling", 8001),
        ("Operator Console", "eidolon.operator.console", 8002),
        ("Customer Dashboard", "eidolon.dashboard.web_ui", 8003),
    ]
    
    processes = []
    
    for name, module, port in components:
        process = await start_component(name, module, port)
        if process:
            processes.append((name, process))
    
    print("\nSystem components started:")
    for name, process in processes:
        print(f"  {name}: PID {process.pid}")
    
    print("\nSystem is running. Press Ctrl+C to stop.")
    
    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down system...")
        
        # Stop all processes
        for name, process in processes:
            print(f"Stopping {name}...")
            process.terminate()
            process.wait()
        
        # Stop infrastructure services
        print("Stopping infrastructure services...")
        subprocess.run(["docker-compose", "down"])
        
        print("System stopped.")

if __name__ == "__main__":
    asyncio.run(main())
