"""Main entry point for the robot edge system."""

import asyncio
import signal
import sys
from typing import Optional
import structlog

from .brain_client import BrainClient
from ..common.config import get_config, get_robot_config

logger = structlog.get_logger(__name__)


class RobotSystem:
    """Main robot system coordinator."""
    
    def __init__(self):
        self.config = get_config()
        self.robot_config = get_robot_config()
        self.brain_client: Optional[BrainClient] = None
        self.running = False
    
    async def start(self):
        """Start the robot system."""
        logger.info("Starting robot system", 
                   robot_id=self.robot_config.robot_id,
                   tenant_id=self.robot_config.tenant_id)
        
        try:
            # Initialize brain client
            self.brain_client = BrainClient(
                self.robot_config.robot_id,
                self.robot_config.tenant_id
            )
            
            # Initialize and start brain client
            await self.brain_client.initialize()
            await self.brain_client.start()
            
            self.running = True
            logger.info("Robot system started successfully")
            
            # Keep running until shutdown
            while self.running:
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error("Failed to start robot system", error=str(e))
            raise
    
    async def stop(self):
        """Stop the robot system."""
        logger.info("Stopping robot system")
        self.running = False
        
        if self.brain_client:
            await self.brain_client.stop()
        
        logger.info("Robot system stopped")
    
    def get_status(self) -> dict:
        """Get robot system status."""
        if self.brain_client:
            return self.brain_client.get_robot_status()
        return {"status": "not_initialized"}


async def main():
    """Main entry point."""
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
    
    # Create robot system
    robot_system = RobotSystem()
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        asyncio.create_task(robot_system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start robot system
        await robot_system.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error("Robot system error", error=str(e))
        sys.exit(1)
    finally:
        # Ensure clean shutdown
        await robot_system.stop()


if __name__ == "__main__":
    asyncio.run(main())
