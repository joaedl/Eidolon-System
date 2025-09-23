"""Main entry point for the cloud server."""

import asyncio
import signal
import sys
from typing import Optional
import structlog
import uvicorn
from fastapi import FastAPI

from .api_gateway import app as api_app
from .orchestrator import Orchestrator
from .planner import PlannerService
from ..common.config import get_config

logger = structlog.get_logger(__name__)


class CloudServer:
    """Main cloud server coordinator."""
    
    def __init__(self):
        self.config = get_config()
        self.orchestrator: Optional[Orchestrator] = None
        self.planner: Optional[PlannerService] = None
        self.running = False
    
    async def start(self):
        """Start the cloud server."""
        logger.info("Starting cloud server")
        
        try:
            # Initialize orchestrator
            self.orchestrator = Orchestrator()
            await self.orchestrator.start()
            
            # Initialize planner
            self.planner = PlannerService()
            await self.planner.start()
            
            self.running = True
            logger.info("Cloud server started successfully")
            
            # Keep running until shutdown
            while self.running:
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error("Failed to start cloud server", error=str(e))
            raise
    
    async def stop(self):
        """Stop the cloud server."""
        logger.info("Stopping cloud server")
        self.running = False
        
        if self.orchestrator:
            await self.orchestrator.stop()
        
        if self.planner:
            await self.planner.stop()
        
        logger.info("Cloud server stopped")
    
    def get_status(self) -> dict:
        """Get cloud server status."""
        status = {
            "running": self.running,
            "orchestrator": self.orchestrator.get_statistics() if self.orchestrator else None,
            "planner": {
                "available_policies": self.planner.get_available_policies() if self.planner else []
            }
        }
        return status


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
    
    # Create cloud server
    cloud_server = CloudServer()
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        asyncio.create_task(cloud_server.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start cloud server
        await cloud_server.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error("Cloud server error", error=str(e))
        sys.exit(1)
    finally:
        # Ensure clean shutdown
        await cloud_server.stop()


def run_api_server():
    """Run the API server."""
    uvicorn.run(
        api_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        run_api_server()
    else:
        asyncio.run(main())
