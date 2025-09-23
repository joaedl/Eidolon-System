"""gRPC device service implementation."""

import asyncio
import time
from typing import Dict, Any, Optional
import structlog
import grpc
from grpc import aio

from ..common.config import get_config
from ..common.security import SecurityManager
from ..cloud.orchestrator import Orchestrator
from ..monitoring.metrics import MetricsCollector

logger = structlog.get_logger(__name__)


class DeviceServiceServicer:
    """gRPC device service implementation."""
    
    def __init__(self, orchestrator: Orchestrator, metrics_collector: MetricsCollector):
        self.orchestrator = orchestrator
        self.metrics_collector = metrics_collector
        self.config = get_config()
        self.security_manager = SecurityManager(self.config)
        self.registered_devices: Dict[str, Dict[str, Any]] = {}
        self.device_heartbeats: Dict[str, float] = {}
    
    async def Register(self, request, context):
        """Handle device registration."""
        try:
            # Validate device certificate (in real implementation)
            device_id = request.device_id
            tenant_id = request.tenant_id
            
            # Create session token
            session_token = self.security_manager.create_device_token(
                device_id, tenant_id
            )
            
            # Register device with orchestrator
            from ..cloud.orchestrator import Robot
            robot = Robot(
                id=device_id,
                name=f"Robot {device_id}",
                tenant_id=tenant_id,
                capabilities=request.capabilities
            )
            self.orchestrator.register_robot(robot)
            
            # Store device info
            self.registered_devices[device_id] = {
                "device_id": device_id,
                "tenant_id": tenant_id,
                "hardware_revision": request.hardware_revision,
                "firmware_version": request.firmware_version,
                "capabilities": request.capabilities,
                "registered_at": time.time()
            }
            
            # Record metrics
            self.metrics_collector.prometheus_metrics.record_robot_online(device_id, tenant_id, True)
            
            logger.info("Device registered", device_id=device_id, tenant_id=tenant_id)
            
            # Return registration acknowledgment
            from proto.device_pb2 import RegisterAck
            return RegisterAck(
                success=True,
                session_token=session_token,
                expires_at=int(time.time() + 3600),  # 1 hour
                allowed_services=["telemetry", "subgoal", "heartbeat"]
            )
            
        except Exception as e:
            logger.error("Device registration failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Registration failed: {str(e)}")
            return None
    
    async def Heartbeat(self, request, context):
        """Handle device heartbeat."""
        try:
            device_id = request.device_id
            current_time = time.time()
            
            # Update heartbeat timestamp
            self.device_heartbeats[device_id] = current_time
            
            # Update robot status in orchestrator
            if device_id in self.registered_devices:
                tenant_id = self.registered_devices[device_id]["tenant_id"]
                self.orchestrator.update_robot_status(
                    device_id, request.status, 
                    battery_level=request.metrics.battery_level,
                    last_seen=current_time
                )
                
                # Record metrics
                self.metrics_collector.prometheus_metrics.record_robot_battery(
                    device_id, tenant_id, request.metrics.battery_level
                )
                
                if request.metrics.safety_violations_count > 0:
                    self.metrics_collector.prometheus_metrics.record_safety_violation(
                        device_id, tenant_id, "general"
                    )
            
            logger.debug("Device heartbeat received", device_id=device_id, status=request.status)
            
            # Return heartbeat acknowledgment
            from proto.device_pb2 import HeartbeatAck
            return HeartbeatAck(
                success=True,
                message="Heartbeat received",
                server_time=int(current_time)
            )
            
        except Exception as e:
            logger.error("Heartbeat processing failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Heartbeat failed: {str(e)}")
            return None
    
    async def StreamTelemetry(self, request_iterator, context):
        """Handle telemetry streaming."""
        try:
            processed_count = 0
            
            async for telemetry in request_iterator:
                device_id = telemetry.device_id
                
                # Process telemetry data
                if device_id in self.registered_devices:
                    tenant_id = self.registered_devices[device_id]["tenant_id"]
                    
                    # Record metrics
                    for metric in telemetry.metrics:
                        if metric.name == "cpu_usage":
                            self.metrics_collector.prometheus_metrics.metrics['cpu_usage'].set(metric.value)
                        elif metric.name == "memory_usage":
                            self.metrics_collector.prometheus_metrics.metrics['memory_usage'].set(metric.value)
                    
                    # Record robot commands if any
                    if "commands_sent" in telemetry.metadata:
                        self.metrics_collector.prometheus_metrics.record_robot_command(
                            device_id, tenant_id, "telemetry"
                        )
                
                processed_count += 1
                
                # Send acknowledgment
                from proto.device_pb2 import StreamAck
                yield StreamAck(
                    success=True,
                    message="Telemetry processed",
                    processed_count=processed_count
                )
            
            logger.info("Telemetry stream completed", processed_count=processed_count)
            
        except Exception as e:
            logger.error("Telemetry streaming failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Telemetry streaming failed: {str(e)}")
    
    async def GetSubgoal(self, request, context):
        """Handle subgoal requests."""
        try:
            device_id = request.device_id
            
            if device_id not in self.registered_devices:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Device not registered")
                return None
            
            # Get current robot state (simplified)
            from proto.device_pb2 import Subgoal, Pose
            from proto.planner_pb2 import Pose as PlannerPose
            
            # Create a simple subgoal (in real implementation, this would use the planner)
            subgoal = Subgoal(
                id=f"subgoal_{int(time.time())}",
                target=Pose(
                    x=1.0,
                    y=2.0,
                    z=0.5,
                    qx=0.0,
                    qy=0.0,
                    qz=0.0,
                    qw=1.0
                ),
                deadline=int(time.time() + 300),  # 5 minutes
                priority=1
            )
            
            logger.debug("Subgoal provided", device_id=device_id, subgoal_id=subgoal.id)
            
            return subgoal
            
        except Exception as e:
            logger.error("Subgoal request failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Subgoal request failed: {str(e)}")
            return None
    
    async def AcknowledgeSubgoal(self, request, context):
        """Handle subgoal acknowledgment."""
        try:
            device_id = request.device_id
            subgoal_id = request.subgoal_id
            accepted = request.accepted
            
            if accepted:
                logger.info("Subgoal accepted", device_id=device_id, subgoal_id=subgoal_id)
            else:
                logger.warning("Subgoal rejected", device_id=device_id, subgoal_id=subgoal_id, reason=request.reason)
            
            # Return acknowledgment response
            from proto.device_pb2 import AckResponse
            return AckResponse(
                success=True,
                message="Subgoal acknowledgment received"
            )
            
        except Exception as e:
            logger.error("Subgoal acknowledgment failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Subgoal acknowledgment failed: {str(e)}")
            return None


async def create_device_service(orchestrator: Orchestrator, metrics_collector: MetricsCollector):
    """Create and configure device service."""
    from proto import device_pb2_grpc
    
    servicer = DeviceServiceServicer(orchestrator, metrics_collector)
    
    # Add servicer to gRPC server
    server = aio.server()
    device_pb2_grpc.add_DeviceServiceServicer_to_server(servicer, server)
    
    return server
