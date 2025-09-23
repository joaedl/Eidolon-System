"""gRPC teleop service implementation."""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional
import structlog
import grpc
from grpc import aio

from ..common.config import get_config
from ..common.security import SecurityManager
from ..monitoring.metrics import MetricsCollector
from ..monitoring.security import SecurityMonitor

logger = structlog.get_logger(__name__)


class TeleopServiceServicer:
    """gRPC teleop service implementation."""
    
    def __init__(self, metrics_collector: MetricsCollector, security_monitor: SecurityMonitor):
        self.metrics_collector = metrics_collector
        self.security_monitor = security_monitor
        self.config = get_config()
        self.security_manager = SecurityManager(self.config)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_recordings: Dict[str, Dict[str, Any]] = {}
    
    async def CreateSession(self, request, context):
        """Handle teleop session creation."""
        try:
            robot_id = request.robot_id
            operator_id = request.operator_id
            tenant_id = request.tenant_id
            session_type = request.session_type
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Create session token
            session_token = self.security_manager.create_operator_token(
                operator_id, tenant_id, ["teleop"]
            )
            
            # Generate TURN credentials
            turn_credentials = {
                "username": f"session_{session_id}",
                "password": f"password_{session_id}",
                "server": self.config.webrtc.turn_server,
                "port": str(self.config.webrtc.turn_port),
                "expires_at": int(time.time() + 3600)  # 1 hour
            }
            
            # Create SDP offer (simplified)
            sdp_offer = f"v=0\r\no=- {int(time.time())} 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0\r\na=msid-semantic: WMS\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\nc=IN IP4 127.0.0.1\r\na=ice-ufrag:test\r\na=ice-pwd:test\r\na=ice-options:trickle\r\na=fingerprint:sha-256 test\r\na=setup:actpass\r\na=mid:0\r\na=sctp-port:5000\r\na=max-message-size:262144\r\n"
            
            # Store session info
            self.active_sessions[session_id] = {
                "session_id": session_id,
                "robot_id": robot_id,
                "operator_id": operator_id,
                "tenant_id": tenant_id,
                "session_type": session_type,
                "created_at": time.time(),
                "status": "created",
                "permissions": request.permissions
            }
            
            # Record metrics
            self.metrics_collector.prometheus_metrics.record_teleop_session(tenant_id, True)
            
            # Record security audit
            self.security_monitor.record_teleop_session(
                session_id, operator_id, robot_id, tenant_id, "127.0.0.1"
            )
            
            logger.info("Teleop session created", session_id=session_id, robot_id=robot_id, operator_id=operator_id)
            
            # Return session creation response
            from proto.teleop_pb2 import CreateSessionResp, TurnCredentials
            return CreateSessionResp(
                success=True,
                session_id=session_id,
                session_token=session_token,
                turn_credentials=TurnCredentials(
                    username=turn_credentials["username"],
                    password=turn_credentials["password"],
                    server=turn_credentials["server"],
                    port=int(turn_credentials["port"]),
                    expires_at=turn_credentials["expires_at"]
                ),
                sdp_offer=sdp_offer,
                ice_servers=[f"stun:{server}" for server in self.config.webrtc.stun_servers],
                expires_at=int(time.time() + 3600)
            )
            
        except Exception as e:
            logger.error("Session creation failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Session creation failed: {str(e)}")
            return None
    
    async def Signal(self, request, context):
        """Handle WebRTC signaling."""
        try:
            session_id = request.session_id
            signal_type = request.signal_type
            signal_data = request.signal_data
            
            if session_id not in self.active_sessions:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Session not found")
                return None
            
            # Process signaling message
            if signal_type == "offer":
                # Handle SDP offer
                logger.debug("Received SDP offer", session_id=session_id)
            elif signal_type == "answer":
                # Handle SDP answer
                logger.debug("Received SDP answer", session_id=session_id)
            elif signal_type == "ice_candidate":
                # Handle ICE candidate
                logger.debug("Received ICE candidate", session_id=session_id)
            
            # Return signaling acknowledgment
            from proto.teleop_pb2 import SignalAck
            return SignalAck(
                success=True,
                message="Signal processed",
                response_signal=signal_data  # Echo back for now
            )
            
        except Exception as e:
            logger.error("Signaling failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Signaling failed: {str(e)}")
            return None
    
    async def SendControl(self, request, context):
        """Handle control commands."""
        try:
            session_id = request.session_id
            command_type = request.command_type
            parameters = request.parameters
            operator_id = request.operator_id
            
            if session_id not in self.active_sessions:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Session not found")
                return None
            
            session = self.active_sessions[session_id]
            
            # Validate command permissions
            if command_type == "emergency_stop" and "emergency" not in session.get("permissions", []):
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details("No permission for emergency stop")
                return None
            
            # Record metrics
            self.metrics_collector.prometheus_metrics.record_teleop_command(session_id, session["tenant_id"])
            
            # Record security audit
            if command_type == "emergency_stop":
                self.security_monitor.record_emergency_stop(
                    session["robot_id"], operator_id, session["tenant_id"], "127.0.0.1"
                )
            
            logger.debug("Control command received", session_id=session_id, command_type=command_type)
            
            # Return control acknowledgment
            from proto.teleop_pb2 import ControlAck
            return ControlAck(
                success=True,
                message="Control command processed",
                processed_at=int(time.time())
            )
            
        except Exception as e:
            logger.error("Control command failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Control command failed: {str(e)}")
            return None
    
    async def GetSafetyStatus(self, request, context):
        """Get safety status for robot."""
        try:
            robot_id = request
            
            # Get safety status (simplified)
            from proto.teleop_pb2 import SafetyStatus
            return SafetyStatus(
                robot_id=robot_id,
                emergency_stop=False,
                safety_zone_violation=False,
                communication_lost=False,
                velocity_limit=1.0,
                force_limit=100.0,
                timestamp=int(time.time())
            )
            
        except Exception as e:
            logger.error("Safety status request failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Safety status request failed: {str(e)}")
            return None
    
    async def EndSession(self, request, context):
        """Handle session termination."""
        try:
            session_id = request.session_id
            reason = request.reason
            
            if session_id not in self.active_sessions:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Session not found")
                return None
            
            session = self.active_sessions[session_id]
            
            # Calculate session duration
            duration = time.time() - session["created_at"]
            
            # Record metrics
            self.metrics_collector.prometheus_metrics.record_teleop_session(session["tenant_id"], False)
            self.metrics_collector.prometheus_metrics.record_teleop_duration(session["tenant_id"], duration)
            
            # Create session recording info
            recording = {
                "session_id": session_id,
                "robot_id": session["robot_id"],
                "operator_id": session["operator_id"],
                "start_time": session["created_at"],
                "end_time": time.time(),
                "duration": duration,
                "recording_url": f"https://storage.eidolon.cloud/recordings/{session_id}.mp4",
                "encrypted": True,
                "encryption_key_id": f"key_{session_id}"
            }
            
            self.session_recordings[session_id] = recording
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info("Session ended", session_id=session_id, duration=duration, reason=reason)
            
            # Return session end response
            from proto.teleop_pb2 import EndSessionResp, SessionRecording
            return EndSessionResp(
                success=True,
                message="Session ended",
                recording=SessionRecording(
                    session_id=recording["session_id"],
                    robot_id=recording["robot_id"],
                    operator_id=recording["operator_id"],
                    start_time=int(recording["start_time"]),
                    end_time=int(recording["end_time"]),
                    recording_url=recording["recording_url"],
                    encrypted=recording["encrypted"],
                    encryption_key_id=recording["encryption_key_id"]
                )
            )
            
        except Exception as e:
            logger.error("Session end failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Session end failed: {str(e)}")
            return None
    
    async def StartRecording(self, request, context):
        """Start session recording."""
        try:
            session_id = request.session_id
            include_video = request.include_video
            include_audio = request.include_audio
            include_controls = request.include_controls
            encryption_key_id = request.encryption_key_id
            
            if session_id not in self.active_sessions:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Session not found")
                return None
            
            recording_id = f"recording_{session_id}_{int(time.time())}"
            
            logger.info("Recording started", session_id=session_id, recording_id=recording_id)
            
            # Return recording start response
            from proto.teleop_pb2 import StartRecordingResp
            return StartRecordingResp(
                success=True,
                recording_id=recording_id,
                message="Recording started"
            )
            
        except Exception as e:
            logger.error("Recording start failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Recording start failed: {str(e)}")
            return None
    
    async def StopRecording(self, request, context):
        """Stop session recording."""
        try:
            session_id = request.session_id
            recording_id = request.recording_id
            
            recording_url = f"https://storage.eidolon.cloud/recordings/{recording_id}.mp4"
            
            logger.info("Recording stopped", session_id=session_id, recording_id=recording_id)
            
            # Return recording stop response
            from proto.teleop_pb2 import StopRecordingResp
            return StopRecordingResp(
                success=True,
                message="Recording stopped",
                recording_url=recording_url
            )
            
        except Exception as e:
            logger.error("Recording stop failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Recording stop failed: {str(e)}")
            return None


async def create_teleop_service(metrics_collector: MetricsCollector, security_monitor: SecurityMonitor):
    """Create and configure teleop service."""
    from proto import teleop_pb2_grpc
    
    servicer = TeleopServiceServicer(metrics_collector, security_monitor)
    
    # Add servicer to gRPC server
    server = aio.server()
    teleop_pb2_grpc.add_TeleopServiceServicer_to_server(servicer, server)
    
    return server
