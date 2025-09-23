"""WebRTC signaling server for teleoperation sessions."""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import jwt

from ..common.config import get_config
from ..common.security import SecurityManager

logger = structlog.get_logger(__name__)


class SessionStatus(Enum):
    """Session status enumeration."""
    CREATED = "created"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


@dataclass
class TeleopSession:
    """Teleoperation session."""
    id: str
    robot_id: str
    operator_id: str
    tenant_id: str
    status: SessionStatus = SessionStatus.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=1))
    robot_websocket: Optional[WebSocket] = None
    operator_websocket: Optional[WebSocket] = None
    turn_credentials: Dict[str, str] = field(default_factory=dict)
    session_token: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalingMessage:
    """WebRTC signaling message."""
    type: str  # offer, answer, ice_candidate, error
    data: Dict[str, Any]
    from_participant: str  # robot or operator
    timestamp: float = field(default_factory=time.time)


class SignalingServer:
    """WebRTC signaling server."""
    
    def __init__(self):
        self.config = get_config()
        self.security_manager = SecurityManager(self.config)
        self.sessions: Dict[str, TeleopSession] = {}
        self.robot_connections: Dict[str, str] = {}  # robot_id -> session_id
        self.operator_connections: Dict[str, str] = {}  # operator_id -> session_id
        self.running = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the signaling server."""
        self.running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Signaling server started")
    
    async def stop(self):
        """Stop the signaling server."""
        self.running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all sessions
        for session in self.sessions.values():
            await self._close_session(session)
        
        logger.info("Signaling server stopped")
    
    async def _cleanup_loop(self):
        """Cleanup expired sessions."""
        while self.running:
            try:
                await self._cleanup_expired_sessions()
                await asyncio.sleep(60.0)  # 1-minute cleanup interval
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(10.0)
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time > session.expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self._close_session(self.sessions[session_id])
            del self.sessions[session_id]
            logger.info("Expired session cleaned up", session_id=session_id)
    
    async def create_session(self, robot_id: str, operator_id: str, tenant_id: str) -> TeleopSession:
        """Create a new teleoperation session."""
        session_id = str(uuid.uuid4())
        
        # Generate TURN credentials
        turn_credentials = await self._generate_turn_credentials(session_id)
        
        # Create session token
        session_token = self.security_manager.create_operator_token(
            operator_id, tenant_id, ["teleop"]
        )
        
        session = TeleopSession(
            id=session_id,
            robot_id=robot_id,
            operator_id=operator_id,
            tenant_id=tenant_id,
            turn_credentials=turn_credentials,
            session_token=session_token
        )
        
        self.sessions[session_id] = session
        logger.info("Session created", session_id=session_id, robot_id=robot_id, operator_id=operator_id)
        
        return session
    
    async def _generate_turn_credentials(self, session_id: str) -> Dict[str, str]:
        """Generate TURN server credentials for session."""
        # In a real implementation, this would generate ephemeral credentials
        return {
            "username": f"session_{session_id}",
            "password": f"password_{session_id}",
            "server": self.config.webrtc.turn_server,
            "port": str(self.config.webrtc.turn_port)
        }
    
    async def connect_robot(self, session_id: str, websocket: WebSocket):
        """Connect robot to session."""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = self.sessions[session_id]
        session.robot_websocket = websocket
        session.status = SessionStatus.CONNECTING
        self.robot_connections[session.robot_id] = session_id
        
        logger.info("Robot connected", session_id=session_id, robot_id=session.robot_id)
    
    async def connect_operator(self, session_id: str, websocket: WebSocket):
        """Connect operator to session."""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = self.sessions[session_id]
        session.operator_websocket = websocket
        session.status = SessionStatus.CONNECTING
        self.operator_connections[session.operator_id] = session_id
        
        logger.info("Operator connected", session_id=session_id, operator_id=session.operator_id)
    
    async def handle_signaling_message(self, session_id: str, message: SignalingMessage):
        """Handle WebRTC signaling message."""
        if session_id not in self.sessions:
            logger.warning("Message for unknown session", session_id=session_id)
            return
        
        session = self.sessions[session_id]
        
        # Route message to appropriate participant
        if message.from_participant == "robot" and session.operator_websocket:
            await self._send_to_operator(session, message)
        elif message.from_participant == "operator" and session.robot_websocket:
            await self._send_to_robot(session, message)
        else:
            logger.warning("No target participant for message", 
                         session_id=session_id, 
                         from_participant=message.from_participant)
    
    async def _send_to_operator(self, session: TeleopSession, message: SignalingMessage):
        """Send message to operator."""
        try:
            await session.operator_websocket.send_text(json.dumps({
                "type": message.type,
                "data": message.data,
                "timestamp": message.timestamp
            }))
        except Exception as e:
            logger.error("Failed to send message to operator", error=str(e))
    
    async def _send_to_robot(self, session: TeleopSession, message: SignalingMessage):
        """Send message to robot."""
        try:
            await session.robot_websocket.send_text(json.dumps({
                "type": message.type,
                "data": message.data,
                "timestamp": message.timestamp
            }))
        except Exception as e:
            logger.error("Failed to send message to robot", error=str(e))
    
    async def _close_session(self, session: TeleopSession):
        """Close a session."""
        try:
            if session.robot_websocket:
                await session.robot_websocket.close()
        except Exception as e:
            logger.error("Error closing robot websocket", error=str(e))
        
        try:
            if session.operator_websocket:
                await session.operator_websocket.close()
        except Exception as e:
            logger.error("Error closing operator websocket", error=str(e))
        
        # Clean up connections
        if session.robot_id in self.robot_connections:
            del self.robot_connections[session.robot_id]
        if session.operator_id in self.operator_connections:
            del self.operator_connections[session.operator_id]
        
        session.status = SessionStatus.DISCONNECTED
        logger.info("Session closed", session_id=session.id)
    
    def get_session(self, session_id: str) -> Optional[TeleopSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def get_robot_session(self, robot_id: str) -> Optional[TeleopSession]:
        """Get session for robot."""
        session_id = self.robot_connections.get(robot_id)
        return self.sessions.get(session_id) if session_id else None
    
    def get_operator_session(self, operator_id: str) -> Optional[TeleopSession]:
        """Get session for operator."""
        session_id = self.operator_connections.get(operator_id)
        return self.sessions.get(session_id) if session_id else None


# FastAPI app for signaling server
app = FastAPI(title="Teleop Signaling Server", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global signaling server instance
signaling_server = SignalingServer()


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    await signaling_server.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    await signaling_server.stop()


@app.websocket("/ws/robot/{session_id}")
async def robot_websocket(websocket: WebSocket, session_id: str):
    """Robot WebSocket endpoint."""
    await websocket.accept()
    
    try:
        # Connect robot to session
        await signaling_server.connect_robot(session_id, websocket)
        
        # Handle messages from robot
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message = SignalingMessage(
                type=message_data.get("type", "unknown"),
                data=message_data.get("data", {}),
                from_participant="robot"
            )
            
            await signaling_server.handle_signaling_message(session_id, message)
            
    except WebSocketDisconnect:
        logger.info("Robot disconnected", session_id=session_id)
    except Exception as e:
        logger.error("Robot websocket error", session_id=session_id, error=str(e))


@app.websocket("/ws/operator/{session_id}")
async def operator_websocket(websocket: WebSocket, session_id: str):
    """Operator WebSocket endpoint."""
    await websocket.accept()
    
    try:
        # Connect operator to session
        await signaling_server.connect_operator(session_id, websocket)
        
        # Handle messages from operator
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message = SignalingMessage(
                type=message_data.get("type", "unknown"),
                data=message_data.get("data", {}),
                from_participant="operator"
            )
            
            await signaling_server.handle_signaling_message(session_id, message)
            
    except WebSocketDisconnect:
        logger.info("Operator disconnected", session_id=session_id)
    except Exception as e:
        logger.error("Operator websocket error", session_id=session_id, error=str(e))


@app.post("/api/v1/sessions")
async def create_session(session_request: dict):
    """Create a new teleoperation session."""
    robot_id = session_request.get("robot_id")
    operator_id = session_request.get("operator_id")
    tenant_id = session_request.get("tenant_id")
    
    if not all([robot_id, operator_id, tenant_id]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    session = await signaling_server.create_session(robot_id, operator_id, tenant_id)
    
    return {
        "session_id": session.id,
        "robot_id": session.robot_id,
        "operator_id": session.operator_id,
        "tenant_id": session.tenant_id,
        "status": session.status.value,
        "expires_at": session.expires_at.isoformat(),
        "turn_credentials": session.turn_credentials,
        "session_token": session.session_token,
        "websocket_urls": {
            "robot": f"/ws/robot/{session.id}",
            "operator": f"/ws/operator/{session.id}"
        }
    }


@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    session = signaling_server.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.id,
        "robot_id": session.robot_id,
        "operator_id": session.operator_id,
        "tenant_id": session.tenant_id,
        "status": session.status.value,
        "created_at": session.created_at.isoformat(),
        "expires_at": session.expires_at.isoformat()
    }


@app.delete("/api/v1/sessions/{session_id}")
async def close_session(session_id: str):
    """Close a session."""
    session = signaling_server.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    await signaling_server._close_session(session)
    del signaling_server.sessions[session_id]
    
    return {"status": "closed"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
