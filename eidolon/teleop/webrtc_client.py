"""WebRTC client for teleoperation."""

import asyncio
import json
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import structlog
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.signaling import TcpSocketSignaling
import websockets

logger = structlog.get_logger(__name__)


class ConnectionState(Enum):
    """WebRTC connection state."""
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class TeleopConfig:
    """Teleoperation configuration."""
    session_id: str
    robot_id: str
    operator_id: str
    tenant_id: str
    signaling_url: str
    turn_servers: List[Dict[str, str]]
    stun_servers: List[Dict[str, str]]


class WebRTCClient:
    """WebRTC client for teleoperation."""
    
    def __init__(self, config: TeleopConfig):
        self.config = config
        self.pc: Optional[RTCPeerConnection] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connection_state = ConnectionState.NEW
        self.running = False
        
        # Callbacks
        self.on_connection_state_change: Optional[Callable] = None
        self.on_data_channel_open: Optional[Callable] = None
        self.on_data_channel_message: Optional[Callable] = None
        self.on_video_frame: Optional[Callable] = None
        self.on_audio_frame: Optional[Callable] = None
    
    async def start(self):
        """Start the WebRTC client."""
        self.running = True
        
        # Create peer connection
        self.pc = RTCPeerConnection()
        
        # Set up event handlers
        self._setup_event_handlers()
        
        # Connect to signaling server
        await self._connect_signaling()
        
        logger.info("WebRTC client started", session_id=self.config.session_id)
    
    async def stop(self):
        """Stop the WebRTC client."""
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
        
        if self.pc:
            await self.pc.close()
        
        self.connection_state = ConnectionState.CLOSED
        logger.info("WebRTC client stopped")
    
    def _setup_event_handlers(self):
        """Set up WebRTC event handlers."""
        @self.pc.on("connectionstatechange")
        def on_connectionstatechange():
            self.connection_state = ConnectionState(self.pc.connectionState)
            logger.info("Connection state changed", state=self.connection_state.value)
            
            if self.on_connection_state_change:
                self.on_connection_state_change(self.connection_state)
        
        @self.pc.on("datachannel")
        def on_datachannel(channel):
            logger.info("Data channel opened", label=channel.label)
            
            @channel.on("open")
            def on_open():
                if self.on_data_channel_open:
                    self.on_data_channel_open(channel)
            
            @channel.on("message")
            def on_message(message):
                if self.on_data_channel_message:
                    self.on_data_channel_message(channel, message)
    
    async def _connect_signaling(self):
        """Connect to signaling server."""
        try:
            # Connect to WebSocket
            self.websocket = await websockets.connect(self.config.signaling_url)
            
            # Start signaling loop
            asyncio.create_task(self._signaling_loop())
            
        except Exception as e:
            logger.error("Failed to connect to signaling server", error=str(e))
            raise
    
    async def _signaling_loop(self):
        """Main signaling loop."""
        try:
            while self.running and self.websocket:
                message = await self.websocket.recv()
                await self._handle_signaling_message(json.loads(message))
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Signaling connection closed")
        except Exception as e:
            logger.error("Signaling loop error", error=str(e))
    
    async def _handle_signaling_message(self, message: Dict[str, Any]):
        """Handle signaling message."""
        message_type = message.get("type")
        data = message.get("data", {})
        
        if message_type == "offer":
            await self._handle_offer(data)
        elif message_type == "answer":
            await self._handle_answer(data)
        elif message_type == "ice_candidate":
            await self._handle_ice_candidate(data)
        else:
            logger.warning("Unknown signaling message type", type=message_type)
    
    async def _handle_offer(self, offer_data: Dict[str, Any]):
        """Handle SDP offer."""
        try:
            offer = RTCSessionDescription(
                sdp=offer_data["sdp"],
                type=offer_data["type"]
            )
            
            await self.pc.setRemoteDescription(offer)
            
            # Create answer
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            
            # Send answer
            await self._send_signaling_message("answer", {
                "sdp": answer.sdp,
                "type": answer.type
            })
            
        except Exception as e:
            logger.error("Error handling offer", error=str(e))
    
    async def _handle_answer(self, answer_data: Dict[str, Any]):
        """Handle SDP answer."""
        try:
            answer = RTCSessionDescription(
                sdp=answer_data["sdp"],
                type=answer_data["type"]
            )
            
            await self.pc.setRemoteDescription(answer)
            
        except Exception as e:
            logger.error("Error handling answer", error=str(e))
    
    async def _handle_ice_candidate(self, candidate_data: Dict[str, Any]):
        """Handle ICE candidate."""
        try:
            candidate = RTCIceCandidate(
                candidate=candidate_data["candidate"],
                sdpMid=candidate_data.get("sdpMid"),
                sdpMLineIndex=candidate_data.get("sdpMLineIndex")
            )
            
            await self.pc.addIceCandidate(candidate)
            
        except Exception as e:
            logger.error("Error handling ICE candidate", error=str(e))
    
    async def _send_signaling_message(self, message_type: str, data: Dict[str, Any]):
        """Send signaling message."""
        if self.websocket:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": time.time()
            }
            await self.websocket.send(json.dumps(message))
    
    async def create_offer(self):
        """Create and send SDP offer."""
        try:
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            await self._send_signaling_message("offer", {
                "sdp": offer.sdp,
                "type": offer.type
            })
            
        except Exception as e:
            logger.error("Error creating offer", error=str(e))
    
    async def create_data_channel(self, label: str) -> Any:
        """Create a data channel."""
        return self.pc.createDataChannel(label)
    
    async def send_control_command(self, command: Dict[str, Any]):
        """Send control command through data channel."""
        # This would send through the control data channel
        logger.debug("Sending control command", command=command)
    
    def get_connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self.connection_state
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connection_state == ConnectionState.CONNECTED


class TeleopClient:
    """High-level teleoperation client."""
    
    def __init__(self, config: TeleopConfig):
        self.config = config
        self.webrtc_client = WebRTCClient(config)
        self.running = False
        self.control_channel: Optional[Any] = None
        self.video_channel: Optional[Any] = None
    
    async def start(self):
        """Start the teleoperation client."""
        self.running = True
        
        # Set up WebRTC client callbacks
        self.webrtc_client.on_connection_state_change = self._on_connection_state_change
        self.webrtc_client.on_data_channel_open = self._on_data_channel_open
        self.webrtc_client.on_data_channel_message = self._on_data_channel_message
        
        # Start WebRTC client
        await self.webrtc_client.start()
        
        logger.info("Teleop client started", session_id=self.config.session_id)
    
    async def stop(self):
        """Stop the teleoperation client."""
        self.running = False
        await self.webrtc_client.stop()
        logger.info("Teleop client stopped")
    
    def _on_connection_state_change(self, state: ConnectionState):
        """Handle connection state change."""
        logger.info("Teleop connection state changed", state=state.value)
    
    def _on_data_channel_open(self, channel):
        """Handle data channel open."""
        if channel.label == "control":
            self.control_channel = channel
            logger.info("Control channel opened")
        elif channel.label == "video":
            self.video_channel = channel
            logger.info("Video channel opened")
    
    def _on_data_channel_message(self, channel, message):
        """Handle data channel message."""
        if channel.label == "control":
            self._handle_control_message(message)
        elif channel.label == "video":
            self._handle_video_message(message)
    
    def _handle_control_message(self, message):
        """Handle control message."""
        logger.debug("Received control message", message=message)
    
    def _handle_video_message(self, message):
        """Handle video message."""
        logger.debug("Received video message", size=len(message))
    
    async def send_velocity_command(self, linear: List[float], angular: List[float]):
        """Send velocity command."""
        command = {
            "type": "velocity",
            "linear": linear,
            "angular": angular,
            "timestamp": time.time()
        }
        
        if self.control_channel:
            await self.control_channel.send(json.dumps(command))
    
    async def send_position_command(self, position: List[float], orientation: List[float]):
        """Send position command."""
        command = {
            "type": "position",
            "position": position,
            "orientation": orientation,
            "timestamp": time.time()
        }
        
        if self.control_channel:
            await self.control_channel.send(json.dumps(command))
    
    async def send_emergency_stop(self):
        """Send emergency stop command."""
        command = {
            "type": "emergency_stop",
            "timestamp": time.time()
        }
        
        if self.control_channel:
            await self.control_channel.send(json.dumps(command))
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.webrtc_client.is_connected()
