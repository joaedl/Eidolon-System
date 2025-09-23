"""Teleoperation interface for robot control."""

import asyncio
import time
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import structlog

from .config import RobotConfig
from .hardware import HardwareManager
from .control import ControlManager, ControlCommand, ControlMode

logger = structlog.get_logger(__name__)


class TeleopMode(Enum):
    """Teleoperation modes."""
    DISABLED = "disabled"
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    IMPEDANCE = "impedance"


@dataclass
class TeleopCommand:
    """Teleoperation command."""
    command_type: str
    arm_id: Optional[str] = None
    joint_name: Optional[str] = None
    position: Optional[float] = None
    velocity: Optional[float] = None
    torque: Optional[float] = None
    target_pose: Optional[List[float]] = None
    timestamp: float = 0.0


class TeleopInterface:
    """Teleoperation interface for robot control."""
    
    def __init__(self, robot_config: RobotConfig, hardware_manager: HardwareManager, control_manager: ControlManager):
        self.robot_config = robot_config
        self.hardware_manager = hardware_manager
        self.control_manager = control_manager
        
        self.teleop_mode = TeleopMode.DISABLED
        self.active_arm = None
        self.active_joints: List[str] = []
        self.running = False
        
        # Command processing
        self.command_queue: List[TeleopCommand] = []
        self.last_command_time = 0.0
        self.command_rate = 10.0  # Hz
        
        # Safety limits
        self.max_velocity = 1.0
        self.max_acceleration = 2.0
        self.workspace_limits = robot_config.workspace_limits
        
        # Callbacks
        self.on_command_received: Optional[Callable] = None
        self.on_safety_violation: Optional[Callable] = None
    
    async def start(self):
        """Start the teleoperation interface."""
        self.running = True
        logger.info("Teleop interface started", robot_id=self.robot_config.robot_id)
    
    async def stop(self):
        """Stop the teleoperation interface."""
        self.running = False
        self.teleop_mode = TeleopMode.DISABLED
        logger.info("Teleop interface stopped")
    
    def set_teleop_mode(self, mode: TeleopMode):
        """Set teleoperation mode."""
        self.teleop_mode = mode
        logger.info("Teleop mode changed", mode=mode.value)
    
    def set_active_arm(self, arm_id: str):
        """Set active arm for teleoperation."""
        if arm_id in [arm.arm_id for arm in self.robot_config.arms]:
            self.active_arm = arm_id
            self.active_joints = self.hardware_manager.get_arm_joints(arm_id)
            logger.info("Active arm set", arm_id=arm_id, joints=self.active_joints)
        else:
            logger.warning("Invalid arm ID", arm_id=arm_id)
    
    def set_active_joints(self, joint_names: List[str]):
        """Set active joints for teleoperation."""
        self.active_joints = joint_names
        logger.info("Active joints set", joints=joint_names)
    
    async def process_command(self, command: TeleopCommand):
        """Process a teleoperation command."""
        if not self.running or self.teleop_mode == TeleopMode.DISABLED:
            return
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_command_time < 1.0 / self.command_rate:
            return
        
        # Validate command
        if not self._validate_command(command):
            return
        
        # Process command based on type
        if command.command_type == "position":
            await self._process_position_command(command)
        elif command.command_type == "velocity":
            await self._process_velocity_command(command)
        elif command.command_type == "torque":
            await self._process_torque_command(command)
        elif command.command_type == "pose":
            await self._process_pose_command(command)
        elif command.command_type == "emergency_stop":
            await self._process_emergency_stop()
        else:
            logger.warning("Unknown command type", command_type=command.command_type)
        
        self.last_command_time = current_time
        
        # Trigger callback
        if self.on_command_received:
            self.on_command_received(command)
    
    def _validate_command(self, command: TeleopCommand) -> bool:
        """Validate teleoperation command."""
        # Check if arm is active
        if command.arm_id and command.arm_id != self.active_arm:
            logger.warning("Command for inactive arm", arm_id=command.arm_id, active_arm=self.active_arm)
            return False
        
        # Check joint validity
        if command.joint_name and command.joint_name not in self.active_joints:
            logger.warning("Command for inactive joint", joint=command.joint_name, active_joints=self.active_joints)
            return False
        
        # Check velocity limits
        if command.velocity and abs(command.velocity) > self.max_velocity:
            logger.warning("Velocity limit exceeded", velocity=command.velocity, limit=self.max_velocity)
            if self.on_safety_violation:
                self.on_safety_violation("velocity_limit_exceeded")
            return False
        
        return True
    
    async def _process_position_command(self, command: TeleopCommand):
        """Process position command."""
        if command.joint_name and command.position is not None:
            await self.hardware_manager.set_joint_position(command.joint_name, command.position)
            logger.debug("Position command executed", joint=command.joint_name, position=command.position)
        elif command.arm_id and command.target_pose:
            await self._move_arm_to_pose(command.arm_id, command.target_pose)
    
    async def _process_velocity_command(self, command: TeleopCommand):
        """Process velocity command."""
        if command.joint_name and command.velocity is not None:
            await self.hardware_manager.set_joint_velocity(command.joint_name, command.velocity)
            logger.debug("Velocity command executed", joint=command.joint_name, velocity=command.velocity)
    
    async def _process_torque_command(self, command: TeleopCommand):
        """Process torque command."""
        if command.joint_name and command.torque is not None:
            await self.hardware_manager.set_joint_torque(command.joint_name, command.torque)
            logger.debug("Torque command executed", joint=command.joint_name, torque=command.torque)
    
    async def _process_pose_command(self, command: TeleopCommand):
        """Process pose command."""
        if command.arm_id and command.target_pose:
            await self._move_arm_to_pose(command.arm_id, command.target_pose)
    
    async def _process_emergency_stop(self):
        """Process emergency stop command."""
        logger.critical("Emergency stop activated via teleop")
        await self.control_manager.clear_command_queue()
        # Additional emergency stop actions would go here
    
    async def _move_arm_to_pose(self, arm_id: str, target_pose: List[float]):
        """Move arm to target pose."""
        joint_names = self.hardware_manager.get_arm_joints(arm_id)
        
        if len(target_pose) != len(joint_names):
            logger.error("Pose dimension mismatch", arm=arm_id, expected=len(joint_names), got=len(target_pose))
            return
        
        # Move all joints simultaneously
        for joint_name, position in zip(joint_names, target_pose):
            await self.hardware_manager.set_joint_position(joint_name, position)
        
        logger.debug("Arm pose command executed", arm=arm_id, pose=target_pose)
    
    def get_teleop_status(self) -> Dict[str, Any]:
        """Get teleoperation status."""
        return {
            "enabled": self.teleop_mode != TeleopMode.DISABLED,
            "mode": self.teleop_mode.value,
            "active_arm": self.active_arm,
            "active_joints": self.active_joints,
            "command_rate": self.command_rate,
            "max_velocity": self.max_velocity,
            "workspace_limits": self.workspace_limits
        }
    
    def get_arm_workspace(self, arm_id: str) -> Dict[str, float]:
        """Get workspace limits for an arm."""
        for arm in self.robot_config.arms:
            if arm.arm_id == arm_id:
                return arm.workspace_limits
        return {}
    
    def get_joint_limits(self, joint_name: str) -> Dict[str, float]:
        """Get joint limits."""
        for arm in self.robot_config.arms:
            for joint in arm.joints:
                if joint.name == joint_name:
                    return {
                        "min_position": joint.min_position,
                        "max_position": joint.max_position,
                        "max_velocity": joint.max_velocity,
                        "max_torque": joint.max_torque
                    }
        
        if self.robot_config.head:
            for joint in self.robot_config.head.joints:
                if joint.name == joint_name:
                    return {
                        "min_position": joint.min_position,
                        "max_position": joint.max_position,
                        "max_velocity": joint.max_velocity,
                        "max_torque": joint.max_torque
                    }
        
        return {}


class TeleopCommandProcessor:
    """Process teleoperation commands from various sources."""
    
    def __init__(self, teleop_interface: TeleopInterface):
        self.teleop_interface = teleop_interface
        self.running = False
        self._processor_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start command processing."""
        self.running = True
        self._processor_task = asyncio.create_task(self._process_commands())
        logger.info("Teleop command processor started")
    
    async def stop(self):
        """Stop command processing."""
        self.running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Teleop command processor stopped")
    
    async def _process_commands(self):
        """Process command queue."""
        while self.running:
            try:
                if self.teleop_interface.command_queue:
                    command = self.teleop_interface.command_queue.pop(0)
                    await self.teleop_interface.process_command(command)
                
                await asyncio.sleep(0.01)  # 100Hz processing
            except Exception as e:
                logger.error("Command processing error", error=str(e))
                await asyncio.sleep(0.1)
    
    def add_command(self, command: TeleopCommand):
        """Add command to queue."""
        command.timestamp = time.time()
        self.teleop_interface.command_queue.append(command)
        logger.debug("Command added to queue", command_type=command.command_type)
    
    def add_position_command(self, joint_name: str, position: float, arm_id: str = None):
        """Add position command."""
        command = TeleopCommand(
            command_type="position",
            joint_name=joint_name,
            position=position,
            arm_id=arm_id
        )
        self.add_command(command)
    
    def add_velocity_command(self, joint_name: str, velocity: float, arm_id: str = None):
        """Add velocity command."""
        command = TeleopCommand(
            command_type="velocity",
            joint_name=joint_name,
            velocity=velocity,
            arm_id=arm_id
        )
        self.add_command(command)
    
    def add_pose_command(self, arm_id: str, target_pose: List[float]):
        """Add pose command."""
        command = TeleopCommand(
            command_type="pose",
            arm_id=arm_id,
            target_pose=target_pose
        )
        self.add_command(command)
    
    def add_emergency_stop(self):
        """Add emergency stop command."""
        command = TeleopCommand(command_type="emergency_stop")
        self.add_command(command)
