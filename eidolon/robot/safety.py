"""Safety systems for robot edge control."""

import asyncio
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class SafetyState(Enum):
    """Safety system states."""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"


@dataclass
class SafetyLimits:
    """Safety limits configuration."""
    max_velocity: float = 1.0
    max_acceleration: float = 2.0
    max_force: float = 100.0
    max_torque: float = 50.0
    safety_zone_radius: float = 2.0
    emergency_stop_timeout: float = 0.1


@dataclass
class SafetyStatus:
    """Current safety status."""
    state: SafetyState
    emergency_stop: bool
    velocity_limit: float
    force_limit: float
    safety_zone_violation: bool
    communication_lost: bool
    last_heartbeat: float
    violations: list


class SafetyController:
    """Main safety controller for the robot."""
    
    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self.status = SafetyStatus(
            state=SafetyState.SAFE,
            emergency_stop=False,
            velocity_limit=limits.max_velocity,
            force_limit=limits.max_force,
            safety_zone_violation=False,
            communication_lost=False,
            last_heartbeat=time.time(),
            violations=[]
        )
        self.callbacks: Dict[str, Callable] = {}
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    def register_callback(self, event: str, callback: Callable):
        """Register a safety event callback."""
        self.callbacks[event] = callback
    
    async def start(self):
        """Start the safety controller."""
        self.running = True
        self._task = asyncio.create_task(self._safety_loop())
        logger.info("Safety controller started")
    
    async def stop(self):
        """Stop the safety controller."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Safety controller stopped")
    
    async def _safety_loop(self):
        """Main safety monitoring loop."""
        while self.running:
            try:
                await self._check_safety_conditions()
                await asyncio.sleep(0.01)  # 100Hz safety loop
            except Exception as e:
                logger.error("Safety loop error", error=str(e))
                await asyncio.sleep(0.1)
    
    async def _check_safety_conditions(self):
        """Check all safety conditions."""
        current_time = time.time()
        
        # Check communication timeout
        if current_time - self.status.last_heartbeat > 5.0:
            self.status.communication_lost = True
            self._trigger_emergency("Communication timeout")
            return
        
        # Check for emergency stop
        if self.status.emergency_stop:
            self.status.state = SafetyState.EMERGENCY
            self._trigger_emergency("Emergency stop activated")
            return
        
        # Check velocity limits
        if self._check_velocity_violation():
            self.status.state = SafetyState.DANGER
            self._trigger_emergency("Velocity limit exceeded")
            return
        
        # Check force limits
        if self._check_force_violation():
            self.status.state = SafetyState.DANGER
            self._trigger_emergency("Force limit exceeded")
            return
        
        # Check safety zone
        if self._check_safety_zone_violation():
            self.status.state = SafetyState.WARNING
            self._trigger_warning("Safety zone violation")
            return
        
        # If all checks pass, set to safe
        if self.status.state != SafetyState.SAFE:
            self.status.state = SafetyState.SAFE
            logger.info("Safety state returned to SAFE")
    
    def _check_velocity_violation(self) -> bool:
        """Check if velocity limits are exceeded."""
        # This would interface with actual motor controllers
        # For now, return False as a placeholder
        return False
    
    def _check_force_violation(self) -> bool:
        """Check if force limits are exceeded."""
        # This would interface with force sensors
        # For now, return False as a placeholder
        return False
    
    def _check_safety_zone_violation(self) -> bool:
        """Check if robot is in safety zone violation."""
        # This would interface with perception system
        # For now, return False as a placeholder
        return False
    
    def _trigger_emergency(self, reason: str):
        """Trigger emergency stop."""
        self.status.emergency_stop = True
        self.status.violations.append({
            "type": "emergency",
            "reason": reason,
            "timestamp": time.time()
        })
        
        logger.critical("Emergency stop triggered", reason=reason)
        
        if "emergency" in self.callbacks:
            try:
                self.callbacks["emergency"](reason)
            except Exception as e:
                logger.error("Emergency callback failed", error=str(e))
    
    def _trigger_warning(self, reason: str):
        """Trigger safety warning."""
        self.status.violations.append({
            "type": "warning",
            "reason": reason,
            "timestamp": time.time()
        })
        
        logger.warning("Safety warning triggered", reason=reason)
        
        if "warning" in self.callbacks:
            try:
                self.callbacks["warning"](reason)
            except Exception as e:
                logger.error("Warning callback failed", error=str(e))
    
    def update_heartbeat(self):
        """Update the last heartbeat time."""
        self.status.last_heartbeat = time.time()
        self.status.communication_lost = False
    
    def set_emergency_stop(self, state: bool):
        """Set emergency stop state."""
        self.status.emergency_stop = state
        if state:
            self._trigger_emergency("Manual emergency stop")
    
    def get_status(self) -> SafetyStatus:
        """Get current safety status."""
        return self.status
    
    def is_safe_to_move(self) -> bool:
        """Check if it's safe to move the robot."""
        return (
            not self.status.emergency_stop and
            not self.status.communication_lost and
            self.status.state in [SafetyState.SAFE, SafetyState.WARNING]
        )


class HardwareSafetyChain:
    """Hardware-level safety chain independent of software."""
    
    def __init__(self):
        self.emergency_stop_pressed = False
        self.hardware_limits_ok = True
        self.watchdog_active = True
        self._watchdog_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start hardware safety monitoring."""
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        logger.info("Hardware safety chain started")
    
    async def stop(self):
        """Stop hardware safety monitoring."""
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        logger.info("Hardware safety chain stopped")
    
    async def _watchdog_loop(self):
        """Hardware watchdog loop."""
        while True:
            try:
                # Check hardware emergency stop
                if self._check_hardware_estop():
                    self.emergency_stop_pressed = True
                    logger.critical("Hardware emergency stop detected")
                
                # Check hardware limits
                if not self._check_hardware_limits():
                    self.hardware_limits_ok = False
                    logger.error("Hardware limits violated")
                
                await asyncio.sleep(0.01)  # 100Hz hardware monitoring
            except Exception as e:
                logger.error("Hardware safety loop error", error=str(e))
                await asyncio.sleep(0.1)
    
    def _check_hardware_estop(self) -> bool:
        """Check hardware emergency stop button."""
        # This would interface with actual hardware
        # For now, return False as a placeholder
        return False
    
    def _check_hardware_limits(self) -> bool:
        """Check hardware limit switches."""
        # This would interface with actual hardware
        # For now, return True as a placeholder
        return True
    
    def is_hardware_safe(self) -> bool:
        """Check if hardware safety chain is OK."""
        return (
            not self.emergency_stop_pressed and
            self.hardware_limits_ok and
            self.watchdog_active
        )
