"""Security monitoring and audit system."""

import asyncio
import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
import ipaddress
from collections import defaultdict, deque

from ..common.config import get_config
from ..common.security import SecurityManager

logger = structlog.get_logger(__name__)


class SecurityEventType(Enum):
    """Security event type enumeration."""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_TOKEN = "invalid_token"
    DEVICE_COMPROMISE = "device_compromise"
    NETWORK_INTRUSION = "network_intrusion"
    DATA_BREACH = "data_breach"
    POLICY_VIOLATION = "policy_violation"
    EMERGENCY_OVERRIDE = "emergency_override"


class SecuritySeverity(Enum):
    """Security severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record."""
    id: str
    event_type: SecurityEventType
    severity: SecuritySeverity
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    tenant_id: Optional[str] = None
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class AuditLog:
    """Audit log entry."""
    id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    tenant_id: Optional[str] = None
    action: str = ""
    resource: str = ""
    result: str = ""  # success, failure, error
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class RateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        requests = self.requests[identifier]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check if under limit
        if len(requests) >= self.max_requests:
            return False
        
        # Add current request
        requests.append(now)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        requests = self.requests[identifier]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        return max(0, self.max_requests - len(requests))


class IntrusionDetectionSystem:
    """Intrusion detection system."""
    
    def __init__(self):
        self.suspicious_ips: Dict[str, int] = defaultdict(int)
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        self.blocked_ips: set = set()
        self.whitelist_ips: set = set()
        self.running = False
        self._ids_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the IDS."""
        self.running = True
        self._ids_task = asyncio.create_task(self._ids_loop())
        logger.info("Intrusion detection system started")
    
    async def stop(self):
        """Stop the IDS."""
        self.running = False
        if self._ids_task:
            self._ids_task.cancel()
            try:
                await self._ids_task
            except asyncio.CancelledError:
                pass
        logger.info("Intrusion detection system stopped")
    
    async def _ids_loop(self):
        """Main IDS monitoring loop."""
        while self.running:
            try:
                await self._analyze_suspicious_activity()
                await asyncio.sleep(60.0)  # 1-minute IDS loop
            except Exception as e:
                logger.error("IDS loop error", error=str(e))
                await asyncio.sleep(10.0)
    
    async def _analyze_suspicious_activity(self):
        """Analyze suspicious activity patterns."""
        # Check for IPs with high failure rates
        for ip, failures in self.failed_attempts.items():
            if failures > 10:  # Threshold for suspicious activity
                self.suspicious_ips[ip] += 1
                
                if self.suspicious_ips[ip] > 5:  # Block after multiple violations
                    self.blocked_ips.add(ip)
                    logger.warning("IP blocked due to suspicious activity", ip=ip)
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips
    
    def is_ip_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted."""
        return ip in self.whitelist_ips
    
    def record_failed_attempt(self, ip: str):
        """Record failed authentication attempt."""
        self.failed_attempts[ip] += 1
    
    def record_successful_attempt(self, ip: str):
        """Record successful authentication attempt."""
        # Reset failure count on successful attempt
        if ip in self.failed_attempts:
            del self.failed_attempts[ip]
    
    def add_to_whitelist(self, ip: str):
        """Add IP to whitelist."""
        self.whitelist_ips.add(ip)
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
        logger.info("IP added to whitelist", ip=ip)
    
    def remove_from_whitelist(self, ip: str):
        """Remove IP from whitelist."""
        self.whitelist_ips.discard(ip)
        logger.info("IP removed from whitelist", ip=ip)


class SecurityAuditor:
    """Security audit system."""
    
    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.audit_logs: List[AuditLog] = []
        self.running = False
        self._audit_task: Optional[asyncio.Task] = None
        self.max_events = 10000
        self.max_logs = 50000
    
    async def start(self):
        """Start the security auditor."""
        self.running = True
        self._audit_task = asyncio.create_task(self._audit_loop())
        logger.info("Security auditor started")
    
    async def stop(self):
        """Stop the security auditor."""
        self.running = False
        if self._audit_task:
            self._audit_task.cancel()
            try:
                await self._audit_task
            except asyncio.CancelledError:
                pass
        logger.info("Security auditor stopped")
    
    async def _audit_loop(self):
        """Main audit processing loop."""
        while self.running:
            try:
                await self._process_audit_events()
                await asyncio.sleep(30.0)  # 30-second audit loop
            except Exception as e:
                logger.error("Audit loop error", error=str(e))
                await asyncio.sleep(5.0)
    
    async def _process_audit_events(self):
        """Process audit events."""
        # Clean up old events
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        self.security_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        self.audit_logs = [l for l in self.audit_logs if l.timestamp > cutoff_time]
        
        # Keep only recent events
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
        if len(self.audit_logs) > self.max_logs:
            self.audit_logs = self.audit_logs[-self.max_logs:]
    
    def record_security_event(self, event_type: SecurityEventType, severity: SecuritySeverity,
                            source_ip: str = None, user_id: str = None, device_id: str = None,
                            tenant_id: str = None, description: str = "", details: Dict[str, Any] = None):
        """Record a security event."""
        event_id = f"sec_{int(time.time())}_{hashlib.md5(f'{event_type}_{source_ip}_{user_id}'.encode()).hexdigest()[:8]}"
        
        event = SecurityEvent(
            id=event_id,
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            device_id=device_id,
            tenant_id=tenant_id,
            description=description,
            details=details or {}
        )
        
        self.security_events.append(event)
        
        # Log critical events immediately
        if severity == SecuritySeverity.CRITICAL:
            logger.critical("Critical security event", event_id=event_id, event_type=event_type.value)
        elif severity == SecuritySeverity.HIGH:
            logger.warning("High severity security event", event_id=event_id, event_type=event_type.value)
        else:
            logger.info("Security event recorded", event_id=event_id, event_type=event_type.value)
    
    def record_audit_log(self, user_id: str = None, device_id: str = None, tenant_id: str = None,
                        action: str = "", resource: str = "", result: str = "success",
                        details: Dict[str, Any] = None, ip_address: str = None, user_agent: str = None):
        """Record an audit log entry."""
        log_id = f"audit_{int(time.time())}_{hashlib.md5(f'{user_id}_{action}_{resource}'.encode()).hexdigest()[:8]}"
        
        log_entry = AuditLog(
            id=log_id,
            user_id=user_id,
            device_id=device_id,
            tenant_id=tenant_id,
            action=action,
            resource=resource,
            result=result,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.audit_logs.append(log_entry)
        
        # Log failed actions
        if result == "failure":
            logger.warning("Audit log: failed action", action=action, resource=resource, user_id=user_id)
    
    def get_security_events(self, tenant_id: str = None, severity: SecuritySeverity = None,
                           start_time: datetime = None, end_time: datetime = None) -> List[SecurityEvent]:
        """Get security events with filters."""
        events = self.security_events
        
        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]
        if severity:
            events = [e for e in events if e.severity == severity]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events
    
    def get_audit_logs(self, user_id: str = None, device_id: str = None, tenant_id: str = None,
                      start_time: datetime = None, end_time: datetime = None) -> List[AuditLog]:
        """Get audit logs with filters."""
        logs = self.audit_logs
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        if device_id:
            logs = [l for l in logs if l.device_id == device_id]
        if tenant_id:
            logs = [l for l in logs if l.tenant_id == tenant_id]
        if start_time:
            logs = [l for l in logs if l.timestamp >= start_time]
        if end_time:
            logs = [l for l in logs if l.timestamp <= end_time]
        
        return logs
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary statistics."""
        total_events = len(self.security_events)
        total_logs = len(self.audit_logs)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for event in self.security_events:
            severity_counts[event.severity.value] += 1
        
        # Count by event type
        event_type_counts = defaultdict(int)
        for event in self.security_events:
            event_type_counts[event.event_type.value] += 1
        
        # Count unresolved events
        unresolved_events = len([e for e in self.security_events if not e.resolved])
        
        return {
            "total_security_events": total_events,
            "total_audit_logs": total_logs,
            "unresolved_events": unresolved_events,
            "severity_counts": dict(severity_counts),
            "event_type_counts": dict(event_type_counts)
        }


class SecurityMonitor:
    """Main security monitoring system."""
    
    def __init__(self):
        self.config = get_config()
        self.security_manager = SecurityManager(self.config)
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
        self.ids = IntrusionDetectionSystem()
        self.auditor = SecurityAuditor()
        self.running = False
    
    async def start(self):
        """Start the security monitor."""
        self.running = True
        await self.ids.start()
        await self.auditor.start()
        logger.info("Security monitor started")
    
    async def stop(self):
        """Stop the security monitor."""
        self.running = False
        await self.auditor.stop()
        await self.ids.stop()
        logger.info("Security monitor stopped")
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        return self.rate_limiter.is_allowed(identifier)
    
    def check_ip_security(self, ip: str) -> Tuple[bool, str]:
        """Check IP security status."""
        if self.ids.is_ip_whitelisted(ip):
            return True, "whitelisted"
        elif self.ids.is_ip_blocked(ip):
            return False, "blocked"
        else:
            return True, "allowed"
    
    def record_authentication_attempt(self, ip: str, user_id: str, success: bool, tenant_id: str = None):
        """Record authentication attempt."""
        if success:
            self.ids.record_successful_attempt(ip)
            self.auditor.record_audit_log(
                user_id=user_id, tenant_id=tenant_id, action="authentication",
                resource="system", result="success", ip_address=ip
            )
        else:
            self.ids.record_failed_attempt(ip)
            self.auditor.record_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE, SecuritySeverity.MEDIUM,
                source_ip=ip, user_id=user_id, tenant_id=tenant_id,
                description="Failed authentication attempt"
            )
            self.auditor.record_audit_log(
                user_id=user_id, tenant_id=tenant_id, action="authentication",
                resource="system", result="failure", ip_address=ip
            )
    
    def record_api_access(self, user_id: str, resource: str, method: str, ip: str, success: bool, tenant_id: str = None):
        """Record API access."""
        self.auditor.record_audit_log(
            user_id=user_id, tenant_id=tenant_id, action=f"{method} {resource}",
            resource=resource, result="success" if success else "failure",
            ip_address=ip
        )
    
    def record_teleop_session(self, session_id: str, operator_id: str, robot_id: str, tenant_id: str, ip: str):
        """Record teleop session."""
        self.auditor.record_audit_log(
            user_id=operator_id, tenant_id=tenant_id, action="teleop_session_start",
            resource=f"robot:{robot_id}", result="success", ip_address=ip,
            details={"session_id": session_id, "robot_id": robot_id}
        )
    
    def record_emergency_stop(self, robot_id: str, operator_id: str, tenant_id: str, ip: str):
        """Record emergency stop."""
        self.auditor.record_security_event(
            SecurityEventType.EMERGENCY_OVERRIDE, SecuritySeverity.HIGH,
            user_id=operator_id, tenant_id=tenant_id, device_id=robot_id,
            description="Emergency stop activated", ip_address=ip
        )
        self.auditor.record_audit_log(
            user_id=operator_id, tenant_id=tenant_id, action="emergency_stop",
            resource=f"robot:{robot_id}", result="success", ip_address=ip
        )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security status."""
        return {
            "monitoring_active": self.running,
            "rate_limiter_status": "active",
            "ids_status": "active" if self.ids.running else "inactive",
            "auditor_status": "active" if self.auditor.running else "inactive",
            "summary": self.auditor.get_security_summary()
        }
