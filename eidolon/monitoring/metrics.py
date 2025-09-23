"""Monitoring and metrics collection system."""

import asyncio
import time
import psutil
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import threading

from ..common.config import get_config

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricData:
    """Metric data point."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metric_type: MetricType = MetricType.GAUGE


class PrometheusMetrics:
    """Prometheus metrics collector."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up Prometheus metrics."""
        # System metrics
        self.metrics['cpu_usage'] = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.metrics['memory_usage'] = Gauge('system_memory_usage_bytes', 'Memory usage in bytes')
        self.metrics['disk_usage'] = Gauge('system_disk_usage_bytes', 'Disk usage in bytes')
        self.metrics['network_bytes_sent'] = Counter('system_network_bytes_sent_total', 'Total network bytes sent')
        self.metrics['network_bytes_received'] = Counter('system_network_bytes_received_total', 'Total network bytes received')
        
        # Robot metrics
        self.metrics['robot_online'] = Gauge('robot_online_total', 'Number of online robots', ['tenant_id'])
        self.metrics['robot_battery_level'] = Gauge('robot_battery_level_percent', 'Robot battery level', ['robot_id', 'tenant_id'])
        self.metrics['robot_safety_violations'] = Counter('robot_safety_violations_total', 'Total safety violations', ['robot_id', 'tenant_id', 'violation_type'])
        self.metrics['robot_commands_sent'] = Counter('robot_commands_sent_total', 'Total commands sent to robots', ['robot_id', 'tenant_id', 'command_type'])
        
        # Teleoperation metrics
        self.metrics['teleop_sessions_active'] = Gauge('teleop_sessions_active_total', 'Number of active teleop sessions', ['tenant_id'])
        self.metrics['teleop_session_duration'] = Histogram('teleop_session_duration_seconds', 'Teleop session duration', ['tenant_id'])
        self.metrics['teleop_commands_sent'] = Counter('teleop_commands_sent_total', 'Total teleop commands sent', ['session_id', 'tenant_id'])
        
        # Task metrics
        self.metrics['tasks_pending'] = Gauge('tasks_pending_total', 'Number of pending tasks', ['tenant_id'])
        self.metrics['tasks_running'] = Gauge('tasks_running_total', 'Number of running tasks', ['tenant_id'])
        self.metrics['tasks_completed'] = Counter('tasks_completed_total', 'Total completed tasks', ['tenant_id', 'status'])
        self.metrics['task_duration'] = Histogram('task_duration_seconds', 'Task execution duration', ['tenant_id', 'task_type'])
        
        # API metrics
        self.metrics['api_requests_total'] = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status_code'])
        self.metrics['api_request_duration'] = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
        self.metrics['api_errors_total'] = Counter('api_errors_total', 'Total API errors', ['method', 'endpoint', 'error_type'])
        
        # Storage metrics
        self.metrics['storage_objects_total'] = Gauge('storage_objects_total', 'Total storage objects', ['tenant_id', 'type'])
        self.metrics['storage_size_bytes'] = Gauge('storage_size_bytes', 'Storage size in bytes', ['tenant_id', 'type'])
        self.metrics['storage_operations'] = Counter('storage_operations_total', 'Total storage operations', ['tenant_id', 'operation'])
        
        # Security metrics
        self.metrics['security_events_total'] = Counter('security_events_total', 'Total security events', ['event_type', 'severity'])
        self.metrics['authentication_attempts'] = Counter('authentication_attempts_total', 'Total authentication attempts', ['result'])
        self.metrics['failed_logins'] = Counter('failed_logins_total', 'Total failed login attempts', ['user_type'])
        
        logger.info("Prometheus metrics initialized")
    
    def update_system_metrics(self):
        """Update system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics['cpu_usage'].set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics['memory_usage'].set(memory.used)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics['disk_usage'].set(disk.used)
        
        # Network usage
        network = psutil.net_io_counters()
        self.metrics['network_bytes_sent']._value._value = network.bytes_sent
        self.metrics['network_bytes_received']._value._value = network.bytes_recv
    
    def record_robot_online(self, robot_id: str, tenant_id: str, online: bool):
        """Record robot online status."""
        self.metrics['robot_online'].labels(tenant_id=tenant_id).set(1 if online else 0)
    
    def record_robot_battery(self, robot_id: str, tenant_id: str, battery_level: float):
        """Record robot battery level."""
        self.metrics['robot_battery_level'].labels(robot_id=robot_id, tenant_id=tenant_id).set(battery_level)
    
    def record_safety_violation(self, robot_id: str, tenant_id: str, violation_type: str):
        """Record safety violation."""
        self.metrics['robot_safety_violations'].labels(
            robot_id=robot_id, tenant_id=tenant_id, violation_type=violation_type
        ).inc()
    
    def record_robot_command(self, robot_id: str, tenant_id: str, command_type: str):
        """Record robot command."""
        self.metrics['robot_commands_sent'].labels(
            robot_id=robot_id, tenant_id=tenant_id, command_type=command_type
        ).inc()
    
    def record_teleop_session(self, tenant_id: str, active: bool):
        """Record teleop session status."""
        self.metrics['teleop_sessions_active'].labels(tenant_id=tenant_id).set(1 if active else 0)
    
    def record_teleop_duration(self, tenant_id: str, duration: float):
        """Record teleop session duration."""
        self.metrics['teleop_session_duration'].labels(tenant_id=tenant_id).observe(duration)
    
    def record_teleop_command(self, session_id: str, tenant_id: str):
        """Record teleop command."""
        self.metrics['teleop_commands_sent'].labels(session_id=session_id, tenant_id=tenant_id).inc()
    
    def record_task_status(self, tenant_id: str, status: str, count: int):
        """Record task status."""
        if status == "pending":
            self.metrics['tasks_pending'].labels(tenant_id=tenant_id).set(count)
        elif status == "running":
            self.metrics['tasks_running'].labels(tenant_id=tenant_id).set(count)
        elif status in ["completed", "failed", "cancelled"]:
            self.metrics['tasks_completed'].labels(tenant_id=tenant_id, status=status).inc(count)
    
    def record_task_duration(self, tenant_id: str, task_type: str, duration: float):
        """Record task duration."""
        self.metrics['task_duration'].labels(tenant_id=tenant_id, task_type=task_type).observe(duration)
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request."""
        self.metrics['api_requests_total'].labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.metrics['api_request_duration'].labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_api_error(self, method: str, endpoint: str, error_type: str):
        """Record API error."""
        self.metrics['api_errors_total'].labels(method=method, endpoint=endpoint, error_type=error_type).inc()
    
    def record_storage_operation(self, tenant_id: str, operation: str):
        """Record storage operation."""
        self.metrics['storage_operations'].labels(tenant_id=tenant_id, operation=operation).inc()
    
    def record_security_event(self, event_type: str, severity: str):
        """Record security event."""
        self.metrics['security_events_total'].labels(event_type=event_type, severity=severity).inc()
    
    def record_auth_attempt(self, result: str):
        """Record authentication attempt."""
        self.metrics['authentication_attempts'].labels(result=result).inc()
    
    def record_failed_login(self, user_type: str):
        """Record failed login."""
        self.metrics['failed_logins'].labels(user_type=user_type).inc()


class MetricsCollector:
    """Main metrics collector."""
    
    def __init__(self):
        self.config = get_config()
        self.prometheus_metrics = PrometheusMetrics()
        self.custom_metrics: List[MetricData] = []
        self.running = False
        self._collector_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable] = []
    
    async def start(self):
        """Start the metrics collector."""
        self.running = True
        
        # Start Prometheus HTTP server
        start_http_server(self.config.monitoring.prometheus_port)
        
        # Start collection loop
        self._collector_task = asyncio.create_task(self._collection_loop())
        
        logger.info("Metrics collector started", port=self.config.monitoring.prometheus_port)
    
    async def stop(self):
        """Stop the metrics collector."""
        self.running = False
        if self._collector_task:
            self._collector_task.cancel()
            try:
                await self._collector_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collector stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                # Update system metrics
                self.prometheus_metrics.update_system_metrics()
                
                # Trigger callbacks
                for callback in self._callbacks:
                    try:
                        await callback()
                    except Exception as e:
                        logger.error("Metrics callback error", error=str(e))
                
                await asyncio.sleep(10.0)  # 10-second collection interval
                
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(1.0)
    
    def add_callback(self, callback: Callable):
        """Add metrics collection callback."""
        self._callbacks.append(callback)
    
    def record_metric(self, metric: MetricData):
        """Record a custom metric."""
        self.custom_metrics.append(metric)
        
        # Keep only recent metrics (last 1000)
        if len(self.custom_metrics) > 1000:
            self.custom_metrics = self.custom_metrics[-1000:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "custom_metrics_count": len(self.custom_metrics),
            "prometheus_metrics_count": len(self.prometheus_metrics.metrics),
            "collection_running": self.running
        }


class HealthChecker:
    """System health checker."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_status: Dict[str, Any] = {}
        self.running = False
        self._health_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the health checker."""
        self.running = True
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info("Health checker started")
    
    async def stop(self):
        """Stop the health checker."""
        self.running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")
    
    async def _health_loop(self):
        """Main health checking loop."""
        while self.running:
            try:
                await self._check_system_health()
                await asyncio.sleep(30.0)  # 30-second health check interval
            except Exception as e:
                logger.error("Health check error", error=str(e))
                await asyncio.sleep(5.0)
    
    async def _check_system_health(self):
        """Check system health."""
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        health_status["components"]["cpu"] = {
            "status": "healthy" if cpu_percent < 80 else "warning" if cpu_percent < 95 else "critical",
            "usage_percent": cpu_percent
        }
        
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        health_status["components"]["memory"] = {
            "status": "healthy" if memory_percent < 80 else "warning" if memory_percent < 95 else "critical",
            "usage_percent": memory_percent,
            "available_gb": memory.available / (1024**3)
        }
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        health_status["components"]["disk"] = {
            "status": "healthy" if disk_percent < 80 else "warning" if disk_percent < 95 else "critical",
            "usage_percent": disk_percent,
            "free_gb": disk.free / (1024**3)
        }
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in health_status["components"].values()]
        if "critical" in component_statuses:
            health_status["overall_status"] = "critical"
        elif "warning" in component_statuses:
            health_status["overall_status"] = "warning"
        
        self.health_status = health_status
        
        # Record health metrics
        if health_status["overall_status"] != "healthy":
            self.metrics_collector.prometheus_metrics.record_security_event(
                "health_check_failed", health_status["overall_status"]
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.health_status


class MonitoringSystem:
    """Main monitoring system."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker(self.metrics_collector)
        self.running = False
    
    async def start(self):
        """Start the monitoring system."""
        self.running = True
        await self.metrics_collector.start()
        await self.health_checker.start()
        logger.info("Monitoring system started")
    
    async def stop(self):
        """Stop the monitoring system."""
        self.running = False
        await self.health_checker.stop()
        await self.metrics_collector.stop()
        logger.info("Monitoring system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "monitoring_running": self.running,
            "metrics": self.metrics_collector.get_metrics_summary(),
            "health": self.health_checker.get_health_status()
        }
