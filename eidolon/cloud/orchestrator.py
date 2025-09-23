"""Orchestration service for task management and scheduling."""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority enumeration."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    EMERGENCY = 5


@dataclass
class Task:
    """Task definition."""
    id: str
    name: str
    description: str
    robot_id: str
    tenant_id: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None


@dataclass
class Robot:
    """Robot information."""
    id: str
    name: str
    tenant_id: str
    status: str = "offline"
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    last_seen: datetime = field(default_factory=datetime.utcnow)
    location: Dict[str, float] = field(default_factory=dict)
    battery_level: float = 100.0
    safety_status: str = "safe"


class TaskScheduler:
    """Task scheduler for managing robot tasks."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.robots: Dict[str, Robot] = {}
        self.task_queue: List[str] = []
        self.running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._callbacks: Dict[str, List[Callable]] = {}
    
    async def start(self):
        """Start the task scheduler."""
        self.running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the task scheduler."""
        self.running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Task scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                await self._process_task_queue()
                await self._check_running_tasks()
                await asyncio.sleep(1.0)  # 1-second scheduler loop
            except Exception as e:
                logger.error("Scheduler loop error", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _process_task_queue(self):
        """Process tasks in the queue."""
        if not self.task_queue:
            return
        
        # Sort tasks by priority and creation time
        sorted_tasks = sorted(
            self.task_queue,
            key=lambda task_id: (
                self.tasks[task_id].priority.value,
                self.tasks[task_id].created_at
            ),
            reverse=True
        )
        
        for task_id in sorted_tasks:
            task = self.tasks[task_id]
            
            # Check if task can be scheduled
            if await self._can_schedule_task(task):
                await self._schedule_task(task)
                self.task_queue.remove(task_id)
                break
    
    async def _can_schedule_task(self, task: Task) -> bool:
        """Check if a task can be scheduled."""
        # Check if robot is available
        if task.robot_id not in self.robots:
            return False
        
        robot = self.robots[task.robot_id]
        if robot.status != "online" or robot.current_task:
            return False
        
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def _schedule_task(self, task: Task):
        """Schedule a task for execution."""
        task.status = TaskStatus.SCHEDULED
        task.scheduled_at = datetime.utcnow()
        task.updated_at = datetime.utcnow()
        
        # Update robot status
        if task.robot_id in self.robots:
            self.robots[task.robot_id].current_task = task.id
        
        logger.info("Task scheduled", task_id=task.id, robot_id=task.robot_id)
        
        # Trigger callbacks
        await self._trigger_callbacks("task_scheduled", task)
    
    async def _check_running_tasks(self):
        """Check status of running tasks."""
        for task in self.tasks.values():
            if task.status == TaskStatus.RUNNING:
                # Check if task has timed out
                if task.started_at and datetime.utcnow() - task.started_at > timedelta(hours=1):
                    await self._fail_task(task, "Task timeout")
    
    async def _fail_task(self, task: Task, error_message: str):
        """Mark a task as failed."""
        task.status = TaskStatus.FAILED
        task.error_message = error_message
        task.updated_at = datetime.utcnow()
        
        # Free up robot
        if task.robot_id in self.robots:
            self.robots[task.robot_id].current_task = None
        
        logger.error("Task failed", task_id=task.id, error=error_message)
        await self._trigger_callbacks("task_failed", task)
    
    async def _trigger_callbacks(self, event: str, task: Task):
        """Trigger registered callbacks for an event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    await callback(task)
                except Exception as e:
                    logger.error("Callback error", event=event, error=str(e))
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for task events."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def create_task(self, task_data: Dict[str, Any]) -> Task:
        """Create a new task."""
        task_id = f"task-{int(time.time())}"
        task = Task(
            id=task_id,
            name=task_data.get("name", "Unnamed Task"),
            description=task_data.get("description", ""),
            robot_id=task_data["robot_id"],
            tenant_id=task_data["tenant_id"],
            priority=TaskPriority(task_data.get("priority", 2)),
            parameters=task_data.get("parameters", {}),
            dependencies=task_data.get("dependencies", [])
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        logger.info("Task created", task_id=task_id, robot_id=task.robot_id)
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        task.updated_at = datetime.utcnow()
        
        # Free up robot
        if task.robot_id in self.robots and self.robots[task.robot_id].current_task == task_id:
            self.robots[task.robot_id].current_task = None
        
        # Remove from queue if pending
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)
        
        logger.info("Task cancelled", task_id=task_id)
        return True
    
    def register_robot(self, robot: Robot):
        """Register a robot with the scheduler."""
        self.robots[robot.id] = robot
        logger.info("Robot registered", robot_id=robot.id, tenant_id=robot.tenant_id)
    
    def update_robot_status(self, robot_id: str, status: str, **kwargs):
        """Update robot status."""
        if robot_id in self.robots:
            self.robots[robot_id].status = status
            self.robots[robot_id].last_seen = datetime.utcnow()
            
            for key, value in kwargs.items():
                if hasattr(self.robots[robot_id], key):
                    setattr(self.robots[robot_id], key, value)
            
            logger.debug("Robot status updated", robot_id=robot_id, status=status)
    
    def get_robot_tasks(self, robot_id: str) -> List[Task]:
        """Get tasks for a specific robot."""
        return [task for task in self.tasks.values() if task.robot_id == robot_id]
    
    def get_tenant_tasks(self, tenant_id: str) -> List[Task]:
        """Get tasks for a specific tenant."""
        return [task for task in self.tasks.values() if task.tenant_id == tenant_id]
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get task statistics."""
        total_tasks = len(self.tasks)
        status_counts = {}
        
        for status in TaskStatus:
            status_counts[status.value] = len([
                task for task in self.tasks.values() if task.status == status
            ])
        
        return {
            "total_tasks": total_tasks,
            "status_counts": status_counts,
            "queue_length": len(self.task_queue),
            "active_robots": len([r for r in self.robots.values() if r.status == "online"])
        }


class Orchestrator:
    """Main orchestrator service."""
    
    def __init__(self):
        self.scheduler = TaskScheduler()
        self.running = False
    
    async def start(self):
        """Start the orchestrator."""
        self.running = True
        await self.scheduler.start()
        logger.info("Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator."""
        self.running = False
        await self.scheduler.stop()
        logger.info("Orchestrator stopped")
    
    def create_task(self, task_data: Dict[str, Any]) -> Task:
        """Create a new task."""
        return self.scheduler.create_task(task_data)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.scheduler.get_task(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        return self.scheduler.cancel_task(task_id)
    
    def register_robot(self, robot: Robot):
        """Register a robot."""
        self.scheduler.register_robot(robot)
    
    def update_robot_status(self, robot_id: str, status: str, **kwargs):
        """Update robot status."""
        self.scheduler.update_robot_status(robot_id, status, **kwargs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return self.scheduler.get_task_statistics()
