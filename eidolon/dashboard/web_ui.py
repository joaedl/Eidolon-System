"""Customer dashboard web UI for fleet management."""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import structlog

from ..common.config import get_config
from ..common.security import SecurityManager

logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Eidolon Customer Dashboard",
    description="Fleet management dashboard",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security_manager = SecurityManager(get_config())


class DashboardManager:
    """Dashboard manager for fleet data."""
    
    def __init__(self):
        self.robots: Dict[str, Dict[str, Any]] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.teleop_sessions: Dict[str, Dict[str, Any]] = {}
        self.analytics: Dict[str, Any] = {}
        self.running = False
        self._update_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the dashboard manager."""
        self.running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Dashboard manager started")
    
    async def stop(self):
        """Stop the dashboard manager."""
        self.running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("Dashboard manager stopped")
    
    async def _update_loop(self):
        """Update dashboard data."""
        while self.running:
            try:
                await self._update_robot_status()
                await self._update_task_status()
                await self._update_analytics()
                await asyncio.sleep(5.0)  # 5-second update interval
            except Exception as e:
                logger.error("Dashboard update error", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _update_robot_status(self):
        """Update robot status."""
        # Simulate robot status updates
        for robot_id in self.robots:
            self.robots[robot_id]["last_seen"] = datetime.utcnow().isoformat()
            self.robots[robot_id]["battery_level"] = max(0, self.robots[robot_id]["battery_level"] - 0.1)
    
    async def _update_task_status(self):
        """Update task status."""
        # Simulate task progress
        for task_id in self.tasks:
            if self.tasks[task_id]["status"] == "running":
                self.tasks[task_id]["progress"] = min(100, self.tasks[task_id]["progress"] + 1)
                if self.tasks[task_id]["progress"] >= 100:
                    self.tasks[task_id]["status"] = "completed"
                    self.tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()
    
    async def _update_analytics(self):
        """Update analytics data."""
        self.analytics = {
            "total_robots": len(self.robots),
            "online_robots": len([r for r in self.robots.values() if r["status"] == "online"]),
            "active_tasks": len([t for t in self.tasks.values() if t["status"] == "running"]),
            "completed_tasks_today": len([t for t in self.tasks.values() if t["status"] == "completed"]),
            "uptime_percentage": 99.5,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def add_robot(self, robot_data: Dict[str, Any]):
        """Add a robot to the dashboard."""
        robot_id = robot_data["id"]
        self.robots[robot_id] = {
            "id": robot_id,
            "name": robot_data.get("name", f"Robot {robot_id}"),
            "status": robot_data.get("status", "offline"),
            "battery_level": robot_data.get("battery_level", 100.0),
            "location": robot_data.get("location", {"x": 0, "y": 0, "z": 0}),
            "capabilities": robot_data.get("capabilities", []),
            "last_seen": datetime.utcnow().isoformat(),
            "tenant_id": robot_data.get("tenant_id", "")
        }
        logger.info("Robot added to dashboard", robot_id=robot_id)
    
    def add_task(self, task_data: Dict[str, Any]):
        """Add a task to the dashboard."""
        task_id = task_data["id"]
        self.tasks[task_id] = {
            "id": task_id,
            "name": task_data.get("name", "Unnamed Task"),
            "robot_id": task_data.get("robot_id", ""),
            "status": task_data.get("status", "pending"),
            "progress": task_data.get("progress", 0.0),
            "priority": task_data.get("priority", 1),
            "created_at": task_data.get("created_at", datetime.utcnow().isoformat()),
            "tenant_id": task_data.get("tenant_id", "")
        }
        logger.info("Task added to dashboard", task_id=task_id)
    
    def get_fleet_summary(self) -> Dict[str, Any]:
        """Get fleet summary."""
        return self.analytics
    
    def get_robots(self) -> List[Dict[str, Any]]:
        """Get all robots."""
        return list(self.robots.values())
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks."""
        return list(self.tasks.values())


# Global dashboard manager
dashboard_manager = DashboardManager()


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    await dashboard_manager.start()
    
    # Add some sample data
    dashboard_manager.add_robot({
        "id": "robot-001",
        "name": "Warehouse Robot 1",
        "status": "online",
        "battery_level": 85.0,
        "location": {"x": 10.5, "y": 5.2, "z": 0.0},
        "capabilities": ["navigation", "manipulation", "perception"],
        "tenant_id": "tenant-001"
    })
    
    dashboard_manager.add_robot({
        "id": "robot-002",
        "name": "Warehouse Robot 2",
        "status": "offline",
        "battery_level": 45.0,
        "location": {"x": 15.0, "y": 8.0, "z": 0.0},
        "capabilities": ["navigation", "manipulation"],
        "tenant_id": "tenant-001"
    })
    
    dashboard_manager.add_task({
        "id": "task-001",
        "name": "Pick and Place Operation",
        "robot_id": "robot-001",
        "status": "running",
        "progress": 65.0,
        "priority": 2,
        "tenant_id": "tenant-001"
    })


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    await dashboard_manager.stop()


@app.get("/")
async def get_dashboard():
    """Get dashboard HTML."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Eidolon Fleet Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .stat-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .stat-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
            .stat-label { color: #7f8c8d; margin-top: 5px; }
            .robots-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .robot-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .robot-status { display: inline-block; padding: 5px 10px; border-radius: 15px; color: white; font-size: 0.8em; }
            .status-online { background: #27ae60; }
            .status-offline { background: #e74c3c; }
            .battery-bar { width: 100%; height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; margin: 10px 0; }
            .battery-fill { height: 100%; background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%); }
            .tasks-section { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .task-item { padding: 10px; border-bottom: 1px solid #ecf0f1; }
            .task-progress { width: 100%; height: 10px; background: #ecf0f1; border-radius: 5px; overflow: hidden; margin: 5px 0; }
            .progress-fill { height: 100%; background: #3498db; }
            .controls { margin: 20px 0; }
            .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
            .btn-primary { background: #3498db; color: white; }
            .btn-success { background: #27ae60; color: white; }
            .btn-warning { background: #f39c12; color: white; }
            .btn-danger { background: #e74c3c; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Eidolon Fleet Dashboard</h1>
                <p>Real-time robot fleet management and monitoring</p>
            </div>
            
            <div class="stats-grid" id="stats-grid">
                <!-- Stats will be populated by JavaScript -->
            </div>
            
            <div class="robots-grid" id="robots-grid">
                <!-- Robots will be populated by JavaScript -->
            </div>
            
            <div class="tasks-section">
                <h3>Active Tasks</h3>
                <div id="tasks-list">
                    <!-- Tasks will be populated by JavaScript -->
                </div>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="refreshData()">Refresh</button>
                <button class="btn btn-success" onclick="createTask()">Create Task</button>
                <button class="btn btn-warning" onclick="requestTeleop()">Request Teleop</button>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket('ws://localhost:8003/ws');
            
            ws.onopen = function() {
                console.log('Connected to dashboard');
                refreshData();
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'fleet_update') {
                    updateDashboard(data.data);
                }
            };
            
            function refreshData() {
                fetch('/api/v1/fleet/summary')
                    .then(response => response.json())
                    .then(data => updateStats(data));
                
                fetch('/api/v1/robots')
                    .then(response => response.json())
                    .then(data => updateRobots(data.robots));
                
                fetch('/api/v1/tasks')
                    .then(response => response.json())
                    .then(data => updateTasks(data.tasks));
            }
            
            function updateStats(data) {
                const statsGrid = document.getElementById('stats-grid');
                statsGrid.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-value">${data.total_robots}</div>
                        <div class="stat-label">Total Robots</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.online_robots}</div>
                        <div class="stat-label">Online Robots</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.active_tasks}</div>
                        <div class="stat-label">Active Tasks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${data.uptime_percentage}%</div>
                        <div class="stat-label">Uptime</div>
                    </div>
                `;
            }
            
            function updateRobots(robots) {
                const robotsGrid = document.getElementById('robots-grid');
                robotsGrid.innerHTML = robots.map(robot => `
                    <div class="robot-card">
                        <h4>${robot.name}</h4>
                        <span class="robot-status status-${robot.status}">${robot.status.toUpperCase()}</span>
                        <div class="battery-bar">
                            <div class="battery-fill" style="width: ${robot.battery_level}%"></div>
                        </div>
                        <p>Battery: ${robot.battery_level.toFixed(1)}%</p>
                        <p>Location: (${robot.location.x.toFixed(1)}, ${robot.location.y.toFixed(1)})</p>
                        <button class="btn btn-primary" onclick="controlRobot('${robot.id}')">Control</button>
                    </div>
                `).join('');
            }
            
            function updateTasks(tasks) {
                const tasksList = document.getElementById('tasks-list');
                tasksList.innerHTML = tasks.map(task => `
                    <div class="task-item">
                        <h5>${task.name}</h5>
                        <p>Robot: ${task.robot_id} | Status: ${task.status}</p>
                        <div class="task-progress">
                            <div class="progress-fill" style="width: ${task.progress}%"></div>
                        </div>
                        <p>Progress: ${task.progress.toFixed(1)}%</p>
                    </div>
                `).join('');
            }
            
            function controlRobot(robotId) {
                alert(`Controlling robot ${robotId}`);
            }
            
            function createTask() {
                alert('Create task functionality');
            }
            
            function requestTeleop() {
                alert('Request teleop functionality');
            }
            
            // Auto-refresh every 30 seconds
            setInterval(refreshData, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    
    try:
        while True:
            # Send fleet update
            fleet_data = dashboard_manager.get_fleet_summary()
            await websocket.send_text(json.dumps({
                "type": "fleet_update",
                "data": fleet_data
            }))
            
            await asyncio.sleep(5.0)  # Send update every 5 seconds
            
    except WebSocketDisconnect:
        logger.info("Dashboard client disconnected")
    except Exception as e:
        logger.error("Dashboard websocket error", error=str(e))


@app.get("/api/v1/fleet/summary")
async def get_fleet_summary():
    """Get fleet summary."""
    return dashboard_manager.get_fleet_summary()


@app.get("/api/v1/robots")
async def get_robots():
    """Get all robots."""
    robots = dashboard_manager.get_robots()
    return {"robots": robots, "total": len(robots)}


@app.get("/api/v1/robots/{robot_id}")
async def get_robot(robot_id: str):
    """Get robot details."""
    robots = dashboard_manager.get_robots()
    robot = next((r for r in robots if r["id"] == robot_id), None)
    
    if not robot:
        raise HTTPException(status_code=404, detail="Robot not found")
    
    return robot


@app.post("/api/v1/robots/{robot_id}/actions")
async def robot_action(robot_id: str, action: dict):
    """Execute robot action."""
    action_type = action.get("type")
    
    if action_type == "reboot":
        return {"status": "success", "message": "Robot reboot initiated"}
    elif action_type == "safe_park":
        return {"status": "success", "message": "Robot moving to safe position"}
    elif action_type == "emergency_stop":
        return {"status": "success", "message": "Emergency stop activated"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action type: {action_type}")


@app.get("/api/v1/tasks")
async def get_tasks():
    """Get all tasks."""
    tasks = dashboard_manager.get_tasks()
    return {"tasks": tasks, "total": len(tasks)}


@app.post("/api/v1/tasks")
async def create_task(task_data: dict):
    """Create a new task."""
    task_id = f"task-{int(time.time())}"
    task = {
        "id": task_id,
        "name": task_data.get("name", "Unnamed Task"),
        "robot_id": task_data.get("robot_id", ""),
        "status": "pending",
        "progress": 0.0,
        "priority": task_data.get("priority", 1),
        "created_at": datetime.utcnow().isoformat(),
        "tenant_id": task_data.get("tenant_id", "")
    }
    
    dashboard_manager.add_task(task)
    return task


@app.post("/api/v1/teleop/requests")
async def request_teleop(request_data: dict):
    """Request teleoperation session."""
    robot_id = request_data.get("robot_id")
    operator_id = request_data.get("operator_id")
    
    if not robot_id or not operator_id:
        raise HTTPException(status_code=400, detail="Missing robot_id or operator_id")
    
    # Create teleop request
    request_id = f"teleop-{int(time.time())}"
    
    return {
        "request_id": request_id,
        "robot_id": robot_id,
        "operator_id": operator_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "message": "Teleop request submitted to control room"
    }


@app.get("/api/v1/billing/usage")
async def get_billing_usage(from_date: str = None, to_date: str = None):
    """Get billing usage information."""
    return {
        "period": {
            "from": from_date or (datetime.utcnow() - timedelta(days=30)).isoformat(),
            "to": to_date or datetime.utcnow().isoformat()
        },
        "usage": {
            "compute_hours": 120.5,
            "storage_gb": 45.2,
            "bandwidth_gb": 12.8,
            "teleop_minutes": 180.0
        },
        "cost": {
            "compute": 24.10,
            "storage": 4.52,
            "bandwidth": 1.28,
            "teleop": 18.00,
            "total": 47.90
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
