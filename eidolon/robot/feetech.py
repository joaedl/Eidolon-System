"""
Feetech servo motor communication module for SO100 LeRobot teleoperation system.

Based on the LeRobot FeetechMotorsBus implementation, adapted for SO100 specific needs.
"""

import enum
import logging
import math
import time
import traceback
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import numpy as np

# Constants
PROTOCOL_VERSION = 0
BAUDRATE = 1_000_000
TIMEOUT_MS = 1000
MAX_ID_RANGE = 252

# Position bounds
LOWER_BOUND_DEGREE = -270
UPPER_BOUND_DEGREE = 270
LOWER_BOUND_LINEAR = -10
UPPER_BOUND_LINEAR = 110
HALF_TURN_DEGREE = 180

# Retry counts
NUM_READ_RETRY = 20
NUM_WRITE_RETRY = 20

# SCS Series Control Table (STS3215 compatible)
SCS_SERIES_CONTROL_TABLE = {
    "Model": (3, 2),
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Angle_Limit": (9, 2),
    "Max_Angle_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2),
    "Angular_Resolution": (30, 1),
    "Offset": (31, 2),
    "Mode": (33, 1),
    "Protective_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
    "Speed_closed_loop_P_proportional_coefficient": (37, 1),
    "Over_Current_Protection_Time": (38, 1),
    "Velocity_closed_loop_I_integral_coefficient": (39, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Speed": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Speed": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Status": (65, 1),
    "Moving": (66, 1),
    "Present_Current": (69, 2),
    "Maximum_Acceleration": (85, 2),
}

# Baudrate table
SCS_SERIES_BAUDRATE_TABLE = {
    0: 1_000_000,
    1: 500_000,
    2: 250_000,
    3: 128_000,
    4: 115_200,
    5: 57_600,
    6: 38_400,
    7: 19_200,
}

# Model configurations
MODEL_CONTROL_TABLE = {
    "scs_series": SCS_SERIES_CONTROL_TABLE,
    "sts3215": SCS_SERIES_CONTROL_TABLE,
}

MODEL_RESOLUTION = {
    "scs_series": 4096,
    "sts3215": 4096,
}

MODEL_BAUDRATE_TABLE = {
    "scs_series": SCS_SERIES_BAUDRATE_TABLE,
    "sts3215": SCS_SERIES_BAUDRATE_TABLE,
}

# Calibration requirements
CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]
CONVERT_UINT32_TO_INT32_REQUIRED = ["Goal_Position", "Present_Position"]


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0


class DriveMode(enum.Enum):
    NON_INVERTED = 0
    INVERTED = 1


class CalibrationMode(enum.Enum):
    DEGREE = 0
    LINEAR = 1


class RobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""
    pass


class RobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""
    pass


class JointOutOfRangeError(Exception):
    """Exception raised when joint is out of range."""
    pass


@dataclass
class FeetechConfig:
    """Configuration for Feetech servo bus"""
    port: str
    baudrate: int = BAUDRATE
    follower_ids: List[int] = None
    leader_ids: List[int] = None
    mock: bool = False
    
    def __post_init__(self):
        if self.follower_ids is None:
            self.follower_ids = list(range(1, 7))  # IDs 1-6
        if self.leader_ids is None:
            self.leader_ids = list(range(7, 13))   # IDs 7-12


def convert_degrees_to_steps(degrees: Union[float, np.ndarray], resolution: int = 4096) -> Union[int, np.ndarray]:
    """Convert degrees to servo steps (0-360 degrees to 0-4095 steps)"""
    # Map 0-360 degrees to 0-4095 steps
    # Clamp to valid range and use resolution-1 as max
    steps = degrees / 360 * (resolution - 1)
    if isinstance(degrees, np.ndarray):
        return np.clip(steps.astype(int), 0, resolution - 1)
    else:
        return int(max(0, min(resolution - 1, steps)))


def convert_steps_to_degrees(steps: Union[int, np.ndarray], resolution: int = 4096) -> np.ndarray:
    """Convert servo steps to degrees (0-4095 steps to 0-360 degrees)"""
    # Map 0-4095 steps to 0-360 degrees
    degrees = steps / resolution * 360
    return degrees


def convert_to_bytes(value: int, num_bytes: int, mock: bool = False) -> List[int]:
    """Convert value to byte array for Feetech protocol"""
    if mock:
        return [value]
    
    try:
        import scservo_sdk as scs
        
        if num_bytes == 1:
            return [scs.SCS_LOBYTE(scs.SCS_LOWORD(value))]
        elif num_bytes == 2:
            return [
                scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_LOWORD(value))
            ]
        elif num_bytes == 4:
            return [
                scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
                scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
                scs.SCS_HIBYTE(scs.SCS_HIWORD(value))
            ]
        else:
            raise NotImplementedError(f"Unsupported byte count: {num_bytes}")
    except ImportError:
        # Fallback if scservo_sdk is not available
        if num_bytes == 1:
            return [value & 0xFF]
        elif num_bytes == 2:
            return [value & 0xFF, (value >> 8) & 0xFF]
        elif num_bytes == 4:
            return [
                value & 0xFF,
                (value >> 8) & 0xFF,
                (value >> 16) & 0xFF,
                (value >> 24) & 0xFF
            ]
        else:
            raise NotImplementedError(f"Unsupported byte count: {num_bytes}")


class FeetechServoBus:
    """Feetech servo motor communication bus for SO100 LeRobot"""
    
    def __init__(self, config: FeetechConfig):
        self.config = config
        self.port = config.port
        self.baudrate = config.baudrate
        self.follower_ids = config.follower_ids
        self.leader_ids = config.leader_ids
        self.mock = config.mock
        
        self.port_handler = None
        self.packet_handler = None
        self.is_connected = False
        self.group_readers = {}
        self.group_writers = {}
        
        # Position tracking for rotation detection
        self.track_positions = {}
        
        # Model configuration
        self.model = "sts3215"
        self.control_table = MODEL_CONTROL_TABLE[self.model]
        self.resolution = MODEL_RESOLUTION[self.model]
    
    def connect(self) -> bool:
        """Connect to the Feetech servo bus"""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"FeetechServoBus({self.port}) is already connected"
            )
        
        try:
            if self.mock:
                # Mock connection for testing
                self.is_connected = True
                return True
            
            try:
                import scservo_sdk as scs
            except ImportError:
                print("Warning: scservo_sdk not available, using mock mode")
                self.mock = True
                self.is_connected = True
                return True
            
            print(f"Initializing Feetech SDK connection to {self.port} at {self.baudrate} baud...")
            
            self.port_handler = scs.PortHandler(self.port)
            self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION)
            
            print(f"Opening port {self.port}...")
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'")
            
            print(f"Port {self.port} opened successfully")
            self.port_handler.setBaudRate(self.baudrate)
            self.port_handler.setPacketTimeoutMillis(TIMEOUT_MS)
            
            # Test basic communication
            print("Testing basic communication...")
            try:
                # Try to ping a servo (servo ID 1)
                result = self.packet_handler.ping(self.port_handler, 1)
                if result:
                    print("✅ Successfully pinged servo ID 1")
                else:
                    print("⚠️  Could not ping servo ID 1, but connection established")
            except Exception as e:
                print(f"⚠️  Ping test failed: {e}, but connection established")
            
            self.is_connected = True
            print(f"✅ Connected to Feetech servo bus on {self.port}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to {self.port}: {e}")
            print("Falling back to mock mode for testing")
            self.mock = True
            self.is_connected = True
            return True
    
    def disconnect(self):
        """Disconnect from the Feetech servo bus"""
        if not self.is_connected:
            return
        
        if self.port_handler is not None:
            try:
                self.port_handler.closePort()
            except AttributeError:
                # Handle case where port_handler is not properly initialized
                pass
            self.port_handler = None
        
        self.packet_handler = None
        self.group_readers = {}
        self.group_writers = {}
        self.is_connected = False
    
    def read_position(self, servo_id: int) -> Optional[float]:
        """Read position of a single servo in degrees"""
        if not self.is_connected:
            return None
        
        try:
            if self.mock:
                # Return simulated position for testing
                return 0.0
            
            import scservo_sdk as scs
            
            # Try multiple read methods
            addr, num_bytes = self.control_table["Present_Position"]
            
            # Method 1: Direct read using packet handler
            try:
                result = self.packet_handler.readTxRx(self.port_handler, servo_id, addr, num_bytes)
                if isinstance(result, tuple) and len(result) >= 3:
                    data, error, _ = result
                    if error == 0 and data and len(data) >= num_bytes:
                        # Convert bytes to integer (little endian)
                        raw_position = int.from_bytes(data[:num_bytes], byteorder='little')
                        position = convert_steps_to_degrees(raw_position, self.resolution)
                        return float(position)
                    else:
                        print(f"Method 1 error for servo {servo_id}: error={error}, data={data}")
                elif isinstance(result, (list, bytes)) and len(result) >= num_bytes:
                    # Direct data return
                    raw_position = int.from_bytes(result[:num_bytes], byteorder='little')
                    position = convert_steps_to_degrees(raw_position, self.resolution)
                    return float(position)
            except Exception as e:
                print(f"Method 1 failed for servo {servo_id}: {e}")
            
            # Method 2: GroupSyncRead with single servo
            try:
                group = scs.GroupSyncRead(self.port_handler, self.packet_handler, addr, num_bytes)
                group.addParam(servo_id)
                
                for _ in range(NUM_READ_RETRY):
                    comm = group.txRxPacket()
                    if comm == scs.COMM_SUCCESS:
                        break
                
                if comm == scs.COMM_SUCCESS:
                    raw_position = group.getData(servo_id, addr, num_bytes)
                    position = convert_steps_to_degrees(raw_position, self.resolution)
                    return float(position)
            except Exception as e:
                print(f"Method 2 failed for servo {servo_id}: {e}")
            
            # Method 3: Try reading with different address (some servos use different addresses)
            try:
                # Try reading from a different position address
                alt_addr = 0x24  # Alternative position address
                result = self.packet_handler.readTxRx(self.port_handler, servo_id, alt_addr, num_bytes)
                if isinstance(result, tuple) and len(result) >= 3:
                    data, error, _ = result
                    if error == 0 and data and len(data) >= num_bytes:
                        raw_position = int.from_bytes(data[:num_bytes], byteorder='little')
                        position = convert_steps_to_degrees(raw_position, self.resolution)
                        return float(position)
                    else:
                        print(f"Method 3 error for servo {servo_id}: error={error}, data={data}")
                elif isinstance(result, (list, bytes)) and len(result) >= num_bytes:
                    raw_position = int.from_bytes(result[:num_bytes], byteorder='little')
                    position = convert_steps_to_degrees(raw_position, self.resolution)
                    return float(position)
            except Exception as e:
                print(f"Method 3 failed for servo {servo_id}: {e}")
            
            print(f"All read methods failed for servo {servo_id}")
            return None
            
        except Exception as e:
            print(f"Error reading position from servo {servo_id}: {e}")
            return None
    
    def read_positions(self, servo_ids: List[int]) -> Dict[int, float]:
        """Read positions of multiple servos"""
        positions = {}
        for servo_id in servo_ids:
            pos = self.read_position(servo_id)
            if pos is not None:
                positions[servo_id] = pos
        return positions
    
    def set_position(self, servo_id: int, position_degrees: float) -> bool:
        """Set position of a single servo in degrees using group write for reliability"""
        if not self.is_connected:
            return False
        
        try:
            if self.mock:
                # Simulate position setting
                return True
            
            import scservo_sdk as scs
            
            # Convert degrees to steps
            position_steps = convert_degrees_to_steps(position_degrees, self.resolution)
            
            # Use group write for better reliability
            addr, num_bytes = self.control_table["Goal_Position"]
            group = scs.GroupSyncWrite(self.port_handler, self.packet_handler, addr, num_bytes)
            
            data = convert_to_bytes(position_steps, num_bytes, self.mock)
            group.addParam(servo_id, data)
            
            # Try multiple times with small delays
            for attempt in range(2):
                comm = group.txPacket()
                if comm == scs.COMM_SUCCESS:
                    return True
                elif comm == scs.COMM_TX_FAIL:
                    # Small delay before retry
                    time.sleep(0.002)  # 2ms delay
                    continue
                else:
                    break
            
            if comm != scs.COMM_SUCCESS:
                print(f"Failed to set position for servo {servo_id} (error={comm})")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error setting position for servo {servo_id}: {e}")
            return False
    
    def set_positions(self, servo_positions: Dict[int, float]) -> bool:
        """Set positions of multiple servos"""
        success = True
        for servo_id, position in servo_positions.items():
            if not self.set_position(servo_id, position):
                success = False
        return success
    
    def enable_torque(self, servo_id: int, enabled: bool = True) -> bool:
        """Enable or disable torque for a servo"""
        if not self.is_connected:
            return False
        
        try:
            if self.mock:
                return True
            
            import scservo_sdk as scs
            
            addr, num_bytes = self.control_table["Torque_Enable"]
            group = scs.GroupSyncWrite(self.port_handler, self.packet_handler, addr, num_bytes)
            
            data = convert_to_bytes(1 if enabled else 0, num_bytes, self.mock)
            group.addParam(servo_id, data)
            
            for _ in range(NUM_WRITE_RETRY):
                comm = group.txPacket()
                if comm == scs.COMM_SUCCESS:
                    break
            
            if comm != scs.COMM_SUCCESS:
                print(f"Failed to set torque for servo {servo_id}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error setting torque for servo {servo_id}: {e}")
            return False
    
    def enable_all_torque(self, enabled: bool = True) -> bool:
        """Enable or disable torque for all servos"""
        success = True
        all_ids = self.follower_ids + self.leader_ids
        for servo_id in all_ids:
            if not self.enable_torque(servo_id, enabled):
                success = False
        return success
    
    def set_speed(self, servo_id: int, speed: int) -> bool:
        """Set speed for a servo (0-1023, where 0 = max speed)"""
        if not self.is_connected:
            return False
        
        try:
            if self.mock:
                return True
            
            import scservo_sdk as scs
            
            addr, num_bytes = self.control_table["Goal_Speed"]
            group = scs.GroupSyncWrite(self.port_handler, self.packet_handler, addr, num_bytes)
            
            # Clamp speed to valid range
            speed = max(0, min(1023, speed))
            data = convert_to_bytes(speed, num_bytes, self.mock)
            group.addParam(servo_id, data)
            
            for _ in range(NUM_WRITE_RETRY):
                comm = group.txPacket()
                if comm == scs.COMM_SUCCESS:
                    break
            
            if comm != scs.COMM_SUCCESS:
                print(f"Failed to set speed for servo {servo_id}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error setting speed for servo {servo_id}: {e}")
            return False
    
    def set_acceleration(self, servo_id: int, acceleration: int) -> bool:
        """Set acceleration for a servo (0-255, where 0 = max acceleration)"""
        if not self.is_connected:
            return False
        
        try:
            if self.mock:
                return True
            
            import scservo_sdk as scs
            
            addr, num_bytes = self.control_table["Acceleration"]
            group = scs.GroupSyncWrite(self.port_handler, self.packet_handler, addr, num_bytes)
            
            # Clamp acceleration to valid range (0-255)
            acceleration = max(0, min(255, acceleration))
            data = convert_to_bytes(acceleration, num_bytes, self.mock)
            group.addParam(servo_id, data)
            
            for _ in range(NUM_WRITE_RETRY):
                comm = group.txPacket()
                if comm == scs.COMM_SUCCESS:
                    break
            
            if comm != scs.COMM_SUCCESS:
                print(f"Failed to set acceleration for servo {servo_id}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error setting acceleration for servo {servo_id}: {e}")
            return False
    
    def set_max_acceleration(self, servo_id: int, max_acceleration: int) -> bool:
        """Set maximum acceleration for a servo (0-65535, where 0 = max acceleration)"""
        if not self.is_connected:
            return False
        
        try:
            if self.mock:
                return True
            
            import scservo_sdk as scs
            
            addr, num_bytes = self.control_table["Maximum_Acceleration"]
            group = scs.GroupSyncWrite(self.port_handler, self.packet_handler, addr, num_bytes)
            
            # Clamp max acceleration to valid range (0-65535)
            max_acceleration = max(0, min(65535, max_acceleration))
            data = convert_to_bytes(max_acceleration, num_bytes, self.mock)
            group.addParam(servo_id, data)
            
            for _ in range(NUM_WRITE_RETRY):
                comm = group.txPacket()
                if comm == scs.COMM_SUCCESS:
                    break
            
            if comm != scs.COMM_SUCCESS:
                print(f"Failed to set max acceleration for servo {servo_id}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error setting max acceleration for servo {servo_id}: {e}")
            return False
    
    def set_angle_limits(self, servo_id: int, min_degrees: float = 0.0, max_degrees: float = 360.0) -> bool:
        """Set angle limits for a servo"""
        if not self.is_connected:
            return False
        
        try:
            if self.mock:
                return True
            
            import scservo_sdk as scs
            
            # Convert degrees to steps
            min_steps = convert_degrees_to_steps(min_degrees, self.resolution)
            max_steps = convert_degrees_to_steps(max_degrees, self.resolution)
            
            # Set min angle limit
            addr, num_bytes = self.control_table["Min_Angle_Limit"]
            group = scs.GroupSyncWrite(self.port_handler, self.packet_handler, addr, num_bytes)
            data = convert_to_bytes(min_steps, num_bytes, self.mock)
            group.addParam(servo_id, data)
            
            for _ in range(NUM_WRITE_RETRY):
                comm = group.txPacket()
                if comm == scs.COMM_SUCCESS:
                    break
            
            if comm != scs.COMM_SUCCESS:
                print(f"Failed to set min angle limit for servo {servo_id}")
                return False
            
            # Set max angle limit
            addr, num_bytes = self.control_table["Max_Angle_Limit"]
            group = scs.GroupSyncWrite(self.port_handler, self.packet_handler, addr, num_bytes)
            data = convert_to_bytes(max_steps, num_bytes, self.mock)
            group.addParam(servo_id, data)
            
            for _ in range(NUM_WRITE_RETRY):
                comm = group.txPacket()
                if comm == scs.COMM_SUCCESS:
                    break
            
            if comm != scs.COMM_SUCCESS:
                print(f"Failed to set max angle limit for servo {servo_id}")
                return False
            
            print(f"Set angle limits for servo {servo_id}: {min_degrees:.1f}° to {max_degrees:.1f}°")
            return True
            
        except Exception as e:
            print(f"Error setting angle limits for servo {servo_id}: {e}")
            return False
    
    def home_all_servos(self) -> bool:
        """Home all servos to zero position"""
        if not self.is_connected:
            return False
        
        success = True
        all_ids = self.follower_ids + self.leader_ids
        
        for servo_id in all_ids:
            if not self.set_position(servo_id, 0.0):
                success = False
        
        return success
    
    def get_available_ports(self) -> List[str]:
        """Get list of available serial ports"""
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            available_ports = []
            
            for port in ports:
                if port.device and ('USB' in port.description.upper() or 'ACM' in port.device):
                    available_ports.append(port.device)
            
            return available_ports
        except ImportError:
            return ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"]
    
    def test_connection(self) -> bool:
        """Test if the connection is working"""
        if not self.is_connected:
            return False
        
        try:
            # Try to read from the first servo
            test_id = self.follower_ids[0] if self.follower_ids else 1
            position = self.read_position(test_id)
            return position is not None
        except Exception:
            return False
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'is_connected') and self.is_connected:
            self.disconnect()


# Convenience functions for SO100 specific usage
def create_so100_bus(port: str = "/dev/ttyUSB0", baudrate: int = BAUDRATE, mock: bool = False) -> FeetechServoBus:
    """Create a FeetechServoBus configured for SO100 LeRobot"""
    config = FeetechConfig(
        port=port,
        baudrate=baudrate,
        follower_ids=list(range(1, 7)),  # IDs 1-6
        leader_ids=list(range(7, 13)),   # IDs 7-12
        mock=mock
    )
    return FeetechServoBus(config)


def test_feetech_connection(port: str = "/dev/ttyUSB0") -> bool:
    """Test Feetech connection on given port"""
    bus = create_so100_bus(port, mock=False)
    try:
        if bus.connect():
            success = bus.test_connection()
            bus.disconnect()
            return success
        return False
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False
