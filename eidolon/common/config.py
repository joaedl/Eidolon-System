"""Configuration management for the Eidolon system."""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings as PydanticBaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    url: str = Field(default="postgresql+asyncpg://eidolon:eidolon@localhost:5432/eidolon")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    echo: bool = Field(default=False)
    
    class Config:
        env_prefix = "DATABASE_"


class RedisConfig(BaseSettings):
    """Redis configuration."""
    
    url: str = Field(default="redis://localhost:6379")
    max_connections: int = Field(default=10)
    socket_timeout: int = Field(default=5)
    
    class Config:
        env_prefix = "REDIS_"


class SecurityConfig(BaseSettings):
    """Security configuration."""
    
    jwt_secret_key: str = Field(..., description="JWT secret key for token signing")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=30)
    device_cert_path: Optional[str] = Field(default=None)
    device_key_path: Optional[str] = Field(default=None)
    ca_cert_path: Optional[str] = Field(default=None)
    
    class Config:
        env_prefix = "SECURITY_"


class CloudConfig(BaseSettings):
    """Cloud server configuration."""
    
    host: str = Field(default="controller.eidolon.cloud")
    port: int = Field(default=443)
    mqtt_host: str = Field(default="mqtt.eidolon.cloud")
    mqtt_port: int = Field(default=8883)
    use_tls: bool = Field(default=True)
    
    class Config:
        env_prefix = "CLOUD_"


class StorageConfig(BaseSettings):
    """Storage configuration."""
    
    s3_endpoint: str = Field(default="https://s3.eidolon.cloud")
    s3_access_key: str = Field(..., description="S3 access key")
    s3_secret_key: str = Field(..., description="S3 secret key")
    s3_bucket: str = Field(default="eidolon-data")
    s3_region: str = Field(default="us-east-1")
    
    class Config:
        env_prefix = "S3_"


class KafkaConfig(BaseSettings):
    """Kafka configuration."""
    
    bootstrap_servers: str = Field(default="localhost:9092")
    telemetry_topic: str = Field(default="telemetry")
    events_topic: str = Field(default="events")
    security_protocol: str = Field(default="PLAINTEXT")
    
    class Config:
        env_prefix = "KAFKA_"


class WebRTCConfig(BaseSettings):
    """WebRTC configuration."""
    
    turn_server: str = Field(default="turn.eidolon.cloud")
    turn_port: int = Field(default=3478)
    turn_username: Optional[str] = Field(default=None)
    turn_password: Optional[str] = Field(default=None)
    stun_servers: list = Field(default=["stun:stun.l.google.com:19302"])
    
    class Config:
        env_prefix = "TURN_"


class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""
    
    prometheus_port: int = Field(default=9090)
    grafana_port: int = Field(default=3000)
    sentry_dsn: Optional[str] = Field(default=None)
    log_level: str = Field(default="INFO")
    
    class Config:
        env_prefix = "MONITORING_"


class RobotConfig(BaseSettings):
    """Robot-specific configuration."""
    
    robot_id: str = Field(..., description="Unique robot identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    safety_enabled: bool = Field(default=True)
    autonomous_mode: bool = Field(default=True)
    max_velocity: float = Field(default=1.0)
    max_acceleration: float = Field(default=2.0)
    safety_zone_radius: float = Field(default=2.0)
    
    class Config:
        env_prefix = "ROBOT_"


class Config(PydanticBaseSettings):
    """Main configuration class."""
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    webrtc: WebRTCConfig = Field(default_factory=WebRTCConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    robot: RobotConfig = Field(default_factory=RobotConfig)
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_config() -> Config:
    """Get the application configuration."""
    return Config()


def get_robot_config() -> RobotConfig:
    """Get robot-specific configuration."""
    return get_config().robot


def get_cloud_config() -> CloudConfig:
    """Get cloud configuration."""
    return get_config().cloud
