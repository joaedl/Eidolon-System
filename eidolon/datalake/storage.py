"""Data storage and management for the data lake."""

import asyncio
import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import structlog
import boto3
from botocore.exceptions import ClientError
import numpy as np
import cv2

from ..common.config import get_config

logger = structlog.get_logger(__name__)


class DataType(Enum):
    """Data type enumeration."""
    TELEMETRY = "telemetry"
    VIDEO = "video"
    AUDIO = "audio"
    SENSOR = "sensor"
    ROSBAG = "rosbag"
    MODEL = "model"
    DATASET = "dataset"


class StorageTier(Enum):
    """Storage tier enumeration."""
    HOT = "hot"      # Frequently accessed
    WARM = "warm"    # Occasionally accessed
    COLD = "cold"    # Rarely accessed
    ARCHIVE = "archive"  # Long-term archival


@dataclass
class DataObject:
    """Data object metadata."""
    id: str
    type: DataType
    tenant_id: str
    robot_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0
    checksum: str = ""
    storage_tier: StorageTier = StorageTier.HOT
    retention_days: int = 30
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """Dataset definition."""
    id: str
    name: str
    description: str
    tenant_id: str
    data_type: DataType
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    object_count: int = 0
    total_size: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class S3StorageManager:
    """S3-compatible storage manager."""
    
    def __init__(self, config):
        self.config = config
        self.s3_client = boto3.client(
            's3',
            endpoint_url=config.storage.s3_endpoint,
            aws_access_key_id=config.storage.s3_access_key,
            aws_secret_access_key=config.storage.s3_secret_key,
            region_name=config.storage.s3_region
        )
        self.bucket_name = config.storage.s3_bucket
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.s3_client.create_bucket(Bucket=self.bucket_name)
            logger.info("Created S3 bucket", bucket=self.bucket_name)
    
    async def upload_object(self, key: str, data: bytes, metadata: Dict[str, str] = None) -> str:
        """Upload object to S3."""
        try:
            # Calculate checksum
            checksum = hashlib.sha256(data).hexdigest()
            
            # Upload with metadata
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                **extra_args
            )
            
            logger.info("Object uploaded", key=key, size=len(data), checksum=checksum)
            return checksum
            
        except Exception as e:
            logger.error("Failed to upload object", key=key, error=str(e))
            raise
    
    async def download_object(self, key: str) -> bytes:
        """Download object from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except Exception as e:
            logger.error("Failed to download object", key=key, error=str(e))
            raise
    
    async def delete_object(self, key: str):
        """Delete object from S3."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info("Object deleted", key=key)
        except Exception as e:
            logger.error("Failed to delete object", key=key, error=str(e))
            raise
    
    async def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """List objects in bucket."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            return response.get('Contents', [])
        except Exception as e:
            logger.error("Failed to list objects", prefix=prefix, error=str(e))
            return []
    
    async def generate_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """Generate presigned URL for object access."""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error("Failed to generate presigned URL", key=key, error=str(e))
            raise


class DataLakeManager:
    """Main data lake manager."""
    
    def __init__(self):
        self.config = get_config()
        self.storage_manager = S3StorageManager(self.config)
        self.data_objects: Dict[str, DataObject] = {}
        self.datasets: Dict[str, Dataset] = {}
        self.running = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the data lake manager."""
        self.running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Data lake manager started")
    
    async def stop(self):
        """Stop the data lake manager."""
        self.running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Data lake manager stopped")
    
    async def _cleanup_loop(self):
        """Cleanup expired data objects."""
        while self.running:
            try:
                await self._cleanup_expired_objects()
                await asyncio.sleep(3600)  # 1-hour cleanup interval
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(60)
    
    async def _cleanup_expired_objects(self):
        """Clean up expired data objects."""
        current_time = datetime.utcnow()
        expired_objects = []
        
        for obj_id, obj in self.data_objects.items():
            if current_time > obj.created_at + timedelta(days=obj.retention_days):
                expired_objects.append(obj_id)
        
        for obj_id in expired_objects:
            await self.delete_data_object(obj_id)
            logger.info("Expired object cleaned up", object_id=obj_id)
    
    async def store_telemetry(self, robot_id: str, tenant_id: str, telemetry_data: Dict[str, Any]) -> str:
        """Store telemetry data."""
        object_id = f"telemetry_{robot_id}_{int(time.time())}"
        key = f"telemetry/{tenant_id}/{robot_id}/{object_id}.json"
        
        # Serialize data
        data_bytes = json.dumps(telemetry_data).encode('utf-8')
        
        # Upload to storage
        checksum = await self.storage_manager.upload_object(
            key, data_bytes, {"type": "telemetry", "robot_id": robot_id}
        )
        
        # Create data object
        data_object = DataObject(
            id=object_id,
            type=DataType.TELEMETRY,
            tenant_id=tenant_id,
            robot_id=robot_id,
            size_bytes=len(data_bytes),
            checksum=checksum,
            storage_tier=StorageTier.HOT,
            retention_days=30,
            metadata=telemetry_data
        )
        
        self.data_objects[object_id] = data_object
        logger.info("Telemetry stored", object_id=object_id, robot_id=robot_id)
        
        return object_id
    
    async def store_video_frame(self, robot_id: str, tenant_id: str, session_id: str, 
                               frame_data: bytes, frame_metadata: Dict[str, Any]) -> str:
        """Store video frame."""
        object_id = f"video_{robot_id}_{session_id}_{int(time.time())}"
        key = f"video/{tenant_id}/{robot_id}/{session_id}/{object_id}.jpg"
        
        # Upload to storage
        checksum = await self.storage_manager.upload_object(
            key, frame_data, {"type": "video", "robot_id": robot_id, "session_id": session_id}
        )
        
        # Create data object
        data_object = DataObject(
            id=object_id,
            type=DataType.VIDEO,
            tenant_id=tenant_id,
            robot_id=robot_id,
            session_id=session_id,
            size_bytes=len(frame_data),
            checksum=checksum,
            storage_tier=StorageTier.WARM,
            retention_days=7,
            metadata=frame_metadata
        )
        
        self.data_objects[object_id] = data_object
        logger.info("Video frame stored", object_id=object_id, robot_id=robot_id)
        
        return object_id
    
    async def store_rosbag(self, robot_id: str, tenant_id: str, session_id: str, 
                          rosbag_data: bytes, bag_metadata: Dict[str, Any]) -> str:
        """Store ROS bag file."""
        object_id = f"rosbag_{robot_id}_{session_id}_{int(time.time())}"
        key = f"rosbags/{tenant_id}/{robot_id}/{session_id}/{object_id}.bag"
        
        # Upload to storage
        checksum = await self.storage_manager.upload_object(
            key, rosbag_data, {"type": "rosbag", "robot_id": robot_id, "session_id": session_id}
        )
        
        # Create data object
        data_object = DataObject(
            id=object_id,
            type=DataType.ROSBAG,
            tenant_id=tenant_id,
            robot_id=robot_id,
            session_id=session_id,
            size_bytes=len(rosbag_data),
            checksum=checksum,
            storage_tier=StorageTier.COLD,
            retention_days=90,
            metadata=bag_metadata
        )
        
        self.data_objects[object_id] = data_object
        logger.info("ROS bag stored", object_id=object_id, robot_id=robot_id)
        
        return object_id
    
    async def store_model(self, model_id: str, tenant_id: str, model_data: bytes, 
                         model_metadata: Dict[str, Any]) -> str:
        """Store ML model."""
        object_id = f"model_{model_id}_{int(time.time())}"
        key = f"models/{tenant_id}/{model_id}/{object_id}.pkl"
        
        # Upload to storage
        checksum = await self.storage_manager.upload_object(
            key, model_data, {"type": "model", "model_id": model_id}
        )
        
        # Create data object
        data_object = DataObject(
            id=object_id,
            type=DataType.MODEL,
            tenant_id=tenant_id,
            size_bytes=len(model_data),
            checksum=checksum,
            storage_tier=StorageTier.HOT,
            retention_days=365,
            metadata=model_metadata
        )
        
        self.data_objects[object_id] = data_object
        logger.info("Model stored", object_id=object_id, model_id=model_id)
        
        return object_id
    
    async def get_data_object(self, object_id: str) -> Optional[DataObject]:
        """Get data object by ID."""
        return self.data_objects.get(object_id)
    
    async def download_data_object(self, object_id: str) -> Optional[bytes]:
        """Download data object."""
        data_object = self.data_objects.get(object_id)
        if not data_object:
            return None
        
        # Construct key from object metadata
        key = self._get_object_key(data_object)
        
        try:
            return await self.storage_manager.download_object(key)
        except Exception as e:
            logger.error("Failed to download data object", object_id=object_id, error=str(e))
            return None
    
    async def delete_data_object(self, object_id: str):
        """Delete data object."""
        data_object = self.data_objects.get(object_id)
        if not data_object:
            return
        
        # Delete from storage
        key = self._get_object_key(data_object)
        try:
            await self.storage_manager.delete_object(key)
        except Exception as e:
            logger.error("Failed to delete object from storage", key=key, error=str(e))
        
        # Remove from metadata
        del self.data_objects[object_id]
        logger.info("Data object deleted", object_id=object_id)
    
    def _get_object_key(self, data_object: DataObject) -> str:
        """Get S3 key for data object."""
        if data_object.type == DataType.TELEMETRY:
            return f"telemetry/{data_object.tenant_id}/{data_object.robot_id}/{data_object.id}.json"
        elif data_object.type == DataType.VIDEO:
            return f"video/{data_object.tenant_id}/{data_object.robot_id}/{data_object.session_id}/{data_object.id}.jpg"
        elif data_object.type == DataType.ROSBAG:
            return f"rosbags/{data_object.tenant_id}/{data_object.robot_id}/{data_object.session_id}/{data_object.id}.bag"
        elif data_object.type == DataType.MODEL:
            return f"models/{data_object.tenant_id}/{data_object.id}.pkl"
        else:
            return f"data/{data_object.tenant_id}/{data_object.id}"
    
    async def create_dataset(self, name: str, description: str, tenant_id: str, 
                           data_type: DataType, tags: Dict[str, str] = None) -> Dataset:
        """Create a new dataset."""
        dataset_id = f"dataset_{int(time.time())}"
        
        dataset = Dataset(
            id=dataset_id,
            name=name,
            description=description,
            tenant_id=tenant_id,
            data_type=data_type,
            tags=tags or {}
        )
        
        self.datasets[dataset_id] = dataset
        logger.info("Dataset created", dataset_id=dataset_id, name=name)
        
        return dataset
    
    async def add_to_dataset(self, dataset_id: str, object_id: str):
        """Add data object to dataset."""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        if object_id not in self.data_objects:
            raise ValueError(f"Data object not found: {object_id}")
        
        dataset = self.datasets[dataset_id]
        data_object = self.data_objects[object_id]
        
        # Update dataset metadata
        dataset.object_count += 1
        dataset.total_size += data_object.size_bytes
        dataset.updated_at = datetime.utcnow()
        
        # Add to dataset metadata
        if "objects" not in dataset.metadata:
            dataset.metadata["objects"] = []
        dataset.metadata["objects"].append(object_id)
        
        logger.info("Object added to dataset", dataset_id=dataset_id, object_id=object_id)
    
    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Get dataset by ID."""
        return self.datasets.get(dataset_id)
    
    async def list_datasets(self, tenant_id: str) -> List[Dataset]:
        """List datasets for tenant."""
        return [d for d in self.datasets.values() if d.tenant_id == tenant_id]
    
    async def generate_presigned_url(self, object_id: str, expiration: int = 3600) -> Optional[str]:
        """Generate presigned URL for data object."""
        data_object = self.data_objects.get(object_id)
        if not data_object:
            return None
        
        key = self._get_object_key(data_object)
        return await self.storage_manager.generate_presigned_url(key, expiration)
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_objects = len(self.data_objects)
        total_size = sum(obj.size_bytes for obj in self.data_objects.values())
        
        type_counts = {}
        for obj in self.data_objects.values():
            type_counts[obj.type.value] = type_counts.get(obj.type.value, 0) + 1
        
        return {
            "total_objects": total_objects,
            "total_size_bytes": total_size,
            "type_counts": type_counts,
            "datasets": len(self.datasets)
        }
