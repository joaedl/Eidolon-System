"""Security utilities for device authentication and encryption."""

import hashlib
import hmac
import os
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import structlog

logger = structlog.get_logger(__name__)


class DeviceCertificate:
    """Device certificate management."""
    
    def __init__(self, cert_path: str, key_path: str, ca_cert_path: Optional[str] = None):
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_cert_path = ca_cert_path
        self._cert: Optional[x509.Certificate] = None
        self._private_key: Optional[rsa.RSAPrivateKey] = None
        self._ca_cert: Optional[x509.Certificate] = None
    
    def load_certificate(self) -> x509.Certificate:
        """Load device certificate from file."""
        if self._cert is None:
            with open(self.cert_path, 'rb') as f:
                cert_data = f.read()
            self._cert = x509.load_pem_x509_certificate(cert_data, default_backend())
        return self._cert
    
    def load_private_key(self) -> rsa.RSAPrivateKey:
        """Load device private key from file."""
        if self._private_key is None:
            with open(self.key_path, 'rb') as f:
                key_data = f.read()
            self._private_key = serialization.load_pem_private_key(
                key_data, password=None, backend=default_backend()
            )
        return self._private_key
    
    def load_ca_certificate(self) -> Optional[x509.Certificate]:
        """Load CA certificate from file."""
        if self.ca_cert_path and self._ca_cert is None:
            with open(self.ca_cert_path, 'rb') as f:
                ca_data = f.read()
            self._ca_cert = x509.load_pem_x509_certificate(ca_data, default_backend())
        return self._ca_cert
    
    def get_device_id(self) -> str:
        """Extract device ID from certificate."""
        cert = self.load_certificate()
        # Look for device ID in certificate subject or SAN
        for name in cert.subject:
            if name.oid == x509.NameOID.COMMON_NAME:
                return name.value
        raise ValueError("Device ID not found in certificate")
    
    def verify_certificate_chain(self) -> bool:
        """Verify certificate chain against CA."""
        try:
            cert = self.load_certificate()
            ca_cert = self.load_ca_certificate()
            
            if ca_cert is None:
                logger.warning("No CA certificate provided for verification")
                return False
            
            # Verify certificate signature
            ca_public_key = ca_cert.public_key()
            ca_public_key.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert.signature_algorithm_oid._name
            )
            
            # Check certificate validity
            now = datetime.utcnow()
            if cert.not_valid_before > now or cert.not_valid_after < now:
                logger.warning("Certificate is not valid at current time")
                return False
            
            return True
        except Exception as e:
            logger.error("Certificate verification failed", error=str(e))
            return False


class JWTManager:
    """JWT token management."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, payload: Dict[str, Any], expires_minutes: int = 30) -> str:
        """Create a JWT token."""
        now = datetime.utcnow()
        payload.update({
            "iat": now,
            "exp": now + timedelta(minutes=expires_minutes)
        })
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid JWT token", error=str(e))
            return None
    
    def create_device_token(self, device_id: str, tenant_id: str, permissions: list) -> str:
        """Create a device-specific JWT token."""
        payload = {
            "device_id": device_id,
            "tenant_id": tenant_id,
            "permissions": permissions,
            "token_type": "device"
        }
        return self.create_token(payload, expires_minutes=60)
    
    def create_operator_token(self, operator_id: str, tenant_id: str, roles: list) -> str:
        """Create an operator-specific JWT token."""
        payload = {
            "operator_id": operator_id,
            "tenant_id": tenant_id,
            "roles": roles,
            "token_type": "operator"
        }
        return self.create_token(payload, expires_minutes=30)


class EncryptionManager:
    """Encryption utilities for secure communication."""
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or self._generate_key()
    
    def _generate_key(self) -> bytes:
        """Generate a random encryption key."""
        return hashlib.sha256(str(time.time()).encode()).digest()
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using AES-256-GCM."""
        iv = os.urandom(12)  # 96-bit IV for GCM
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return iv + encryptor.tag + ciphertext
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using AES-256-GCM."""
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def create_hmac(self, data: bytes) -> str:
        """Create HMAC for data integrity."""
        return hmac.new(self.key, data, hashlib.sha256).hexdigest()
    
    def verify_hmac(self, data: bytes, signature: str) -> bool:
        """Verify HMAC signature."""
        expected = self.create_hmac(data)
        return hmac.compare_digest(expected, signature)


class SecurityManager:
    """Main security manager for the system."""
    
    def __init__(self, config):
        self.config = config
        self.jwt_manager = JWTManager(
            config.security.jwt_secret_key,
            config.security.jwt_algorithm
        )
        self.encryption_manager = EncryptionManager()
        self.device_cert: Optional[DeviceCertificate] = None
        
        if config.security.device_cert_path and config.security.device_key_path:
            self.device_cert = DeviceCertificate(
                config.security.device_cert_path,
                config.security.device_key_path,
                config.security.ca_cert_path
            )
    
    def get_device_identity(self) -> Optional[Dict[str, Any]]:
        """Get device identity from certificate."""
        if not self.device_cert:
            return None
        
        try:
            device_id = self.device_cert.get_device_id()
            if self.device_cert.verify_certificate_chain():
                return {
                    "device_id": device_id,
                    "certificate_valid": True,
                    "certificate": self.device_cert.load_certificate()
                }
        except Exception as e:
            logger.error("Failed to get device identity", error=str(e))
        
        return None
    
    def create_device_token(self, device_id: str, tenant_id: str) -> str:
        """Create a device authentication token."""
        permissions = ["telemetry", "subgoal", "heartbeat"]
        return self.jwt_manager.create_device_token(device_id, tenant_id, permissions)
    
    def create_operator_token(self, operator_id: str, tenant_id: str, roles: list) -> str:
        """Create an operator authentication token."""
        return self.jwt_manager.create_operator_token(operator_id, tenant_id, roles)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify any JWT token."""
        return self.jwt_manager.verify_token(token)
    
    def encrypt_sensitive_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data."""
        return self.encryption_manager.encrypt_data(data)
    
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data."""
        return self.encryption_manager.decrypt_data(encrypted_data)
