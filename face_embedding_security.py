"""Secure storage and retrieval of face embeddings."""

from __future__ import annotations

import base64
import json
import math
import os

import bcrypt
from cryptography.fernet import Fernet, InvalidToken


class FaceEmbeddingManager:
    """Encrypt, decrypt, and fingerprint face embeddings safely."""

    def __init__(self) -> None:
        """Initialize the Fernet cipher from an environment key."""
        key = os.environ.get("EMBEDDING_ENCRYPTION_KEY")
        if not key:
            secret = (
                os.environ.get("EMBEDDING_SECRET")
                or os.environ.get("SECRET_KEY")
                or os.environ.get("JWT_SECRET")
            )
            if not secret:
                raise RuntimeError(
                    "Set EMBEDDING_ENCRYPTION_KEY, EMBEDDING_SECRET, SECRET_KEY, or JWT_SECRET."
                )
            key, _ = self.derive_key_from_password(
                secret,
                salt=(os.environ.get("EMBEDDING_ENCRYPTION_SALT") or "dvseit-embedding-key").encode(
                    "utf-8"
                ),
            )

        try:
            self.cipher = Fernet(key.encode() if isinstance(key, str) else key)
        except Exception as exc:
            raise ValueError(f"Invalid embedding encryption key: {exc}") from exc

    @staticmethod
    def _normalize_embedding(embedding: list[float]) -> list[float]:
        """Validate and normalize an embedding into a numeric float list."""
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Embedding must be a non-empty list of numeric values.")

        normalized: list[float] = []
        for value in embedding:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("Embedding contains non-finite values.")
            normalized.append(numeric)
        return normalized

    @staticmethod
    def _normalize_salt(salt: bytes | str | None) -> bytes:
        """Return a stable bcrypt-compatible salt payload."""
        if salt is None:
            return os.urandom(16)
        if isinstance(salt, str):
            salt = salt.encode("utf-8")
        return (salt[:16]).ljust(16, b"0")

    @classmethod
    def _derive_bytes(
        cls,
        payload: bytes,
        salt: bytes | str | None,
        desired_key_bytes: int = 32,
    ) -> bytes:
        """Derive deterministic bytes using bcrypt instead of SHA-based KDFs."""
        return bcrypt.kdf(
            password=payload,
            salt=cls._normalize_salt(salt),
            desired_key_bytes=desired_key_bytes,
            rounds=max(100, int(os.environ.get("EMBEDDING_BCRYPT_KDF_ROUNDS", "100"))),
        )

    def encrypt_embedding(self, embedding: list[float]) -> str:
        """Encrypt a face embedding before storing it."""
        try:
            normalized = self._normalize_embedding(embedding)
            encrypted = self.cipher.encrypt(json.dumps(normalized).encode("utf-8"))
            return base64.b64encode(encrypted).decode("utf-8")
        except Exception as exc:
            raise ValueError(f"Failed to encrypt embedding: {exc}") from exc

    def decrypt_embedding(self, encrypted_embedding: str) -> list[float]:
        """Decrypt a stored embedding back into a float list."""
        try:
            encrypted = base64.b64decode(encrypted_embedding.encode("utf-8"))
            decrypted = self.cipher.decrypt(encrypted)
            payload = json.loads(decrypted.decode("utf-8"))
            return self._normalize_embedding(payload)
        except InvalidToken as exc:
            raise ValueError("Failed to decrypt embedding: invalid key or corrupted data.") from exc
        except Exception as exc:
            raise ValueError(f"Failed to decrypt embedding: {exc}") from exc

    def hash_embedding(self, embedding: list[float]) -> str:
        """Create a deterministic bcrypt-derived hash for duplicate detection."""
        normalized = self._normalize_embedding(embedding)
        return self._derive_bytes(
            json.dumps(normalized).encode("utf-8"),
            os.environ.get("EMBEDDING_HASH_SALT") or "dvseit-embedding-hash",
        ).hex()

    def get_embedding_hash_for_comparison(self, encrypted_embedding: str) -> str:
        """Hash the encrypted payload itself for lightweight comparisons."""
        return self._derive_bytes(
            encrypted_embedding.encode("utf-8"),
            os.environ.get("EMBEDDING_HASH_SALT") or "dvseit-embedding-hash",
        ).hex()

    @staticmethod
    def generate_encryption_key() -> str:
        """Generate a Fernet-compatible encryption key."""
        return Fernet.generate_key().decode("utf-8")

    @staticmethod
    def derive_key_from_password(
        password: str, salt: bytes | None = None
    ) -> tuple[str, str]:
        """Derive a Fernet key from a password using bcrypt."""
        if not password:
            raise ValueError("Password is required to derive an encryption key.")
        normalized_salt = FaceEmbeddingManager._normalize_salt(salt)
        key = base64.urlsafe_b64encode(
            FaceEmbeddingManager._derive_bytes(password.encode("utf-8"), normalized_salt)
        )
        return key.decode("utf-8"), normalized_salt.hex()


"""
# Example usage
manager = FaceEmbeddingManager()
embedding_data = [0.1, 0.2, 0.3]
encrypted = manager.encrypt_embedding(embedding_data)
embedding_hash = manager.hash_embedding(embedding_data)
decrypted_embedding = manager.decrypt_embedding(encrypted)
"""
