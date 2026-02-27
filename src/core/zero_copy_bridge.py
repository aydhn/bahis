"""
zero_copy_bridge.py – Zero-Copy Data Transfer via Shared Memory.

Achieves microsecond-level latency by eliminating serialization overhead
between Ingestion and Inference stages.
"""
from __future__ import annotations

import numpy as np
from multiprocessing import shared_memory
from typing import Tuple, Optional
import os
import fcntl
from loguru import logger

class ZeroCopyBridge:
    """
    High-performance bridge for transferring numpy arrays between processes
    without copying/pickling. Use for high-frequency feature vectors.
    """

    def __init__(self, name: str, shape: Tuple[int, ...], dtype=np.float32, create: bool = False):
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(shape) * self.dtype.itemsize)
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._arr: Optional[np.ndarray] = None
        self._lock_file = f"/tmp/{self.name}.lock"
        self._create = create

        self._connect()

    def _connect(self):
        """Connect to or create shared memory segment."""
        try:
            if self._create:
                # Clean up existing if we are the creator
                try:
                    temp = shared_memory.SharedMemory(name=self.name)
                    temp.close()
                    temp.unlink()
                except FileNotFoundError:
                    pass

                self._shm = shared_memory.SharedMemory(create=True, size=self.nbytes, name=self.name)
                # Initialize lock file
                with open(self._lock_file, 'w') as f:
                    f.write("LOCK")
                logger.info(f"Created shared memory segment: {self.name} ({self.nbytes} bytes)")
            else:
                self._shm = shared_memory.SharedMemory(name=self.name)
                logger.debug(f"Connected to shared memory segment: {self.name}")

            self._arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self._shm.buf)

        except Exception as e:
            logger.error(f"ZeroCopyBridge connection failed: {e}")
            raise

    def write(self, data: np.ndarray):
        """
        Write data to shared buffer with atomic locking.
        """
        if self._arr is None:
            raise RuntimeError("Shared buffer not connected")

        # Ensure data matches expectations
        # If inputs are not exactly correct shape, try reshape
        try:
            data_reshaped = data.astype(self.dtype).reshape(self.shape)
        except ValueError as e:
            raise ValueError(f"Data shape mismatch. Expected {self.shape}, got {data.shape}") from e

        with open(self._lock_file, 'r+') as lock_f:
            # Acquire exclusive lock
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                # Write to buffer
                np.copyto(self._arr, data_reshaped)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def read(self) -> np.ndarray:
        """
        Read data from shared buffer with atomic locking.
        Returns a copy to ensure thread safety after reading.
        """
        if self._arr is None:
            raise RuntimeError("Shared buffer not connected")

        # Use open with 'r' is enough for locking, but must exist
        with open(self._lock_file, 'r') as lock_f:
            # Acquire shared lock (using exclusive for simplicity to block writers)
            fcntl.flock(lock_f, fcntl.LOCK_SH)
            try:
                # Return a copy to avoid data changing while processing
                return self._arr.copy()
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def close(self):
        """Close access to shared memory."""
        if self._shm:
            self._shm.close()

    def unlink(self):
        """Destroy shared memory segment (Owner only)."""
        if self._shm:
            try:
                self._shm.close()
                self._shm.unlink()
                if os.path.exists(self._lock_file):
                    os.remove(self._lock_file)
                logger.info(f"Unlinked shared memory: {self.name}")
            except FileNotFoundError:
                pass
