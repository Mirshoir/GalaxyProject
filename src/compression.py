"""
Core compression and NCD implementation.
This module provides functions for:
1. Converting images to canonical representations
2. Computing compressed sizes
3. Calculating Normalized Compression Distance (NCD)
"""
import zlib
import bz2
import lzma
from PIL import Image
import numpy as np
from typing import Tuple, Union, Optional
from dataclasses import dataclass
import hashlib


@dataclass
class Compressor:
    """Configuration for compression algorithm."""
    name: str = "zlib"
    level: int = 9

    def compress(self, data: bytes) -> bytes:
        """Compress data using the configured algorithm."""
        if self.name == "zlib":
            return zlib.compress(data, level=self.level)
        elif self.name == "bz2":
            return bz2.compress(data, compresslevel=self.level)
        elif self.name == "lzma":
            return lzma.compress(data, preset=self.level)
        else:
            raise ValueError(f"Unknown compressor: {self.name}")

    def compressed_size(self, data: bytes) -> int:
        """Return size of compressed data in bytes."""
        return len(self.compress(data))


class CanonicalImageProcessor:
    """Handles conversion of images to canonical representations."""

    def __init__(self,
                 target_size: Tuple[int, int] = (256, 256),
                 normalize: bool = True,
                 apply_smoothing: bool = True,
                 sigma: float = 1.0,
                 threshold: Optional[float] = None):
        self.target_size = target_size
        self.normalize = normalize
        self.apply_smoothing = apply_smoothing
        self.sigma = sigma
        self.threshold = threshold

    def process(self, image_path: str) -> bytes:
        """
        Convert image to canonical byte representation.

        Steps:
        1. Load and convert to grayscale
        2. Resize to target dimensions
        3. Apply Gaussian smoothing (optional)
        4. Normalize intensities (optional)
        5. Apply threshold (optional)
        6. Convert to deterministic byte sequence
        """
        # Load and convert to grayscale
        img = Image.open(image_path).convert('L')

        # Resize to target dimensions
        img = img.resize(self.target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        arr = np.array(img, dtype=np.float32)

        # Apply Gaussian smoothing if requested
        if self.apply_smoothing:
            from scipy.ndimage import gaussian_filter
            arr = gaussian_filter(arr, sigma=self.sigma)

        # Normalize intensities
        if self.normalize:
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max - arr_min > 0:
                arr = (arr - arr_min) / (arr_max - arr_min)
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = arr.astype(np.uint8)

        # Apply threshold if specified
        if self.threshold is not None:
            if self.normalize:
                threshold_value = self.threshold * 255
            else:
                threshold_value = self.threshold
            arr = (arr > threshold_value).astype(np.uint8) * 255

        # Ensure deterministic byte order (row-major)
        return arr.tobytes(order='C')


class NCDCalculator:
    """Main class for computing Normalized Compression Distance."""

    def __init__(self, compressor: Compressor = None, processor: CanonicalImageProcessor = None):
        self.compressor = compressor or Compressor()
        self.processor = processor or CanonicalImageProcessor()

    def compute_canonical_bytes(self, image_path: str) -> bytes:
        """Get canonical byte representation of an image."""
        return self.processor.process(image_path)

    def compressed_size(self, data: bytes) -> int:
        """Compute compressed size of data."""
        return self.compressor.compressed_size(data)

    def ncd(self, image_path_1: str, image_path_2: str) -> float:
        """
        Compute Normalized Compression Distance between two images.

        Formula: NCD(x,y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))

        Where:
        - C(x): compressed size of image x
        - C(y): compressed size of image y
        - C(xy): compressed size of concatenated images
        """
        # Get canonical representations
        x_bytes = self.compute_canonical_bytes(image_path_1)
        y_bytes = self.compute_canonical_bytes(image_path_2)

        # Compute individual compressed sizes
        Cx = self.compressed_size(x_bytes)
        Cy = self.compressed_size(y_bytes)

        # Compute compressed size of concatenation
        # Note: We concatenate in a deterministic order
        Cxy = self.compressed_size(x_bytes + y_bytes)

        # Compute NCD
        min_C = min(Cx, Cy)
        max_C = max(Cx, Cy)

        if max_C == 0:
            return 0.0

        return (Cxy - min_C) / max_C

    def compute_self_distance(self, image_path: str) -> Tuple[float, int]:
        """Compute distance of image with itself (should be close to 0)."""
        x_bytes = self.compute_canonical_bytes(image_path)
        Cx = self.compressed_size(x_bytes)
        Cxx = self.compressed_size(x_bytes + x_bytes)

        if Cx == 0:
            return 0.0, Cx

        return (Cxx - Cx) / Cx, Cx


# Default instances for convenience
default_processor = CanonicalImageProcessor()
default_compressor = Compressor()
default_ncd_calculator = NCDCalculator()