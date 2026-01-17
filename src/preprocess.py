"""
Image preprocessing utilities.
Handles batch processing and dataset management.
"""
import os
from glob import glob
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import json
from .compression import CanonicalImageProcessor


class GalaxyDataset:
    """Manages a collection of galaxy images."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.image_paths = []
        self.image_ids = []

    def load_images(self, pattern: str = "*.jpg") -> List[str]:
        """Load all images matching pattern from data directory."""
        search_path = os.path.join(self.data_dir, pattern)
        self.image_paths = sorted(glob(search_path))

        # Generate IDs from filenames
        self.image_ids = [os.path.splitext(os.path.basename(p))[0]
                          for p in self.image_paths]

        print(f"Loaded {len(self.image_paths)} images from {self.data_dir}")
        return self.image_paths

    def get_image_pair(self, idx1: int, idx2: int) -> Tuple[str, str]:
        """Get paths for two images by index."""
        if idx1 >= len(self.image_paths) or idx2 >= len(self.image_paths):
            raise IndexError("Image index out of range")
        return self.image_paths[idx1], self.image_paths[idx2]

    def get_image_by_id(self, image_id: str) -> str:
        """Get image path by ID."""
        for path, id_ in zip(self.image_paths, self.image_ids):
            if id_ == image_id:
                return path
        raise ValueError(f"Image ID '{image_id}' not found")

    def create_subset(self, indices: List[int]) -> 'GalaxyDataset':
        """Create a subset of the dataset."""
        subset = GalaxyDataset(self.data_dir)
        subset.image_paths = [self.image_paths[i] for i in indices]
        subset.image_ids = [self.image_ids[i] for i in indices]
        return subset


class Preprocessor:
    """Batch preprocessing utilities."""

    def __init__(self, processor: CanonicalImageProcessor = None):
        self.processor = processor or CanonicalImageProcessor()

    def process_batch(self, image_paths: List[str]) -> Dict[str, bytes]:
        """Process multiple images to canonical form."""
        results = {}
        for path in image_paths:
            try:
                results[path] = self.processor.process(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results[path] = b''
        return results

    def save_canonical_images(self, image_paths: List[str], output_dir: str):
        """Save canonical representations as numpy arrays."""
        os.makedirs(output_dir, exist_ok=True)

        for path in image_paths:
            try:
                bytes_data = self.processor.process(path)
                arr = np.frombuffer(bytes_data, dtype=np.uint8)
                arr = arr.reshape(self.processor.target_size)

                # Save as numpy file
                filename = os.path.basename(path).split('.')[0] + '.npy'
                output_path = os.path.join(output_dir, filename)
                np.save(output_path, arr)

            except Exception as e:
                print(f"Error saving {path}: {e}")


def validate_image(path: str) -> bool:
    """Check if image is valid and can be processed."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False


def get_image_stats(image_path: str) -> Dict:
    """Get basic statistics about an image."""
    with Image.open(image_path) as img:
        return {
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'width': img.width,
            'height': img.height
        }