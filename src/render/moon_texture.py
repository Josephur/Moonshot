"""Moon surface texture loading and UV sampling.

Provides a cached loader for the bundled equirectangular lunar texture
and a bilinear-interpolated UV sampler.
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from PIL import Image

_TEXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "moon_texture.png",
)


@lru_cache(maxsize=1)
def load_texture(path: str = _TEXTURE_PATH) -> Optional[np.ndarray]:
    """Load moon texture as cached numpy array (H×W×3 uint8).

    Args:
        path: Filesystem path to the texture PNG.

    Returns:
        ``(height, width, 3)`` uint8 array, or ``None`` if the file
        could not be loaded.
    """
    try:
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except (FileNotFoundError, OSError, Exception) as exc:
        warnings.warn(f"Could not load moon texture from {path}: {exc}")
        return None


def sample_texture(
    lon_rad: float,
    lat_rad: float,
    texture: np.ndarray,
) -> Tuple[int, int, int]:
    """Bilinear-interpolated RGB sample at (*lon*, *lat*).

    Args:
        lon_rad: Longitude in radians in [-π, π].
        lat_rad: Latitude in radians in [-π/2, π/2].
        texture: ``(H, W, 3)`` uint8 equirectangular texture array.

    Returns:
        ``(r, g, b)`` tuple of integers in [0, 255].
    """
    h, w = texture.shape[:2]

    # Map (lon, lat) to normalised UV in [0, 1]
    u = (lon_rad + np.pi) / (2.0 * np.pi)  # lon ∈ [-π, π] → u ∈ [0, 1]
    v = (np.pi / 2.0 - lat_rad) / np.pi    # lat ∈ [-π/2, π/2] → v ∈ [0, 1]

    # Wrap longitude
    u = u % 1.0
    v = np.clip(v, 0.0, 1.0)

    # Scaled pixel coordinates
    x = u * (w - 1)
    y = v * (h - 1)

    # Integer bounding corners
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, h - 1)

    # Fractional offsets
    fx = x - x0
    fy = y - y0

    # Four texel samples
    c00 = texture[y0, x0].astype(np.float32)
    c10 = texture[y0, x1].astype(np.float32)
    c01 = texture[y1, x0].astype(np.float32)
    c11 = texture[y1, x1].astype(np.float32)

    # Bilinear interpolation — first along X, then along Y
    c0 = c00 + fx * (c10 - c00)
    c1 = c01 + fx * (c11 - c01)
    result = c0 + fy * (c1 - c0)

    return (int(round(result[0])), int(round(result[1])), int(round(result[2])))
