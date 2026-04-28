"""
Horizon / terrain rendering for Moonshot.

Draws a dark silhouette at the bottom of the sky image representing
the ground / horizon terrain.  The horizon line includes gentle
undulations and accounts for Earth curvature (horizon dip).

All angle values are in degrees unless stated otherwise.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def horizon_dip_degrees(observer_height_m: float) -> float:
    """Compute the angular dip of the horizon due to Earth curvature.

    The horizon dip (in arcminutes) = 1.93 * sqrt(height_m).
    This function returns the value converted to degrees.

    Args:
        observer_height_m: Observer's height above sea level in metres.

    Returns:
        Horizon dip angle in degrees (always >= 0).
    """
    if observer_height_m < 0.0:
        observer_height_m = 0.0
    dip_arcmin = 1.93 * math.sqrt(observer_height_m)
    return dip_arcmin / 60.0  # convert arcminutes to degrees


def _generate_terrain(width: int, height: int,
                      horizon_y: int,
                      amplitude: int = 8,
                      frequency: float = 0.02) -> np.ndarray:
    """Generate a 1-D terrain profile across the image width.

    Uses a summation of sine waves to create gentle, natural-looking
    undulations for the horizon line.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        horizon_y: Base Y position for the horizon.
        amplitude: Maximum vertical deviation in pixels.
        frequency: Base frequency for undulations (cycles per pixel).

    Returns:
        1D numpy array of Y-positions (int) for each X column.
    """
    x = np.arange(width)
    # Sum multiple sine components for natural variation
    terrain = (
        np.sin(x * frequency * 2.0 * math.pi) * 1.0 +
        np.sin(x * frequency * 3.7 * math.pi) * 0.6 +
        np.sin(x * frequency * 0.9 * math.pi) * 1.2 +
        np.sin(x * frequency * 5.3 * math.pi) * 0.3
    )
    # Normalize to [-1, 1] then scale by amplitude
    terrain = terrain / 3.0  # sum of amplitudes = 3.1, approx 3
    return horizon_y + (terrain * amplitude).astype(int)


def horizon_line(image: Image.Image,
                 observer_height_m: float,
                 sun_altitude_deg: float) -> Image.Image:
    """Draw a dark terrain silhouette at the bottom of the image.

    The silhouette covers roughly 8–15% of the image height, with
    gentle undulations to suggest natural terrain.  The horizon line
    is lowered slightly by the horizon dip effect when the observer
    is elevated.

    During twilight/sunset, the silhouette may be slightly lighter
    if the sun is just below the horizon (forward scattering).

    Args:
        image: A PIL ``Image`` in RGB mode.  It is **modified in place**
               (a new image is returned for consistency).
        observer_height_m: Observer height above sea level in metres
                           (used for horizon dip).
        sun_altitude_deg: Sun altitude in degrees (affects silhouette
                          colour — daytime = darker silhouette, twilight
                          = slightly lighter).

    Returns:
        A new PIL ``Image`` with the horizon silhouette applied.
    """
    w, h = image.size

    # Determine horizon base Y position
    dip_deg = horizon_dip_degrees(observer_height_m)

    # Fraction of image height for the land silhouette (8–15%)
    land_fraction = 0.10 + 0.05 * (sun_altitude_deg / 90.0)
    land_fraction = max(0.08, min(0.15, land_fraction))
    base_horizon_y = int(h * (1.0 - land_fraction))

    # Apply horizon dip (small — but we scale it)
    dip_pixels = int(round(dip_deg * h / 90.0))  # ~tiny
    base_horizon_y += dip_pixels

    # Clamp
    base_horizon_y = max(10, min(h - 10, base_horizon_y))

    # Generate terrain profile
    terrain = _generate_terrain(w, h, base_horizon_y)

    # --- Determine silhouette colour ---
    if sun_altitude_deg > 0.0:
        # Daytime: solid black
        colour = (0, 0, 0)
    elif sun_altitude_deg > -12.0:
        # Twilight: dark silhouette, slightly lighter due to sky glow
        # The closer to 0°, the lighter the silhouette
        twilight_factor = (sun_altitude_deg + 12.0) / 12.0  # 0→dark, 1→lighter
        base = int(round(10 + 30 * twilight_factor))
        colour = (base, base, max(0, base - 5))
    else:
        # Night: very dark
        colour = (5, 5, 10)

    # --- Paint the silhouette ---
    # We fill everything below the terrain line with the silhouette colour
    pixels = np.array(image, dtype=np.uint8)
    for col in range(w):
        y_start = max(0, min(h - 1, terrain[col]))
        if y_start < h:
            pixels[y_start:, col, 0] = colour[0]
            pixels[y_start:, col, 1] = colour[1]
            pixels[y_start:, col, 2] = colour[2]

    from PIL import Image
    return Image.fromarray(pixels, mode="RGB")
