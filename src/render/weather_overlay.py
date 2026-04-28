"""
Weather visual effects for Moonshot.

Provides functions to overlay clouds, haze, and fog onto a rendered
sky/moon image for realistic atmospheric conditions.

All overlay functions accept and return PIL ``Image`` objects in RGBA
mode (or convert if needed).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


# ---------------------------------------------------------------------------
#  Perlin-ish noise helper
# ---------------------------------------------------------------------------

def _perlin_noise(width: int, height: int,
                  scale: float = 40.0,
                  octaves: int = 4,
                  seed: int = 0) -> np.ndarray:
    """Generate a simple Perlin-like noise array.

    Uses overlapping octaves of smooth random noise (value noise with
    bilinear interpolation) to produce a cloud-like texture.

    Args:
        width: Output width.
        height: Output height.
        scale: Feature size in pixels (larger = bigger blobs).
        octaves: Number of noise octaves to sum.
        seed: RNG seed for reproducibility.

    Returns:
        A ``(height, width)`` float array in range [0, 1].
    """
    rng = np.random.RandomState(seed)
    total = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    max_amplitude = 0.0

    for o in range(octaves):
        # Determine grid size for this octave
        s = scale / (1.5 ** o)
        grid_w = max(2, int(width / s) + 1)
        grid_h = max(2, int(height / s) + 1)

        grid = rng.rand(grid_h, grid_w).astype(np.float32)

        # Upscale to target size using PIL resize (bilinear)
        grid_img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
        grid_img = grid_img.resize((width, height), Image.BILINEAR)
        noise = np.array(grid_img, dtype=np.float32) / 255.0

        total += noise * amplitude
        max_amplitude += amplitude
        amplitude *= 0.5

    return total / max_amplitude


# ---------------------------------------------------------------------------
#  Clouds
# ---------------------------------------------------------------------------

def render_clouds(image: Image.Image,
                  cloud_cover_pct: float,
                  moon_pos_xy: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Overlay semi-transparent clouds onto the image.

    Clouds are generated from Perlin-like noise and composited as a
    semi-transparent white/gray layer.  Higher cloud cover produces
    more opaque and extensive cloud coverage.

    If ``moon_pos_xy`` is provided and cloud cover is high, the moon
    region will be dimmed and given a soft glow effect to simulate
    moonlight through thin clouds.

    Args:
        image: Input PIL ``Image`` (RGB or RGBA).
        cloud_cover_pct: Cloud cover percentage (0–100).
        moon_pos_xy: ``(x, y)`` pixel coordinate of the moon centre,
                     or None to skip moon-cloud interaction.

    Returns:
        A new PIL ``Image`` with clouds composited.
    """
    cloud_cover_pct = max(0.0, min(100.0, cloud_cover_pct))
    if cloud_cover_pct < 1.0:
        return image.convert("RGBA") if image.mode != "RGBA" else image

    # Ensure RGBA for compositing
    base = image.convert("RGBA")
    w, h = base.size

    # Generate cloud noise
    # Seed based on image dimensions for reproducibility
    seed = hash((w, h, int(cloud_cover_pct))) % (2 ** 31)
    noise = _perlin_noise(w, h, scale=60.0, octaves=5, seed=seed)

    # Threshold and opacity based on cloud cover fraction
    cover_fraction = cloud_cover_pct / 100.0

    # Clouds appear where noise exceeds a threshold, which decreases
    # as cloud cover increases
    threshold = 0.55 - cover_fraction * 0.35  # 0.55 → 0.20

    # Cloud density: where noise > threshold, how opaque?
    cloud_density = np.clip((noise - threshold) / (1.0 - threshold), 0.0, 1.0)

    # Opacity scales with cloud cover
    opacity = cover_fraction * 0.6  # max ~60% opacity for complete overcast
    cloud_alpha = (cloud_density * opacity * 255).astype(np.uint8)

    # Cloud colour: white/gray, slightly darker for thicker clouds
    cloud_brightness = 220 - (cloud_density * 40).astype(np.uint8)
    cloud_r = cloud_brightness
    cloud_g = cloud_brightness
    cloud_b = cloud_brightness + 20  # slightly blue tint

    # Build cloud RGBA layer
    cloud_layer = np.zeros((h, w, 4), dtype=np.uint8)
    cloud_layer[:, :, 0] = cloud_r
    cloud_layer[:, :, 1] = cloud_g
    cloud_layer[:, :, 2] = cloud_b
    cloud_layer[:, :, 3] = cloud_alpha

    # --- Moon behind clouds effect ---
    if moon_pos_xy is not None and cloud_cover_pct > 40:
        mx, my = moon_pos_xy
        # Create a glow for the moon through the clouds
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - mx) ** 2 + (y_grid - my) ** 2)

        # Find cloud density at moon position
        moon_idx_x = max(0, min(w - 1, mx))
        moon_idx_y = max(0, min(h - 1, my))
        moon_cloud_density = cloud_density[moon_idx_y, moon_idx_x]

        # If moon is behind thick clouds, add a warm glow around it
        if moon_cloud_density > 0.3:
            glow_radius = w * 0.04  # 4% of image width
            glow = np.exp(-(dist ** 2) / (2.0 * (glow_radius ** 2)))

            # Warm yellow-white glow
            glow_intensity = moon_cloud_density * 60
            glow_r = (glow * glow_intensity).astype(np.uint8)
            glow_g = (glow * glow_intensity * 0.9).astype(np.uint8)
            glow_b = (glow * glow_intensity * 0.7).astype(np.uint8)

            # Composite glow into cloud layer
            glow_alpha = (glow * moon_cloud_density * 0.5 * 255).astype(np.uint8)
            for c in range(3):
                cloud_layer[:, :, c] = np.maximum(cloud_layer[:, :, c],
                                                  [glow_r, glow_g, glow_b][c])

    # Composite cloud layer over base
    cloud_pil = Image.fromarray(cloud_layer, mode="RGBA")
    result = Image.alpha_composite(base, cloud_pil)
    return result


# ---------------------------------------------------------------------------
#  Haze
# ---------------------------------------------------------------------------

def render_haze(image: Image.Image, visibility_km: float) -> Image.Image:
    """Overlay a horizontal haze gradient near the horizon.

    Haze reduces contrast near the horizon, creating a soft gradient
    that blends the sky colour.  The effect is strongest at the bottom
    of the image and fades upward.

    Args:
        image: Input PIL ``Image`` (RGB or RGBA).
        visibility_km: Visibility range in kilometres (lower = more haze).

    Returns:
        A new PIL ``Image`` with haze applied.
    """
    visibility_km = max(0.1, visibility_km)
    base = image.convert("RGBA")
    w, h = base.size

    # Haze intensity: inverse of visibility
    # 0.1 km = very hazy, 50+ km = almost clear
    haze_strength = max(0.0, min(0.7, 1.0 - visibility_km / 15.0))

    if haze_strength < 0.01:
        return base

    pixels = np.array(base, dtype=np.float32)

    # Create a vertical gradient: strong near bottom, fading upward
    gradient = np.linspace(1.0, 0.0, h, dtype=np.float32)  # bottom=1, top=0
    gradient = gradient ** 1.5  # non-linear fade

    # Haze colour: light grey/blue (scattered light)
    haze_colour = np.array([200, 210, 220], dtype=np.float32)

    # Blend: result = pixel * (1 - alpha) + haze_colour * alpha
    alpha = gradient * haze_strength
    for c in range(3):
        pixels[:, :, c] = (pixels[:, :, c] * (1.0 - alpha[:, np.newaxis])
                           + haze_colour[c] * alpha[:, np.newaxis])

    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGBA")


# ---------------------------------------------------------------------------
#  Fog
# ---------------------------------------------------------------------------

def render_fog(image: Image.Image, humidity_pct: float) -> Image.Image:
    """Optionally overlay a uniform fog effect based on humidity.

    Fog is rendered as a uniform grey/white semi-transparent overlay.
    The effect becomes noticeable above ~80% humidity and strongest
    near 100%.

    This is an optional effect — it has no effect below 80% humidity.

    Args:
        image: Input PIL ``Image`` (RGB or RGBA).
        humidity_pct: Relative humidity percentage (0–100).

    Returns:
        A new PIL ``Image`` with fog applied, or the original if
        humidity < 80%.
    """
    humidity_pct = max(0.0, min(100.0, humidity_pct))
    if humidity_pct < 80.0:
        return image.convert("RGBA") if image.mode != "RGBA" else image

    base = image.convert("RGBA")
    w, h = base.size

    # Fog strength: 0 at 80%, ~0.35 at 100%
    fog_strength = (humidity_pct - 80.0) / 20.0 * 0.35
    fog_strength = min(0.35, fog_strength)

    # Fog is slightly denser near the ground
    gradient = np.linspace(1.0, 0.4, h, dtype=np.float32)
    gradient = gradient ** 0.8

    pixels = np.array(base, dtype=np.float32)
    fog_colour = np.array([190, 195, 200], dtype=np.float32)

    alpha = gradient * fog_strength
    for c in range(3):
        pixels[:, :, c] = (pixels[:, :, c] * (1.0 - alpha[:, np.newaxis])
                           + fog_colour[c] * alpha[:, np.newaxis])

    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGBA")
