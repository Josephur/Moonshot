"""
Moon disk rendering for Moonshot.

Renders the Moon as a circular disk with correct illumination,
terminator angle, size, and atmospheric colour tint.

All angles are in degrees unless explicitly stated otherwise.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw


def moon_size_pixels(angular_diameter_deg: float,
                     fov_deg: float,
                     image_width: int) -> int:
    """Compute the moon's diameter in pixels for a given FOV and image width.

    Args:
        angular_diameter_deg: Moon's apparent angular diameter in degrees
                              (typically 0.49°–0.56°).
        fov_deg: Camera/viewer field of view in degrees.
        image_width: Image width in pixels.

    Returns:
        Moon diameter in pixels (always at least 1 pixel).
    """
    if fov_deg <= 0.0:
        raise ValueError(f"FOV must be positive, got {fov_deg}")
    if image_width < 1:
        raise ValueError(f"Image width must be positive, got {image_width}")

    # Simple proportion: moon angular size / FOV = moon pixels / image width
    moon_px = int(round((angular_diameter_deg / fov_deg) * image_width))
    return max(moon_px, 1)


def render_moon_disk(illumination: float,
                     terminator_angle_deg: float,
                     tint_rgb: Tuple[float, float, float],
                     pixel_radius: int) -> Image.Image:
    """Render the Moon disk with correct phase and colour tint.

    The returned image has an alpha (transparency) channel so it can be
    composited onto any background.

    Args:
        illumination: Illuminated fraction (0.0 = new, 1.0 = full).
        terminator_angle_deg: Position angle of the bright limb (the
            terminator) in degrees, measured eastward from celestial
            north.
        tint_rgb: RGB colour multipliers from atmospheric scattering,
            each in range [0.0, 1.0].
        pixel_radius: Moon disk radius in pixels.

    Returns:
        A ``(2 * pixel_radius, 2 * pixel_radius)`` RGBA image with
        the moon disk centred in the frame.
    """
    if pixel_radius < 1:
        raise ValueError(f"Pixel radius must be positive, got {pixel_radius}")
    illumination = max(0.0, min(1.0, illumination))

    diameter = pixel_radius * 2
    centre = (pixel_radius, pixel_radius)

    # Create an RGBA image (all transparent initially)
    moon_img = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(moon_img)

    # --- Compute pixel mask for the disk ---
    # We use numpy for precise per-pixel illumination
    y_grid, x_grid = np.ogrid[:diameter, :diameter]
    dx = x_grid - centre[0]
    dy = y_grid - centre[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # Binary mask of pixels inside the disk
    inside = dist <= pixel_radius

    # --- Phase (terminator) ---
    # The terminator is the line separating light from dark.
    # We rotate the coordinate system by terminator_angle_deg so that
    # the terminator is a vertical line (x'_pixel = 0).
    #   x' = dx * cos(θ) + dy * sin(θ)
    # The illuminated portion faces the +x' direction.
    theta = math.radians(terminator_angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x_prime = dx * cos_t + dy * sin_t

    # Illumination fraction determines the terminator offset:
    #   illumination = 0.5  →  terminator at x' = 0
    #   illumination = 1.0  →  terminator at x' = -radius (fully lit)
    #   illumination = 0.0  →  terminator at x' = +radius (fully dark)
    terminator_offset = pixel_radius * (1.0 - 2.0 * illumination)

    # Lit region: x' >= terminator_offset
    lit = inside & (x_prime >= terminator_offset)

    # --- Base colours ---
    # Sunlit side: white/yellowish
    # Dark side: very dark gray (not pure black — retains some detail)
    lit_r = 255
    lit_g = 250
    lit_b = 230

    dark_r, dark_g, dark_b = 25, 25, 30

    # --- Apply atmospheric tint ---
    # The tint darkens/reddens the lit side
    tint_r, tint_g, tint_b = tint_rgb
    lit_r = int(round(lit_r * tint_r))
    lit_g = int(round(lit_g * tint_g))
    lit_b = int(round(lit_b * tint_b))

    # --- Build the RGBA pixel array ---
    rgba = np.zeros((diameter, diameter, 4), dtype=np.uint8)

    # Lit pixels
    rgba[lit, 0] = np.clip(lit_r, 0, 255)
    rgba[lit, 1] = np.clip(lit_g, 0, 255)
    rgba[lit, 2] = np.clip(lit_b, 0, 255)
    rgba[lit, 3] = 255  # fully opaque

    # Dark-side pixels inside the disk
    dark = inside & ~lit
    rgba[dark, 0] = dark_r
    rgba[dark, 1] = dark_g
    rgba[dark, 2] = dark_b
    rgba[dark, 3] = 255

    # --- Subtle limb-darkening on the lit side ---
    # Pixels near the edge of the disk get slightly dimmer
    if illumination > 0.05:
        # Normalised distance from terminator (brightest ~25% from terminator)
        edge_factor = dist / pixel_radius
        limb_darken = 1.0 - 0.15 * (edge_factor ** 2)
        # Only apply to lit pixels
        rgba[lit, 0] = (rgba[lit, 0].astype(np.float32) * limb_darken[lit]).astype(np.uint8)
        rgba[lit, 1] = (rgba[lit, 1].astype(np.float32) * limb_darken[lit]).astype(np.uint8)
        rgba[lit, 2] = (rgba[lit, 2].astype(np.float32) * limb_darken[lit]).astype(np.uint8)

    # --- Anti-aliased edge (smooth the alpha at the boundary) ---
    # Use a 1-pixel smooth falloff for pixels near radius
    edge_band = (dist > pixel_radius - 1.0) & (dist <= pixel_radius)
    alpha_smooth = np.clip(pixel_radius - dist, 0.0, 1.0)
    rgba[edge_band, 3] = (alpha_smooth[edge_band] * 255).astype(np.uint8)

    return Image.fromarray(rgba, mode="RGBA")


def moon_position_on_image(altitude_deg: float,
                           azimuth_deg: float,
                           fov_deg: float,
                           image_width: int,
                           image_height: int) -> Tuple[int, int]:
    """Compute the pixel position of the moon centre on the image.

    Coordinate mapping:

    * Altitude 0°  →  horizon (bottom of image)
    * Altitude 90° → zenith (top-centre of image, constrained by FOV)
    * Azimuth is mapped horizontally within the field of view.

    The azimuth mapping is centred so that the middle of the FOV
    corresponds to the direction the viewer is facing.  By default this
    function assumes the viewer faces the moon's azimuth direction
    (i.e., the moon is horizontally centred).  Pure horizontal
    placement is computed from a "viewer direction" at the image centre.

    Args:
        altitude_deg: Apparent altitude of the moon in degrees (0 = horizon).
        azimuth_deg: Apparent azimuth of the moon in degrees from north
                     (0–360).
        fov_deg: Field of view in degrees.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        ``(x, y)`` pixel coordinates for the moon centre.
    """
    if fov_deg <= 0.0:
        raise ValueError(f"FOV must be positive, got {fov_deg}")
    if image_width < 1 or image_height < 1:
        raise ValueError(f"Image dimensions must be positive, got ({image_width}, {image_height})")

    # --- Vertical placement ---
    # Map altitude linearly: 0° → bottom, 90° → top (but limited by FOV).
    # The FOV spans from horizon (bottom) upward.
    # full_height_deg = fov_deg (the sky portion shown)
    # altitude_deg / fov_deg gives the fraction from bottom
    alt_fraction = altitude_deg / fov_deg
    y = int(round(image_height * (1.0 - alt_fraction)))
    # Clamp to image bounds
    y = max(0, min(image_height - 1, y))

    # --- Horizontal placement ---
    # By default place the moon in the centre horizontally.
    # If azimuth is available we can use it to offset horizontally
    # within the FOV.  For simplicity we centre on the image centre.
    # A more sophisticated approach would use a viewer-facing direction
    # and offset within the FOV using azimuth difference.

    # Place at centre by default (the viewer is looking at the moon)
    x = image_width // 2

    return x, y


def moon_position_on_image_with_direction(
    altitude_deg: float,
    azimuth_deg: float,
    viewer_azimuth_deg: float,
    fov_deg: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int]:
    """Compute the pixel position of the moon with an explicit viewer direction.

    The ``viewer_azimuth_deg`` is the direction the viewer is facing.
    The moon's azimuth is offset relative to this, and the offset is
    mapped horizontally within the FOV.

    Args:
        altitude_deg: Apparent altitude of the moon in degrees.
        azimuth_deg: Apparent azimuth of the moon in degrees.
        viewer_azimuth_deg: Viewer facing direction in degrees.
        fov_deg: Field of view in degrees.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        ``(x, y)`` pixel coordinates for the moon centre.
    """
    if fov_deg <= 0.0:
        raise ValueError(f"FOV must be positive, got {fov_deg}")

    # --- Vertical placement (same as above) ---
    alt_fraction = altitude_deg / fov_deg
    y = int(round(image_height * (1.0 - alt_fraction)))
    y = max(0, min(image_height - 1, y))

    # --- Horizontal offset from viewer direction ---
    az_diff = (azimuth_deg - viewer_azimuth_deg) % 360.0
    if az_diff > 180.0:
        az_diff -= 360.0  # bring to [-180, 180]

    # Map azimuth offset to horizontal pixel offset
    # Centre of image = viewer direction
    # Edges of image = ±fov_deg/2
    half_fov = fov_deg / 2.0
    az_fraction = az_diff / half_fov  # -1 to +1
    x = int(round(
        (image_width / 2.0) + az_fraction * (image_width / 2.0)
    ))
    x = max(0, min(image_width - 1, x))

    return x, y
