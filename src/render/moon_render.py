"""
Moon disk rendering for Moonshot.

Renders the Moon as a circular disk with correct illumination,
terminator angle, size, and atmospheric colour tint.

All angles are in degrees unless explicitly stated otherwise.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

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


def render_moon_disk_with_texture(
    illumination: float,
    terminator_angle_deg: float,
    tint_rgb: Tuple[float, float, float],
    pixel_radius: int,
    texture: Optional[np.ndarray],
    parallactic_angle_deg: float = 0.0,
) -> Image.Image:
    """Render the Moon disk with real surface texture and correct phase.

    Uses an equirectangular lunar surface texture mapped via orthographic
    projection, rotated by *parallactic_angle_deg* so that surface
    features appear correctly oriented for the observer's latitude.
    Lighting applies the atmospheric *tint* on the lit side
    and a subtle blue-green earthshine on the dark side, with a smooth
    terminator blend and limb darkening.

    Falls back to the flat-colour ``render_moon_disk()`` if *texture* is
    ``None`` or cannot be loaded (warning issued once).

    Args:
        illumination: Illuminated fraction (0.0 = new, 1.0 = full).
        terminator_angle_deg: Position angle of the bright limb (the
            terminator) in degrees, measured eastward from celestial north.
        tint_rgb: RGB colour multipliers from atmospheric scattering,
            each in range [0.0, 1.0].
        pixel_radius: Moon disk radius in pixels.
        texture: ``(H, W, 3)`` uint8 equirectangular texture array, or
            ``None`` to fall back to the standard flat-colour disk.
        parallactic_angle_deg: Parallactic angle in degrees.  The moon
            texture is rotated by this angle so that surface features
            appear correctly oriented for the observer's latitude.
            Default 0.0 (northern hemisphere / equator).

    Returns:
        A ``(2 * pixel_radius, 2 * pixel_radius)`` RGBA image with
        the moon disk centred in the frame.
    """
    if texture is None:
        if not getattr(render_moon_disk_with_texture, "_fallback_warned", False):
            import warnings
            warnings.warn(
                "No moon texture provided — falling back to flat-colour disk."
            )
            render_moon_disk_with_texture._fallback_warned = True
        return render_moon_disk(illumination, terminator_angle_deg,
                                tint_rgb, pixel_radius)

    if pixel_radius < 1:
        raise ValueError(f"Pixel radius must be positive, got {pixel_radius}")
    illumination = max(0.0, min(1.0, illumination))

    diameter = pixel_radius * 2
    centre = (pixel_radius, pixel_radius)

    # --- Coordinate grids (full 2D, not broadcast) ---
    y_grid, x_grid = np.mgrid[:diameter, :diameter]
    dx = x_grid - centre[0]
    dy = y_grid - centre[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    inside = dist <= pixel_radius

    # Normalised pixel offsets in [-1, 1]
    px = dx.astype(np.float64) / pixel_radius
    py = dy.astype(np.float64) / pixel_radius

    # --- Texture rotation for observer latitude ---
    # Rotate the (px, py) coordinates by the parallactic angle so that
    # the moon's surface features appear correctly oriented for the
    # observer's hemisphere (southern hemisphere → upside-down relative
    # to northern).
    if parallactic_angle_deg != 0.0:
        q = math.radians(parallactic_angle_deg)
        cos_q = math.cos(q)
        sin_q = math.sin(q)
        px_tex = px * cos_q - py * sin_q
        py_tex = px * sin_q + py * cos_q
    else:
        px_tex = px
        py_tex = py

    # --- Orthographic projection ---
    # Compute latitude and longitude for each pixel
    # lat = arcsin(py_tex)
    # lon = atan2(px_tex, sqrt(1 - px_tex² - py_tex²))
    lat = np.arcsin(np.clip(py_tex, -1.0, 1.0))
    sq = px_tex ** 2 + py_tex ** 2
    sqrt_term = np.sqrt(np.clip(1.0 - sq, 0.0, 1.0))
    lon = np.arctan2(px_tex, sqrt_term)

    # --- Phase (terminator) ---
    theta = math.radians(terminator_angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x_prime = dx * cos_t + dy * sin_t

    terminator_offset = pixel_radius * (1.0 - 2.0 * illumination)
    lit = inside & (x_prime >= terminator_offset)
    dark = inside & ~lit

    # --- Sample texture for each pixel ---
    # We only sample inside the disk
    rgba = np.zeros((diameter, diameter, 4), dtype=np.uint8)

    tint_r, tint_g, tint_b = tint_rgb

    # Texture sample for lit side
    for idx in np.argwhere(lit):
        y_i, x_i = int(idx[0]), int(idx[1])
        r, g, b = _sample_moon_texture(
            float(lon[y_i, x_i]), float(lat[y_i, x_i]), texture
        )
        # Apply tint
        r = int(round(r * tint_r))
        g = int(round(g * tint_g))
        b = int(round(b * tint_b))
        rgba[y_i, x_i, 0] = np.clip(r, 0, 255)
        rgba[y_i, x_i, 1] = np.clip(g, 0, 255)
        rgba[y_i, x_i, 2] = np.clip(b, 0, 255)
        rgba[y_i, x_i, 3] = 255

    # Earthshine for dark side
    earthshine_factor = 0.015 * (1.0 - illumination)
    # Blue-green tint for earthshine
    es_r, es_g, es_b = 0.7, 0.8, 1.0
    for idx in np.argwhere(dark):
        y_i, x_i = int(idx[0]), int(idx[1])
        r, g, b = _sample_moon_texture(
            float(lon[y_i, x_i]), float(lat[y_i, x_i]), texture
        )
        r = int(round(r * earthshine_factor * es_r))
        g = int(round(g * earthshine_factor * es_g))
        b = int(round(b * earthshine_factor * es_b))
        rgba[y_i, x_i, 0] = np.clip(r, 0, 255)
        rgba[y_i, x_i, 1] = np.clip(g, 0, 255)
        rgba[y_i, x_i, 2] = np.clip(b, 0, 255)
        rgba[y_i, x_i, 3] = 255

    # --- Terminator smooth blend (4px band) ---
    # For pixels near the terminator, blend between lit and dark
    terminator_width_px = 4.0
    for idx in np.argwhere(inside):
        y_i, x_i = int(idx[0]), int(idx[1])
        xp = x_prime[y_i, x_i]
        offset = terminator_offset
        # Distance from terminator (negative = dark side, positive = lit)
        distance_from_term = xp - offset

        if -terminator_width_px <= distance_from_term <= 0:
            # Dark side blend zone — blend toward lit side
            blend = (distance_from_term + terminator_width_px) / terminator_width_px
            # Sample lit colour at this pixel
            r_lit, g_lit, b_lit = _sample_moon_texture(
                float(lon[y_i, x_i]), float(lat[y_i, x_i]), texture
            )
            r_lit = int(round(r_lit * tint_r))
            g_lit = int(round(g_lit * tint_g))
            b_lit = int(round(b_lit * tint_b))

            r_dark, g_dark, b_dark = _sample_moon_texture(
                float(lon[y_i, x_i]), float(lat[y_i, x_i]), texture
            )
            r_dark = int(round(r_dark * earthshine_factor * es_r))
            g_dark = int(round(g_dark * earthshine_factor * es_g))
            b_dark = int(round(b_dark * earthshine_factor * es_b))

            rgba[y_i, x_i, 0] = np.clip(
                int(round(r_dark + (r_lit - r_dark) * blend)), 0, 255
            )
            rgba[y_i, x_i, 1] = np.clip(
                int(round(g_dark + (g_lit - g_dark) * blend)), 0, 255
            )
            rgba[y_i, x_i, 2] = np.clip(
                int(round(b_dark + (b_lit - b_dark) * blend)), 0, 255
            )
            rgba[y_i, x_i, 3] = 255

    # --- Limb darkening on lit side ---
    if illumination > 0.05:
        edge_factor = dist / pixel_radius
        limb_darken = 1.0 - 0.15 * (edge_factor ** 2)
        lit_mask_np = lit.astype(np.float64)
        rgba[..., 0] = (
            rgba[..., 0].astype(np.float32) * (1.0 - lit_mask_np + lit_mask_np * limb_darken)
        ).astype(np.uint8)
        rgba[..., 1] = (
            rgba[..., 1].astype(np.float32) * (1.0 - lit_mask_np + lit_mask_np * limb_darken)
        ).astype(np.uint8)
        rgba[..., 2] = (
            rgba[..., 2].astype(np.float32) * (1.0 - lit_mask_np + lit_mask_np * limb_darken)
        ).astype(np.uint8)

    # --- Anti-aliased edge ---
    edge_band = (dist > pixel_radius - 1.0) & (dist <= pixel_radius)
    alpha_smooth = np.clip(pixel_radius - dist, 0.0, 1.0)
    rgba[edge_band, 3] = (alpha_smooth[edge_band] * 255).astype(np.uint8)

    return Image.fromarray(rgba, mode="RGBA")


def _sample_moon_texture(
    lon_rad: float,
    lat_rad: float,
    texture: np.ndarray,
) -> Tuple[int, int, int]:
    """Inline bilinear sample of an equirectangular texture.

    Replicates ``moon_texture.sample_texture()`` to avoid circular
    dependency concerns and keep the render module self-contained.

    Args:
        lon_rad: Longitude in radians in [-π, π].
        lat_rad: Latitude in radians in [-π/2, π/2].
        texture: ``(H, W, 3)`` uint8 equirectangular texture.

    Returns:
        ``(r, g, b)`` tuple of integers in [0, 255].
    """
    h, w = texture.shape[:2]
    u = (lon_rad + np.pi) / (2.0 * np.pi)
    v = (np.pi / 2.0 - lat_rad) / np.pi

    u = u % 1.0
    v = min(max(v, 0.0), 1.0)

    x = u * (w - 1)
    y = v * (h - 1)

    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, h - 1)

    fx = x - x0
    fy = y - y0

    c00 = texture[y0, x0].astype(np.float32)
    c10 = texture[y0, x1].astype(np.float32)
    c01 = texture[y1, x0].astype(np.float32)
    c11 = texture[y1, x1].astype(np.float32)

    c0 = c00 + fx * (c10 - c00)
    c1 = c01 + fx * (c11 - c01)
    result = c0 + fy * (c1 - c0)

    return (int(round(result[0])), int(round(result[1])), int(round(result[2])))


def moon_position_on_image(altitude_deg: float,
                           azimuth_deg: float,
                           fov_deg: float,
                           image_width: int,
                           image_height: int,
                           center_on_moon: bool = False) -> Tuple[int, int]:
    """Compute the pixel position of the moon centre on the image.

    Coordinate mapping:

    * Altitude 0°  →  horizon (bottom of image), unless *center_on_moon*
      is True, in which case the moon is placed at the vertical centre
      and the FOV determines the sky visible above and below it.
    * Altitude 90° → zenith (top-centre of image, constrained by FOV).
    * Azimuth is mapped horizontally within the field of view.

    The azimuth mapping is centred so that the middle of the FOV
    corresponds to the direction the viewer is facing.  By default this
    function assumes the viewer faces the moon's azimuth direction
    (i.e., the moon is horizontally centred).

    Args:
        altitude_deg: Apparent altitude of the moon in degrees (0 = horizon).
        azimuth_deg: Apparent azimuth of the moon in degrees from north
                     (0–360).
        fov_deg: Field of view in degrees.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        center_on_moon: When True, the moon is placed at the vertical centre
                        of the image and the FOV determines the visible
                        sky above/below it.  Use this for telephoto-style
                        close-up moon shots.

    Returns:
        ``(x, y)`` pixel coordinates for the moon centre.
    """
    if fov_deg <= 0.0:
        raise ValueError(f"FOV must be positive, got {fov_deg}")
    if image_width < 1 or image_height < 1:
        raise ValueError(f"Image dimensions must be positive, got ({image_width}, {image_height})")

    # --- Vertical placement ---
    if center_on_moon:
        # Place moon at vertical centre; sky above/below within FOV
        y = image_height // 2
    else:
        # Map altitude linearly: 0° → bottom, 90° → top
        alt_fraction = altitude_deg / fov_deg
        y = int(round(image_height * (1.0 - alt_fraction)))
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
