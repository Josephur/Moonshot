"""
Sky gradient rendering for Moonshot.

Generates a full-canvas sky background image with realistic colour
gradients based on the Sun's altitude (day / twilight / night) and a
subtle atmospheric glow near the Moon's position.

All altitude values are in degrees.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# Daytime sky: zenith colour at various sun altitudes, blended toward
# a white/light-blue horizon.
_DAY_ZENITH_RGB: dict[str, Tuple[int, int, int]] = {
    "high": (30, 100, 210),     # sun > 30° — deep blue
    "mid":  (60, 140, 220),     # sun 10-30°
    "low":  (120, 170, 230),    # sun 0-10° — pale blue
}
_DAY_HORIZON_RGB = (220, 230, 245)  # whitish-blue near horizon

# Twilight gradient (sun between -18° and 0°).
# Horizon band is orange/red/purple; upper sky is deep blue.
_TWILIGHT_HORIZON_RGB: list[Tuple[int, int, int]] = [
    (255, 150,  50),   # orange (sun just below horizon)
    (200,  80, 120),   # purple-red
    (100,  50, 150),   # deep purple
]
_TWILIGHT_ZENITH_RGB = (10, 10, 40)  # very dark blue

# Night sky (sun < -18°).
_NIGHT_SKY_RGB = (5, 5, 20)          # deep blue-black
_NIGHT_HORIZON_RGB = (15, 15, 30)    # slightly lighter near horizon


def _interpolate_color(
    t: float,
    a: Tuple[int, int, int],
    b: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Linearly interpolate between two RGB tuples.

    ``t`` is clamped to [0, 1].
    """
    t = max(0.0, min(1.0, t))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def _daytime_gradient(width: int, height: int,
                      sun_alt: float) -> Image.Image:
    """Build a daytime sky gradient.

    Uses a vertical gradient from zenith colour to horizon colour,
    with the zenith colour chosen by sun altitude.
    """
    img = Image.new("RGB", (width, height))
    pixels = np.array(img, dtype=np.uint8)

    # Pick zenith colour based on sun altitude
    if sun_alt >= 30.0:
        zenith = _DAY_ZENITH_RGB["high"]
    elif sun_alt >= 10.0:
        w = (sun_alt - 10.0) / 20.0
        zenith = _interpolate_color(w, _DAY_ZENITH_RGB["mid"], _DAY_ZENITH_RGB["high"])
    else:
        w = (sun_alt - 0.0) / 10.0
        zenith = _interpolate_color(w, _DAY_ZENITH_RGB["low"], _DAY_ZENITH_RGB["mid"])

    horizon = _DAY_HORIZON_RGB

    for y in range(height):
        # t = 0 at top (zenith), 1 at bottom (horizon)
        t = y / (height - 1) if height > 1 else 0.0
        # Non-linear: most of the colour shift is near the horizon
        t_nl = t ** 0.7
        r, g, b = _interpolate_color(t_nl, zenith, horizon)
        pixels[y, :, 0] = r
        pixels[y, :, 1] = g
        pixels[y, :, 2] = b

    return Image.fromarray(pixels, mode="RGB")


def _twilight_gradient(width: int, height: int,
                       sun_alt: float) -> Image.Image:
    """Build a twilight sky gradient.

    The horizon band is brightly coloured (orange → purple), fading to
    dark blue at the zenith.
    """
    img = Image.new("RGB", (width, height))
    pixels = np.array(img, dtype=np.uint8)

    # How deep into twilight?  0 = sun just at horizon, 1 = end of twilight
    twilight_depth = max(0.0, min(1.0, (-sun_alt) / 18.0))

    # Pick a horizon colour from the palette
    if twilight_depth < 0.5:
        w = twilight_depth * 2.0
        horizon_col = _interpolate_color(w,
                                         _TWILIGHT_HORIZON_RGB[0],
                                         _TWILIGHT_HORIZON_RGB[1])
    else:
        w = (twilight_depth - 0.5) * 2.0
        horizon_col = _interpolate_color(w,
                                         _TWILIGHT_HORIZON_RGB[1],
                                         _TWILIGHT_HORIZON_RGB[2])

    zenith = _TWILIGHT_ZENITH_RGB

    for y in range(height):
        t = y / (height - 1) if height > 1 else 1.0
        # Steep gradient — horizon colour only in bottom ~30%
        if t > 0.7:
            # Blend toward horizon colour
            blend = (t - 0.7) / 0.3
            col = _interpolate_color(blend, zenith, horizon_col)
        else:
            col = zenith
        pixels[y, :, 0] = col[0]
        pixels[y, :, 1] = col[1]
        pixels[y, :, 2] = col[2]

    return Image.fromarray(pixels, mode="RGB")


def _night_gradient(width: int, height: int) -> Image.Image:
    """Build a night sky gradient.

    Deep blue-black at the top, slightly lighter at the horizon.
    """
    img = Image.new("RGB", (width, height))
    pixels = np.array(img, dtype=np.uint8)

    for y in range(height):
        t = y / (height - 1) if height > 1 else 0.0
        col = _interpolate_color(t ** 1.5, _NIGHT_SKY_RGB, _NIGHT_HORIZON_RGB)
        pixels[y, :, 0] = col[0]
        pixels[y, :, 1] = col[1]
        pixels[y, :, 2] = col[2]

    return Image.fromarray(pixels, mode="RGB")


def _add_stars(image: Image.Image,
               lat: float = None,
               lon: float = None,
               jd: float = None,
               fov_deg: float = None) -> Image.Image:
    """Add stars to a night/twilight sky.

    When ``lat``, ``lon``, ``jd``, and ``fov_deg`` are all provided,
    renders real stars from the HYG catalog using the full coordinate
    pipeline.  Otherwise falls back to random white dots (legacy
    behaviour).
    """
    # Real star rendering when all geolocation params are present
    if all(v is not None for v in (lat, lon, jd, fov_deg)):
        from render.stars import render_stars_to_sky
        return render_stars_to_sky(image, lat, lon, jd, fov_deg)

    # Legacy fallback: random white dots
    w, h = image.size
    pixels = np.array(image, dtype=np.float32)

    rng = np.random.RandomState(seed=42 + w * h)
    n_stars = int(w * h * 0.0008)  # ~0.08% of pixels are stars

    xs = rng.randint(0, w, size=n_stars)
    ys = rng.randint(0, h, size=n_stars)
    brightnesses = rng.uniform(0.3, 1.0, size=n_stars)

    for x, y, br in zip(xs, ys, brightnesses):
        if y > h * 0.88:
            continue
        intensity = int(round(br * 255))
        pixels[y, x, 0] = min(pixels[y, x, 0] + intensity * 0.5, 255.0)
        pixels[y, x, 1] = min(pixels[y, x, 1] + intensity * 0.5, 255.0)
        pixels[y, x, 2] = min(pixels[y, x, 2] + intensity * 0.5, 255.0)

    return Image.fromarray(pixels.astype(np.uint8), mode="RGB")


def _add_moon_glow(image: Image.Image,
                   moon_altitude_deg: float,
                   moon_x: int,
                   moon_y: int) -> Image.Image:
    """Add a subtle atmospheric glow near the Moon's position.

    The glow is a radial gradient, brightest at the moon position and
    fading outward.  The radius and intensity depend on the moon's
    altitude (low-altitude moon has a stronger, more spread-out glow
    due to greater atmospheric backscatter).
    """
    if moon_altitude_deg <= 0.0:
        return image  # Moon is below the horizon — no glow

    w, h = image.size
    pixels = np.array(image, dtype=np.float32)

    # Glow radius grows when the moon is lower (more atmosphere)
    glow_radius = int(120 + (1.0 - moon_altitude_deg / 90.0) * 200)
    glow_radius = min(glow_radius, w // 2, h // 2)

    # Create a vertical-heatmap-style glow: stronger near the bottom
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - moon_x) ** 2 + (y_grid - moon_y) ** 2)

    # Intensity follows a Gaussian falloff
    sigma = glow_radius / 2.5
    glow = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))

    # Determine glow colour based on sky type (pale for night, warm for twilight)
    mean_brightness = pixels.mean()
    if mean_brightness < 30:
        # Night: cool white/blue glow
        glow_colour = (100, 130, 200)
    elif mean_brightness < 100:
        # Twilight: warm pale glow
        glow_colour = (180, 150, 160)
    else:
        # Daytime: very faint, almost invisible
        glow_colour = (200, 210, 230)

    max_glow_intensity = 0.08 if mean_brightness > 100 else 0.20

    for c in range(3):
        pixels[:, :, c] += glow * max_glow_intensity * glow_colour[c]

    # Clamp
    pixels = np.clip(pixels, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8), mode="RGB")


def sky_gradient(sun_altitude_deg: float,
                 moon_altitude_deg: float,
                 width: int,
                 height: int,
                 lat: Optional[float] = None,
                 lon: Optional[float] = None,
                 jd: Optional[float] = None,
                 fov_deg: Optional[float] = None,
                 moon_px: Optional[int] = None,
                 moon_py: Optional[int] = None) -> Image.Image:
    """Render a sky background image based on the Sun's altitude.

    Three regimes:

    * **Daytime** (sun > 0°):  blue gradient, whiter near the horizon.
    * **Twilight** (sun between -18° and 0°): orange/purple horizon
      band fading to dark blue above.
    * **Night** (sun < -18°): deep blue-black with scattered stars.

    If the moon is above the horizon a subtle atmospheric backscatter
    glow is added near its position.

    When ``lat``, ``lon``, ``jd``, and ``fov_deg`` are all provided,
    real star positions are rendered from the HYG catalog instead of
    random white dots.

    Args:
        sun_altitude_deg: Sun altitude in degrees (above = positive).
        moon_altitude_deg: Moon altitude in degrees.
        width: Image width in pixels.
        height: Image height in pixels.
        lat: Observer latitude (for real star rendering).
        lon: Observer longitude (for real star rendering).
        jd: Julian Date of observation (for real star rendering).
        fov_deg: Vertical field-of-view in degrees (for real star rendering).
        moon_px: Moon pixel X position (for glow placement).  If omitted
            the position is estimated from altitude.
        moon_py: Moon pixel Y position (for glow placement).  If omitted
            the position is estimated from altitude.

    Returns:
        A new PIL ``Image`` in RGB mode of size ``(width, height)``.
    """
    if width < 1 or height < 1:
        raise ValueError(f"Image dimensions must be positive, got ({width}, {height})")

    # Choose and render the base gradient
    if sun_altitude_deg > 0.0:
        sky = _daytime_gradient(width, height, sun_altitude_deg)
    elif sun_altitude_deg > -18.0:
        sky = _twilight_gradient(width, height, sun_altitude_deg)
        sky = _add_stars(sky, lat=lat, lon=lon, jd=jd, fov_deg=fov_deg)
    else:
        sky = _night_gradient(width, height)
        sky = _add_stars(sky, lat=lat, lon=lon, jd=jd, fov_deg=fov_deg)

    # Add moon glow if the moon is above the horizon
    if moon_altitude_deg > 0.0:
        if moon_px is not None and moon_py is not None:
            # Use actual moon pixel position (from moon_position_on_image)
            sky = _add_moon_glow(sky, moon_altitude_deg, moon_px, moon_py)
        else:
            # Fallback estimate — approximate position from altitude alone
            moon_x = width // 2
            moon_y = int(height * (1.0 - (moon_altitude_deg / 90.0) * 0.85))
            sky = _add_moon_glow(sky, moon_altitude_deg, moon_x, moon_y)

    return sky
