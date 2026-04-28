"""
Image assembly / orchestrator for Moonshot.

``generate_moon_image()`` is the main entry point that coordinates all
render layers — sky gradient, moon disk, horizon, weather overlays,
and annotations — into a single composited PIL image.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Tuple

from PIL import Image

# Ensure the project src directory is on the path so we can import
# sibling packages (moon, atmosphere, etc.) regardless of how this
# module is invoked.
_PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_SRC not in sys.path:
    sys.path.insert(0, _PROJECT_SRC)

from moon import position as moonpos
from moon import phase as moonphase
from moon import timeconv as mtime
from atmosphere import scattering as atmo_scatter
from atmosphere import refraction as atmo_refract

from render.sky import sky_gradient
from render.moon_render import (
    moon_size_pixels,
    render_moon_disk,
    moon_position_on_image,
)
from render.horizon import horizon_line, horizon_dip_degrees
from render.weather_overlay import render_clouds, render_haze, render_fog
from render.annotations import (
    annotate_image,
    LocationData,
    MoonData,
    WeatherAnnotationData,
    TimeData,
)

from weather.provider import WeatherData


def generate_moon_image(
    lat: float,
    lon: float,
    city: str = "",
    state: str = "",
    timezone_str: str = "",
    dt: Optional[datetime] = None,
    fov_deg: float = 90.0,
    image_w: int = 1920,
    image_h: int = 1080,
    weather_data: Optional[WeatherData] = None,
    api_key: str = "",
    temp_c: float = 10.0,
    pressure_mbar: float = 1013.0,
    humidity_pct: float = 50.0,
    observer_height_m: float = 250.0,
) -> Image.Image:
    """Render a complete moon-view image from the given parameters.

    Orchestrates all render layers in order:
        1. **Sky gradient** — sky background based on sun altitude
        2. **Moon disk** — phase- and tint-correct moon at correct position
        3. **Horizon** — terrain silhouette
        4. **Weather overlays** — clouds, haze, fog
        5. **Annotations** — text data overlay

    Args:
        lat: Observer latitude in decimal degrees.
        lon: Observer longitude in decimal degrees.
        city: City name (for annotations).
        state: State name (for annotations).
        timezone_str: IANA timezone name (e.g. "America/New_York").
        dt: Timezone-aware observation datetime.  If None, uses current UTC.
        fov_deg: Field of view in degrees.
        image_w: Output image width in pixels.
        image_h: Output image height in pixels.
        weather_data: ``WeatherData`` instance from the weather provider,
                      or None to use defaults.
        api_key: OpenWeatherMap API key (used if weather_data is None
                 and a fetch is needed).
        temp_c: Temperature in °C (fallback).
        pressure_mbar: Pressure in mbar (fallback).
        humidity_pct: Relative humidity percent (fallback).
        observer_height_m: Observer elevation in metres (for horizon dip).

    Returns:
        A PIL ``Image`` in RGBA mode with the complete scene rendered.
    """
    # ---- 1. Resolve observation time and Julian Date ----
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Ensure dt is timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to UTC for calculations
    dt_utc = dt.astimezone(timezone.utc)
    jd = mtime.julian_date(
        dt_utc.year, dt_utc.month, dt_utc.day,
        dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond / 1e6,
    )

    # ---- 2. Moon position ----
    moon_ra, moon_dec, moon_distance_km = moonpos.moon_position(jd)
    moon_alt, moon_az = moonpos.moon_alt_az(lat, lon, jd)

    # ---- 3. Sun position ----
    sun_ra, sun_dec = moonpos.sun_position(jd)
    # Compute sun altitude using the same horizontal conversion
    sun_alt, sun_az = moonpos.moon_alt_az(lat, lon, jd)
    # FIX: moon_alt_az computes moon position — we need sun position
    # Recompute sun altitude/azimuth properly
    jd_now = jd
    # Use the equatorial → horizontal converter directly (it's a private
    # function, so use a workaround via higher-level API)
    from moon.position import _equatorial_to_horizontal
    sun_alt, sun_az = _equatorial_to_horizontal(sun_ra, sun_dec, lat, lon, jd)

    # ---- 4. Moon phase ----
    illum_frac = moonphase.illumination(jd, (sun_ra, sun_dec),
                                        (moon_ra, moon_dec, moon_distance_km))
    waxing = moonphase.is_waxing(sun_ra, moon_ra)
    phase_name = moonphase.phase_name(illum_frac, waxing)
    terminator_angle = moonphase.terminator_angle(jd)

    # ---- 5. Angular diameter ----
    # Moon's angular diameter = 2 * arctan(R_moon / distance)
    # R_moon ≈ 1737.4 km
    moon_radius_km = 1737.4
    angular_diameter = math.degrees(2.0 * math.atan(moon_radius_km / moon_distance_km))

    # ---- 6. Atmospheric extinction / colour tint ----
    # Apply refraction to get apparent altitude for scattering calc
    temp = weather_data.temp_c if weather_data else temp_c
    pressure = weather_data.pressure_mbar if weather_data else pressure_mbar
    humidity = weather_data.humidity if weather_data else humidity_pct

    moon_app_alt = atmo_refract.apparent_from_true(moon_alt, temp, pressure)
    tint_r, tint_g, tint_b = atmo_scatter.moon_color_tint(
        max(moon_app_alt, 0.1), temp, pressure, humidity,
    )

    # ---- 7. Build image layers ----

    # 7a. Sky gradient
    sky = sky_gradient(sun_alt, moon_alt, image_w, image_h)

    # 7b. Moon disk
    moon_radius_px = moon_size_pixels(angular_diameter, fov_deg, image_w) // 2
    moon_radius_px = max(moon_radius_px, 2)  # at least 2px radius for visibility

    moon_disk = render_moon_disk(
        illum_frac,
        terminator_angle,
        (tint_r, tint_g, tint_b),
        moon_radius_px,
    )

    # Position on image
    moon_x, moon_y = moon_position_on_image(
        moon_app_alt, moon_az, fov_deg, image_w, image_h,
    )

    # Composite moon onto sky
    sky_rgba = sky.convert("RGBA")
    # Calculate moon paste position (top-left corner of moon disk image)
    paste_x = moon_x - moon_radius_px
    paste_y = moon_y - moon_radius_px
    sky_rgba.paste(moon_disk, (paste_x, paste_y), moon_disk)

    # 7c. Horizon
    horizon = horizon_line(sky_rgba, observer_height_m, sun_alt)

    # 7d. Weather overlays
    current = horizon.convert("RGBA")

    cloud_cover = weather_data.cloud_cover_pct if weather_data else 0.0
    visibility_km = weather_data.visibility_km if weather_data else 10.0

    if cloud_cover > 1.0:
        current = render_clouds(current, cloud_cover, (moon_x, moon_y))

    if visibility_km < 20.0:
        current = render_haze(current, visibility_km)

    if humidity > 80.0:
        current = render_fog(current, humidity)

    # 7e. Annotations
    loc_data = LocationData(city=city, state=state, lat=lat, lon=lon)
    moon_data = MoonData(
        illumination_pct=illum_frac * 100.0,
        altitude_deg=moon_app_alt,
        azimuth_deg=moon_az,
        phase_name=phase_name,
    )
    w_data = WeatherAnnotationData(
        temp_c=temp,
        conditions=weather_data.conditions if weather_data else "",
        wind_speed_ms=weather_data.wind_speed if weather_data else 0.0,
        humidity_pct=humidity,
    )
    t_data = TimeData(
        date_str=dt_utc.strftime("%Y-%m-%d"),
        time_str=dt_utc.strftime("%H:%M"),
        timezone_str=timezone_str or "UTC",
    )
    result = annotate_image(current, loc_data, moon_data, w_data, t_data)

    return result


def save_image(image: Image.Image, path: str) -> str:
    """Save a PIL ``Image`` as a PNG file.

    Creates parent directories if they do not exist.

    Args:
        image: The PIL ``Image`` to save.
        path: Filesystem path for the output PNG.

    Returns:
        The absolute path of the saved file.
    """
    out_path = os.path.abspath(path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    image.save(out_path, "PNG")
    return out_path
