"""
Moonshot render package.

Provides all components needed to build a complete moon-view image:
sky gradients, moon disk rendering, horizon/terrain, weather overlays,
data annotations, and the top-level compositing orchestrator.
"""

from render.sky import sky_gradient
from render.moon_render import (
    moon_size_pixels,
    render_moon_disk,
    moon_position_on_image,
    moon_position_on_image_with_direction,
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
from render.composite import generate_moon_image, save_image

__all__ = [
    # sky
    "sky_gradient",
    # moon_render
    "moon_size_pixels",
    "render_moon_disk",
    "moon_position_on_image",
    "moon_position_on_image_with_direction",
    # horizon
    "horizon_line",
    "horizon_dip_degrees",
    # weather_overlay
    "render_clouds",
    "render_haze",
    "render_fog",
    # annotations
    "annotate_image",
    "LocationData",
    "MoonData",
    "WeatherAnnotationData",
    "TimeData",
    # composite
    "generate_moon_image",
    "save_image",
]
