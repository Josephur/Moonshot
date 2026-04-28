"""Configuration handler for the Moonshot project.

Provides:
- ``get_api_key()`` — reads the OWM API key from the environment.
- ``Config`` — a dataclass holding render pipeline settings with
  sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


ENV_API_KEY = "MOONSHOT_WEATHER_API_KEY"


def get_api_key() -> Optional[str]:
    """Read the OpenWeatherMap API key from the environment.

    Checks the ``MOONSHOT_WEATHER_API_KEY`` environment variable.

    Returns:
        The API key string or None if the variable is not set.
    """
    return os.environ.get(ENV_API_KEY)


@dataclass
class Config:
    """Render-pipeline configuration with sensible defaults.

    Attributes:
        image_width: Output image width in pixels (default 1920).
        image_height: Output image height in pixels (default 1080).
        fov_deg: Field of view in degrees (default 90).
        output_dir: Directory for rendered output (default "output").
    """
    image_width: int = 1920
    image_height: int = 1080
    fov_deg: int = 90
    output_dir: str = "output"
