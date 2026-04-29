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
    """Read the OpenWeatherMap API key from the environment or .env file.

    Checks the ``MOONSHOT_WEATHER_API_KEY`` environment variable first,
    then falls back to reading from a ``.env`` file in the project root
    (if present).

    Returns:
        The API key string or None if not found.
    """
    # Check environment first
    key = os.environ.get(ENV_API_KEY)
    if key:
        return key

    # Fallback: try loading from .env file
    _env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    try:
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k.strip() == ENV_API_KEY:
                    return v.strip()
    except (FileNotFoundError, OSError):
        pass

    return None


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
