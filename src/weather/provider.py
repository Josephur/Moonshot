"""Weather data fetching using OpenWeatherMap.

Provides the ``WeatherData`` dataclass and functions to retrieve
current weather conditions for a given geographic location.
"""

from dataclasses import dataclass, field
from typing import Optional

import requests

OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


@dataclass
class WeatherData:
    """Container for current observed weather conditions.

    Attributes:
        temp_c: Temperature in degrees Celsius.
        pressure_mbar: Atmospheric pressure in millibars (hPa).
        humidity: Relative humidity in percent.
        cloud_cover_pct: Cloud cover percentage.
        visibility_km: Visibility in kilometers.
        conditions: Human-readable weather description (e.g. "clear sky").
        wind_speed: Wind speed in metres per second.
    """
    temp_c: float
    pressure_mbar: float
    humidity: float
    cloud_cover_pct: float
    visibility_km: float
    conditions: str
    wind_speed: float


def fetch_weather(lat: float, lon: float, api_key: str) -> Optional[WeatherData]:
    """Fetch current weather from the OpenWeatherMap API.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        api_key: OpenWeatherMap API key.

    Returns:
        A ``WeatherData`` instance with parsed response values, or
        None if the request failed.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
    }
    try:
        resp = requests.get(OWM_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        weather_list = data.get("weather", [])
        conditions = weather_list[0].get("description", "unknown") if weather_list else "unknown"

        visibility_m = data.get("visibility", 10000)

        return WeatherData(
            temp_c=float(main.get("temp", 15.0)),
            pressure_mbar=float(main.get("pressure", 1013)),
            humidity=float(main.get("humidity", 50)),
            cloud_cover_pct=float(clouds.get("all", 0)),
            visibility_km=float(visibility_m) / 1000.0,
            conditions=conditions,
            wind_speed=float(wind.get("speed", 0.0)),
        )
    except requests.RequestException as exc:
        print(f"Warning: weather API request failed: {exc}")
        return None
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        print(f"Warning: failed to parse weather response: {exc}")
        return None


def default_weather() -> WeatherData:
    """Return a ``WeatherData`` instance with sensible defaults.

    Use this as a fallback when no API key is available or the request
    fails.
    """
    return WeatherData(
        temp_c=15.0,
        pressure_mbar=1013.0,
        humidity=50.0,
        cloud_cover_pct=0.0,
        visibility_km=10.0,
        conditions="clear sky",
        wind_speed=0.0,
    )
