"""Weather data fetching using OpenWeatherMap One Call API 3.0.

Provides the ``WeatherData`` dataclass and functions to retrieve
current weather conditions for a given geographic location.
"""

from dataclasses import dataclass, field
from typing import Optional

import requests

# One Call API 3.0 endpoint (primary)
OWM_ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"
# Free tier fallback endpoint
OWM_FREE_URL = "https://api.openweathermap.org/data/2.5/weather"


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
    """Fetch current weather from the OpenWeatherMap One Call API 3.0.

    Tries the One Call 3.0 endpoint first, then falls back to the
    free 2.5/weather endpoint if the key doesn't have One Call access.

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
        "exclude": "minutely,hourly,daily,alerts",
    }

    # Try One Call 3.0 first
    urls_to_try = [
        (OWM_ONECALL_URL, False),  # One Call: data nested under "current"
        (OWM_FREE_URL, True),       # Free: data at top level in "main" etc.
    ]

    for url, is_free in urls_to_try:
        try:
            if is_free:
                free_params = {k: v for k, v in params.items() if k != "exclude"}
                resp = requests.get(url, params=free_params, timeout=10)
            else:
                resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if is_free:
                # Parse 2.5/weather response
                main = data.get("main", {})
                wind = data.get("wind", {})
                clouds = data.get("clouds", {})
                weather_list = data.get("weather", [])
                visibility_m = data.get("visibility", 10000)
                current_source = {
                    "temp": main.get("temp", 15.0),
                    "pressure": main.get("pressure", 1013),
                    "humidity": main.get("humidity", 50),
                    "clouds": clouds.get("all", 0),
                    "visibility": visibility_m,
                    "wind_speed": wind.get("speed", 0.0),
                    "weather": weather_list,
                }
            else:
                # Parse 3.0/onecall response
                current_source = data.get("current", {})

            temp_c = float(current_source.get("temp", 15.0))
            pressure_mbar = float(current_source.get("pressure", 1013))
            humidity = float(current_source.get("humidity", 50))
            cloud_cover_pct = float(current_source.get("clouds", 0))
            visibility_m = current_source.get("visibility", 10000)
            wind_speed = float(current_source.get("wind_speed", 0.0))

            weather_list = current_source.get("weather", [])
            conditions = weather_list[0].get("description", "unknown") if weather_list else "unknown"

            return WeatherData(
                temp_c=temp_c,
                pressure_mbar=pressure_mbar,
                humidity=humidity,
                cloud_cover_pct=cloud_cover_pct,
                visibility_km=float(visibility_m) / 1000.0,
                conditions=conditions,
                wind_speed=wind_speed,
            )
        except requests.HTTPError as exc:
            if exc.response.status_code == 401:
                # Auth failed — try the next URL
                continue
            print(f"Warning: weather API request failed: {exc}")
            return None
        except requests.RequestException as exc:
            # Network error, timeout, etc.
            print(f"Warning: weather API request failed: {exc}")
            return None
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            print(f"Warning: failed to parse weather response: {exc}")
            return None

    # Both endpoints failed with 401
    print("Warning: weather API key rejected on all endpoints. Check your key.")
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
