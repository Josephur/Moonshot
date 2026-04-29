"""Shared fixtures and helpers for Moonshot tests."""

import base64
import logging
import subprocess
import sys as _sys
from pathlib import Path

import pytest
import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
MOONSHOT_DIR = Path(__file__).resolve().parents[1]
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OUTPUT_DIR = MOONSHOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Ensure src/ is on sys.path for weather and render imports
_SRC_PATH = str(MOONSHOT_DIR / "src")
if _SRC_PATH not in _sys.path:
    _sys.path.insert(0, _SRC_PATH)

from weather.provider import WeatherData


# ── Ollama availability check ─────────────────────────────────
def ollama_available() -> bool:
    """Return True if Ollama is running and moondream model is available."""
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        models = {m["name"].split(":")[0] for m in r.json().get("models", [])}
        return "moondream" in models
    except (requests.ConnectionError, requests.Timeout):
        return False


# Module-level skipif condition — evaluated at collection time
_ollama_ready = ollama_available()
ollama_not_ready = pytest.mark.skipif(
    not _ollama_ready,
    reason="Ollama/moondream not available (run: ollama pull moondream && ollama serve)",
)


# ── Vision model helpers ──────────────────────────────────────
def ask_model(image_path: str | Path, prompt: str) -> str:
    """Send an image to local moondream and return the raw response text."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": "moondream",
        "prompt": prompt,
        "images": [b64],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    text = resp.json().get("response", "")
    logger.info("Model prompt: %s", prompt)
    logger.info("Model response: %s", text)
    return text


def ask_with_retry(image_path: str | Path, prompt: str, retries: int = 2) -> str:
    """Ask the model, retrying on empty response."""
    for attempt in range(retries + 1):
        response = ask_model(image_path, prompt)
        if response.strip():
            return response
        logger.warning("Empty response on attempt %d/%d", attempt + 1, retries + 1)
    return response


def check_yes(response: str) -> bool:
    """Return True if response starts with Yes/yes or similar affirmation."""
    upper = response.strip().upper()
    return upper.startswith("YES") or "YES," in upper or upper.startswith("Y")


def extract_choice(response: str, options: set[str]) -> str | None:
    """Return the first option found in the response (case-insensitive)."""
    upper = response.upper()
    for option in options:
        if option in upper:
            return option
    return None


# ── Image rendering helper ────────────────────────────────────
def render_image(
    output_name: str,
    lat: float | None = None,
    lon: float | None = None,
    city: str | None = None,
    state: str | None = None,
    date: str | None = None,
    time: str | None = None,
    fov: float | None = None,
) -> Path:
    """Run Moonshot CLI and return the output image path.

    By default uses lat/lon for Indianapolis. Override with keyword args.
    """
    out_path = OUTPUT_DIR / output_name
    cmd = ["python3", "-m", "src.main", "--output", str(out_path)]

    if city and state:
        cmd.extend(["--city", city, "--state", state])
    else:
        cmd.extend([
            "--lat", str(lat if lat is not None else 39.77),
            "--lon", str(lon if lon is not None else -86.16),
        ])

    if date:
        cmd.extend(["--date", date])
    if time:
        cmd.extend(["--time", time])
    if fov is not None:
        cmd.extend(["--fov", str(fov)])

    logger.info("Rendering: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=MOONSHOT_DIR, capture_output=True, check=True, text=True)
    return out_path


# ── Predefined weather scenarios ────────────────────────────
CLEAR_SKY = WeatherData(
    temp_c=15.0, pressure_mbar=1013.0, humidity=50.0,
    cloud_cover_pct=0.0, visibility_km=10.0,
    conditions="clear sky", wind_speed=0.0,
)
OVERCAST = WeatherData(
    temp_c=15.0, pressure_mbar=1013.0, humidity=90.0,
    cloud_cover_pct=100.0, visibility_km=5.0,
    conditions="overcast clouds", wind_speed=5.0,
)
LIGHT_CLOUDS = WeatherData(
    temp_c=15.0, pressure_mbar=1013.0, humidity=50.0,
    cloud_cover_pct=30.0, visibility_km=10.0,
    conditions="scattered clouds", wind_speed=3.0,
)
FOGGY = WeatherData(
    temp_c=15.0, pressure_mbar=1013.0, humidity=95.0,
    cloud_cover_pct=0.0, visibility_km=1.0,
    conditions="fog", wind_speed=2.0,
)
HAZY = WeatherData(
    temp_c=15.0, pressure_mbar=1013.0, humidity=70.0,
    cloud_cover_pct=30.0, visibility_km=3.0,
    conditions="haze", wind_speed=3.0,
)


# ── Weather image rendering helper ──────────────────────────
def render_weather_image(
    output_name: str,
    weather_data: WeatherData,
    lat: float = 39.77,
    lon: float = -86.16,
    date: str = "2026-04-28",
    time: str = "21:00",
    fov: float | None = None,
) -> Path:
    """Render with specific weather, bypassing real API.

    Calls ``generate_moon_image()`` directly with a known ``WeatherData``
    instance, bypassing CLI argument parsing and real weather API.

    Args:
        output_name: Filename for the output image (e.g. "w1_clear.png").
        weather_data: A ``WeatherData`` instance with desired weather.
        lat: Latitude (default Indianapolis).
        lon: Longitude (default Indianapolis).
        date: Date string YYYY-MM-DD.
        time: Time string HH:MM in 24-hour local.
        fov: Field of view in degrees (default auto-calculated).

    Returns:
        Path to the saved output image.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from render.composite import generate_moon_image

    out_path = OUTPUT_DIR / output_name

    year, month, day = map(int, date.split("-"))
    hour, minute = map(int, time.split(":"))
    tz = ZoneInfo("America/Indiana/Indianapolis")
    local_dt = datetime(year, month, day, hour, minute, tzinfo=tz)

    img = generate_moon_image(
        lat=lat, lon=lon,
        city="", state="", country="USA",
        timezone_str="America/Indiana/Indianapolis",
        dt=local_dt,
        weather_data=weather_data,
        image_w=1920, image_h=1080,
        fov_deg=fov if fov is not None else 90.0,
        api_key="",
    )
    img.save(out_path)
    logger.info("Weather render saved: %s", out_path)
    return out_path
