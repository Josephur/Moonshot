"""Moonshot — CLI entry point for moon-position rendering.

Usage::

    python -m src.main --zip 46201
    python -m src.main --city "Indianapolis" --state "IN" --lat 39.7 --lon -86.2
                       [--date 2026-04-27] [--time 21:00] [--fov 90]
                       [--width 1920] [--height 1080]
                       [--output moon.png]
                       [--weather-api-key KEY]
"""

import argparse
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

from .config import Config, get_api_key
from .location.geocode import from_zip, from_city_state, from_lat_lon, Location
from .weather.provider import fetch_weather, default_weather


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Moonshot — render the Moon as seen from a given location and time.",
    )

    # Location group (--zip OR --city/--state/--lat/--lon).
    loc = parser.add_argument_group("location (mutually exclusive)")

    loc.add_argument("--zip", type=str, default=None,
                     help="5-digit US ZIP code")
    loc.add_argument("--city", type=str, default=None,
                     help="City name (requires --state)")
    loc.add_argument("--state", type=str, default=None,
                     help="State name or code (requires --city)")
    loc.add_argument("--lat", type=float, default=None,
                     help="Latitude in decimal degrees")
    loc.add_argument("--lon", type=float, default=None,
                     help="Longitude in decimal degrees")

    # Date / time.
    parser.add_argument("--date", type=str, default=None,
                        help="Observation date YYYY-MM-DD (default: today)")
    parser.add_argument("--time", type=str, default=None,
                        help="Observation time HH:MM in 24-hr local (default: now)")

    # Rendering overrides.
    parser.add_argument("--fov", type=int, default=None,
                        help="Field of view in degrees (default: 90)")
    parser.add_argument("--width", type=int, default=None,
                        help="Output image width in pixels (default: 1920)")
    parser.add_argument("--height", type=int, default=None,
                        help="Output image height in pixels (default: 1080)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image filename (default: moon.png)")

    # Weather.
    parser.add_argument("--weather-api-key", type=str, default=None,
                        help="OpenWeatherMap API key (overrides env var)")

    return parser


def resolve_location(args) -> Optional[Location]:
    """Resolve location from parsed arguments.

    Priority: --zip > --city/--state/--lat/--lon > --lat/--lon alone.

    Returns:
        A ``Location`` namedtuple, or None if resolution fails.
    """
    if args.zip:
        return from_zip(args.zip)

    if args.city and args.state:
        return from_city_state(args.city, args.state)

    if args.lat is not None and args.lon is not None:
        return from_lat_lon(args.lat, args.lon)

    print("Error: provide --zip OR --city/--state/--lat/--lon OR --lat/--lon")
    return None


def compute_julian_date(dt: datetime) -> float:
    """Compute the Julian Date for a given UTC datetime.

    Implements the standard astronomical formula.

    Args:
        dt: Timezone-aware UTC datetime.

    Returns:
        Julian Date as a float.
    """
    y = dt.year
    m = dt.month
    d = dt.day + (
        dt.hour / 24.0 +
        dt.minute / 1440.0 +
        dt.second / 86400.0
    )

    if m <= 2:
        y -= 1
        m += 12

    a = y // 100
    b = 2 - a + a // 4

    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5
    return jd


def compute_moon_phase(jd: float) -> float:
    """Calculate the approximate moon phase (illuminated fraction).

    Based on the standard synodic-month approximation.

    Args:
        jd: Julian Date.

    Returns:
        Illuminated fraction between 0.0 (new) and 1.0 (full).
    """
    # New moon reference: 2000-01-06 18:14 UTC = JD 2451549.5
    new_moon_jd = 2451549.5
    synodic_month = 29.53058867

    days_since_new = (jd - new_moon_jd) % synodic_month
    phase_angle = (days_since_new / synodic_month) * 2.0 * 3.14159265

    # Illuminated fraction = (1 - cos(phase_angle)) / 2
    illuminated = (1.0 - __import__("math").cos(phase_angle)) / 2.0
    return illuminated


def phase_name(fraction: float) -> str:
    """Return a human-readable phase name for the illuminated fraction."""
    if fraction < 0.03:
        return "New Moon"
    if fraction < 0.25:
        return "Waxing Crescent"
    if fraction < 0.47:
        return "First Quarter"
    if fraction < 0.53:
        return "Waxing Gibbous"
    if fraction < 0.97:
        return "Waning Gibbous"
    if fraction < 0.99:
        return "Last Quarter"
    return "Waning Crescent"


def main():
    """CLI entry point — coordinate resolution, weather fetch, and output."""
    parser = build_parser()
    args = parser.parse_args()

    # ---- 1. Resolve location -------------------------------------------------
    location = resolve_location(args)
    if location is None:
        sys.exit(1)

    # ---- 2. Determine observation time ---------------------------------------
    now = datetime.now(timezone.utc)
    if args.date:
        obs_date = args.date  # e.g. "2026-04-27"
    else:
        obs_date = now.strftime("%Y-%m-%d")

    if args.time:
        obs_time = args.time  # e.g. "21:00"
    else:
        obs_time = now.strftime("%H:%M")

    # Build a naive local datetime, then convert to UTC via the timezone string.
    tz_str = location.timezone_str
    if tz_str:
        import zoneinfo
        try:
            tz = zoneinfo.ZoneInfo(tz_str)
            hour, minute = map(int, obs_time.split(":"))
            year, month, day = map(int, obs_date.split("-"))
            local_dt = datetime(year, month, day, hour, minute, tzinfo=tz)
            utc_dt = local_dt.astimezone(timezone.utc)
        except Exception as exc:
            print(f"Warning: timezone conversion failed, falling back to UTC: {exc}")
            utc_dt = now
    else:
        print("Warning: no timezone info; using current UTC time")
        utc_dt = now

    # ---- 3. Compute Julian Date ----------------------------------------------
    jd = compute_julian_date(utc_dt)

    # ---- 4. Fetch weather ----------------------------------------------------
    api_key = args.weather_api_key or get_api_key()
    weather = None
    if api_key:
        weather = fetch_weather(location.lat, location.lon, api_key)
    weather = weather or default_weather()

    # ---- 5. Calculate moon position / phase ----------------------------------
    fraction = compute_moon_phase(jd)
    phase = phase_name(fraction)

    # ---- 6. Calculate atmospheric data (placeholder) -------------------------
    # The atmosphere module will be built by another component.
    # For now we pass the weather data directly.

    # ---- 7. Render pipeline (placeholder) -----------------------------------
    config = Config(
        image_width=args.width or 1920,
        image_height=args.height or 1080,
        fov_deg=args.fov or 90,
        output_dir="output",
    )
    output_path = args.output or "moon.png"

    # The render module will be called here once it is built.
    # render.render_moon(location, weather, config, jd, output_path)

    # ---- 8. Print summary ----------------------------------------------------
    print("=" * 56)
    print("  Moonshot — Summary")
    print("=" * 56)
    print(f"  Location:     {location.city or 'N/A'}, {location.state or 'N/A'}")
    print(f"  Coordinates:  {location.lat:.4f}, {location.lon:.4f}")
    print(f"  Timezone:     {location.timezone_str or 'N/A'}")
    print(f"  Date/Time:    {obs_date} {obs_time} local")
    print(f"  UTC:          {utc_dt.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Julian Date:  {jd:.6f}")
    print(f"  Moon Phase:   {phase} ({fraction:.2%} illuminated)")
    print(f"  Conditions:   {weather.conditions}")
    print(f"  Temperature:  {weather.temp_c:.1f} °C")
    print(f"  Pressure:     {weather.pressure_mbar:.1f} mbar")
    print(f"  Humidity:     {weather.humidity:.1f}%")
    print(f"  Cloud cover:  {weather.cloud_cover_pct:.0f}%")
    print(f"  Visibility:   {weather.visibility_km:.1f} km")
    print(f"  Wind:         {weather.wind_speed:.1f} m/s")
    print(f"  Output:       {output_path}")
    print("=" * 56)
    print("  (Render pipeline — coming soon)")
    print("=" * 56)


if __name__ == "__main__":
    main()
