"""US location resolver using geopy (Nominatim) and timezonefinder.

Provides functions to resolve location data from ZIP codes, city/state
pairs, or coordinate validation.  Each returns a ``Location`` namedtuple
with available geographic metadata.
"""

import re
from collections import namedtuple
from typing import Optional

from geopy.geocoders import Nominatim

from .timezone import timezone_at

# Named tuple returned by all resolve functions.
Location = namedtuple(
    "Location",
    ["lat", "lon", "city", "state", "zip_code", "timezone_str"],
)

_geolocator = Nominatim(user_agent="moonshot-locator/1.0")


def from_zip(zip_code: str) -> Optional[Location]:
    """Resolve a US ZIP code to a ``Location``.

    Args:
        zip_code: 5-digit US ZIP code string.

    Returns:
        A ``Location`` populated with lat, lon, city, state, zip_code
        and timezone_str, or None if geocoding fails.
    """
    if not is_valid_us_zip(zip_code):
        print(f"Warning: invalid ZIP code format: {zip_code}")
        return None

    try:
        loc = _geolocator.geocode(f"{zip_code}, USA", exactly_one=True)
        if loc is None:
            print(f"Warning: could not geocode ZIP code {zip_code}")
            return None

        lat, lon = loc.latitude, loc.longitude
        address = loc.raw.get("address", {})
        city = address.get("city") or address.get("town") or address.get("village", "")
        state = address.get("state", "")
        tz = timezone_at(lat, lon)

        return Location(lat, lon, city, state, zip_code, tz)
    except Exception as exc:
        print(f"Warning: geocoding failed for ZIP {zip_code}: {exc}")
        return None


def from_city_state(city: str, state: str) -> Optional[Location]:
    """Resolve a US city and state (or state code) to a ``Location``.

    Args:
        city: City name.
        state: Full state name or two-letter abbreviation.

    Returns:
        A ``Location`` populated with lat, lon, city, state, zip_code
        (empty string) and timezone_str, or None if geocoding fails.
    """
    query = f"{city}, {state}, USA"
    try:
        loc = _geolocator.geocode(query, exactly_one=True)
        if loc is None:
            print(f"Warning: could not geocode '{query}'")
            return None

        lat, lon = loc.latitude, loc.longitude
        tz = timezone_at(lat, lon)

        return Location(lat, lon, city, state, "", tz)
    except Exception as exc:
        print(f"Warning: geocoding failed for '{query}': {exc}")
        return None


def from_lat_lon(lat: float, lon: float) -> Optional[Location]:
    """Validate and wrap a lat/lon pair into a ``Location``.

    This function only validates coordinate bounds and resolves the
    timezone.  City, state, and ZIP fields are left as empty strings.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        A ``Location`` with validated lat/lon and resolved timezone,
        or None if the coordinates are out of range.
    """
    if not _valid_coords(lat, lon):
        print(f"Warning: coordinates ({lat}, {lon}) are out of range")
        return None

    tz = timezone_at(lat, lon)
    return Location(lat, lon, "", "", "", tz)


def is_valid_us_zip(zip_code: str) -> bool:
    """Check whether *zip_code* is a valid 5-digit US ZIP code.

    Args:
        zip_code: String to validate.

    Returns:
        True if the string consists of exactly 5 decimal digits.
    """
    return bool(re.fullmatch(r"\d{5}", zip_code))


def _valid_coords(lat: float, lon: float) -> bool:
    """Return True if latitude and longitude are within valid ranges."""
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
