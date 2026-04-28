"""Location resolver using geopy (Nominatim) and timezonefinder.

Provides functions to resolve location data from ZIP codes, city/state/
country pairs, or coordinate validation.  Each returns a ``Location``
namedtuple with available geographic metadata.
"""

import re
from collections import namedtuple
from typing import Optional

from geopy.geocoders import Nominatim

from . import timezone as _timezone

# Named tuple returned by all resolve functions.
# Fields: lat, lon, city, state, country, zip_code, timezone_str
Location = namedtuple(
    "Location",
    ["lat", "lon", "city", "state", "country", "zip_code", "timezone_str"],
)

_geolocator = Nominatim(user_agent="moonshot-locator/1.0")


def _resolve_address(loc, lat: float, lon: float) -> dict:
    """Extract a structured address dict from a Nominatim result.

    Nominatim results for place types like ``postcode`` often omit the
    ``"address"`` key.  When that happens we fall back to a reverse
    geocode of the coordinates to get the full address breakdown.

    Args:
        loc: A ``geopy.location.Location`` result.
        lat: Latitude already extracted from *loc*.
        lon: Longitude already extracted from *loc*.

    Returns:
        A dictionary with keys such as ``city``, ``state``, ``country``,
        ``town``, ``village``, etc.  May be empty if both lookups fail.
    """
    address = loc.raw.get("address", {})
    if address:
        return address
    # Fallback: reverse geocode the coordinates
    try:
        rev = _geolocator.reverse((lat, lon), exactly_one=True)
        if rev is not None:
            return rev.raw.get("address", {})
    except Exception:
        pass
    return {}


def from_zip(zip_code: str, country: str = "USA") -> Optional[Location]:
    """Resolve a ZIP / postal code to a ``Location``.

    Args:
        zip_code: ZIP or postal code string.
        country: Country name (default "USA").

    Returns:
        A ``Location`` populated with lat, lon, city, state, country,
        zip_code and timezone_str, or None if geocoding fails.
    """
    if not is_valid_us_zip(zip_code):
        print(f"Warning: ZIP code {zip_code} does not look like a standard "
              f"5-digit US code; querying Nominatim anyway.")
    try:
        loc = _geolocator.geocode(f"{zip_code}, {country}", exactly_one=True)
        if loc is None:
            print(f"Warning: could not geocode ZIP code {zip_code} in {country}")
            return None

        lat, lon = loc.latitude, loc.longitude
        address = _resolve_address(loc, lat, lon)
        city = address.get("city") or address.get("town") or address.get("village", "")
        state = address.get("state", "")
        resolved_country = address.get("country", country)
        tz = _timezone.timezone_at(lat, lon)

        return Location(lat, lon, city, state, resolved_country, zip_code, tz)
    except Exception as exc:
        print(f"Warning: geocoding failed for ZIP {zip_code}: {exc}")
        return None


def from_city_state(city: str, state: str, country: str = "") -> Optional[Location]:
    """Resolve a city and state / province / region to a ``Location``.

    If *country* is empty and *state* is given, defaults to ``"USA"``.

    Args:
        city: City name.
        state: Full state/province name or two-letter abbreviation.
        country: Country name.  When empty with a state, "USA" is used.

    Returns:
        A ``Location`` with lat, lon, city, state, country, zip_code
        (empty) and timezone_str, or None if geocoding fails.
    """
    _country = country if country else "USA"
    query = f"{city}, {state}, {_country}"
    try:
        loc = _geolocator.geocode(query, exactly_one=True)
        if loc is None:
            print(f"Warning: could not geocode '{query}'")
            return None

        lat, lon = loc.latitude, loc.longitude
        address = _resolve_address(loc, lat, lon)
        resolved_country = address.get("country", _country)
        tz = _timezone.timezone_at(lat, lon)

        return Location(lat, lon, city, state, resolved_country, "", tz)
    except Exception as exc:
        print(f"Warning: geocoding failed for '{query}': {exc}")
        return None


def from_city_country(city: str, country: str = "") -> Optional[Location]:
    """Resolve a city and optional country (no state) to a ``Location``.

    When *country* is empty, the city is resolved globally without any
    country restriction.

    Args:
        city: City name.
        country: Country name (optional; empty for global resolution).

    Returns:
        A ``Location`` with lat, lon, city, state (empty string),
        country, zip_code (empty) and timezone_str, or None.
    """
    query = city if not country else f"{city}, {country}"
    try:
        loc = _geolocator.geocode(query, exactly_one=True)
        if loc is None:
            print(f"Warning: could not geocode '{query}'")
            return None

        lat, lon = loc.latitude, loc.longitude
        address = _resolve_address(loc, lat, lon)
        resolved_city = address.get("city") or address.get("town") or address.get("village", city)
        resolved_state = address.get("state", "")
        resolved_country = address.get("country", country)
        tz = _timezone.timezone_at(lat, lon)

        return Location(lat, lon, resolved_city, resolved_state, resolved_country, "", tz)
    except Exception as exc:
        print(f"Warning: geocoding failed for '{query}': {exc}")
        return None


def from_lat_lon(lat: float, lon: float) -> Optional[Location]:
    """Validate and wrap a lat/lon pair into a ``Location``.

    This function only validates coordinate bounds and resolves the
    timezone.  City, state, country, and ZIP fields are left as empty.

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

    tz = _timezone.timezone_at(lat, lon)
    return Location(lat, lon, "", "", "", "", tz)


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
