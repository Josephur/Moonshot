"""Timezone resolution using timezonefinder.

Provides utilities to resolve timezone strings from geographic coordinates
and convert between local and UTC datetime objects.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from timezonefinder import TimezoneFinder

_tf = TimezoneFinder()


def timezone_at(lat: float, lon: float) -> Optional[str]:
    """Resolve the timezone string at the given coordinates.

    Args:
        lat: Latitude in decimal degrees (-90 to 90).
        lon: Longitude in decimal degrees (-180 to 180).

    Returns:
        IANA timezone string (e.g. "America/New_York"), or None if
        the coordinates fall outside any known timezone.
    """
    try:
        return _tf.timezone_at(lat=lat, lng=lon)
    except Exception as exc:
        print(f"Warning: timezone lookup failed for ({lat}, {lon}): {exc}")
        return None


def local_time_to_utc(local_dt: datetime, timezone_str: str) -> Optional[datetime]:
    """Convert a naive local datetime to UTC.

    The *local_dt* is interpreted as wall-clock time in the given IANA
    timezone *timezone_str*.  The returned datetime is timezone-aware
    (UTC).

    Args:
        local_dt: Naive datetime representing local wall-clock time.
        timezone_str: IANA timezone string (e.g. "America/New_York").

    Returns:
        Timezone-aware UTC datetime, or None on failure.
    """
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(timezone_str)
        local_aware = local_dt.replace(tzinfo=tz)
        return local_aware.astimezone(timezone.utc)
    except Exception as exc:
        print(f"Warning: could not convert {local_dt} to UTC for {timezone_str}: {exc}")
        return None


def utc_to_local_time(utc_dt: datetime, timezone_str: str) -> Optional[datetime]:
    """Convert a UTC datetime to local time in the given timezone.

    Args:
        utc_dt: Timezone-aware or naive UTC datetime.  If naive it is
                assumed to be UTC.
        timezone_str: IANA timezone string (e.g. "America/New_York").

    Returns:
        Timezone-aware datetime in the local timezone, or None on failure.
    """
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(timezone_str)
        if utc_dt.tzinfo is None:
            utc_aware = utc_dt.replace(tzinfo=timezone.utc)
        else:
            utc_aware = utc_dt.astimezone(timezone.utc)
        return utc_aware.astimezone(tz)
    except Exception as exc:
        print(f"Warning: could not convert UTC {utc_dt} to {timezone_str}: {exc}")
        return None
