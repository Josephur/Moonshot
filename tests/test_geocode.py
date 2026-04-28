"""Tests for the location.geocode module.

Mocks external APIs (geopy, timezonefinder) to avoid real network calls.
"""

import sys
import os
from unittest.mock import patch, MagicMock
from collections import namedtuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from location import geocode


class TestIsValidZip:
    """ZIP code validation."""

    def test_is_valid_zip_valid(self):
        """'46201' is a valid 5-digit US ZIP code."""
        assert geocode.is_valid_us_zip("46201") is True

    def test_is_valid_zip_letters(self):
        """'abc' is not a valid ZIP code."""
        assert geocode.is_valid_us_zip("abc") is False

    def test_is_valid_zip_too_long(self):
        """'123456' (6 digits) is not a valid 5-digit ZIP code."""
        assert geocode.is_valid_us_zip("123456") is False

    def test_is_valid_zip_empty(self):
        """Empty string is not a valid ZIP code."""
        assert geocode.is_valid_us_zip("") is False

    def test_is_valid_zip_leading_zeros(self):
        """'00501' (leading zeros) should still be valid."""
        assert geocode.is_valid_us_zip("00501") is True


class TestFromLatLon:
    """Lat/Lon validation and Location wrapping."""

    def test_from_lat_lon_valid(self, monkeypatch):
        """Valid lat/lon should return a Location with same coordinates."""
        # Mock timezone_at to avoid needing timezonefinder
        from location import timezone
        monkeypatch.setattr(timezone, 'timezone_at', lambda lat, lon: "America/New_York")

        loc = geocode.from_lat_lon(39.7, -86.2)
        assert loc is not None
        assert loc.lat == pytest.approx(39.7)
        assert loc.lon == pytest.approx(-86.2)
        assert loc.timezone_str == "America/New_York"

    def test_from_lat_lon_invalid_lat(self):
        """Invalid latitude (outside -90 to 90) should return None."""
        loc = geocode.from_lat_lon(100.0, 0.0)
        assert loc is None

    def test_from_lat_lon_invalid_lon(self):
        """Invalid longitude (outside -180 to 180) should return None."""
        loc = geocode.from_lat_lon(0.0, 200.0)
        assert loc is None

    def test_from_lat_lon_edge_cases(self):
        """Boundary coordinates should still be valid."""
        # North pole
        loc_north = geocode.from_lat_lon(90.0, 0.0)
        assert loc_north is not None

        # South pole
        loc_south = geocode.from_lat_lon(-90.0, 0.0)
        assert loc_south is not None

        # Date line
        loc_date = geocode.from_lat_lon(0.0, 180.0)
        assert loc_date is not None


class TestFromZip:
    """ZIP code geocoding (mocked)."""

    def test_from_zip_success(self, monkeypatch):
        """Successful geocoding should return a Location with correct fields."""
        from location import timezone
        monkeypatch.setattr(timezone, 'timezone_at', lambda lat, lon: "America/New_York")

        # Create a mock geopy location
        mock_loc = MagicMock()
        mock_loc.latitude = 39.7684
        mock_loc.longitude = -86.1581
        mock_loc.raw = {
            "address": {
                "city": "Indianapolis",
                "state": "Indiana",
            }
        }

        # Patch the geolocator on the module
        monkeypatch.setattr(geocode._geolocator, 'geocode', lambda q, **kw: mock_loc)

        loc = geocode.from_zip("46201")
        assert loc is not None
        assert loc.lat == pytest.approx(39.7684)
        assert loc.lon == pytest.approx(-86.1581)
        assert loc.city == "Indianapolis"
        assert loc.state == "Indiana"
        assert loc.zip_code == "46201"

    def test_from_zip_invalid_format(self, monkeypatch):
        """Non-standard ZIP format should warn, try geocoding, and return None on failure."""
        monkeypatch.setattr(geocode._geolocator, 'geocode', lambda q, **kw: None)
        loc = geocode.from_zip("abc")
        assert loc is None

    def test_from_zip_geocode_failure(self, monkeypatch):
        """When geocoding returns None, from_zip should return None."""
        monkeypatch.setattr(geocode._geolocator, 'geocode', lambda q, **kw: None)
        loc = geocode.from_zip("00000")
        assert loc is None


class TestFromCityState:
    """City/state geocoding (mocked)."""

    def test_from_city_state_success(self, monkeypatch):
        """Successful city/state geocoding should return a Location."""
        from location import timezone
        monkeypatch.setattr(timezone, 'timezone_at', lambda lat, lon: "America/Denver")

        mock_loc = MagicMock()
        mock_loc.latitude = 39.7392
        mock_loc.longitude = -104.9903
        mock_loc.raw = {"address": {}}

        monkeypatch.setattr(geocode._geolocator, 'geocode', lambda q, **kw: mock_loc)

        loc = geocode.from_city_state("Denver", "CO")
        assert loc is not None
        assert loc.lat == pytest.approx(39.7392)
        assert loc.lon == pytest.approx(-104.9903)
        assert loc.city == "Denver"
        assert loc.state == "CO"
        assert loc.timezone_str == "America/Denver"

    def test_from_city_state_failure(self, monkeypatch):
        """When geocoding fails, from_city_state should return None."""
        monkeypatch.setattr(geocode._geolocator, 'geocode', lambda q, **kw: None)
        loc = geocode.from_city_state("Nowhere", "XX")
        assert loc is None


class TestValidCoords:
    """Internal _valid_coords helper."""

    def test_valid_coords_center(self):
        """(0, 0) should be valid."""
        assert geocode._valid_coords(0.0, 0.0) is True

    def test_valid_coords_boundary(self):
        """Boundary values should be valid."""
        assert geocode._valid_coords(-90.0, -180.0) is True
        assert geocode._valid_coords(90.0, 180.0) is True

    def test_valid_coords_invalid(self):
        """Out of range values should be invalid."""
        assert geocode._valid_coords(90.1, 0.0) is False
        assert geocode._valid_coords(0.0, 180.1) is False
