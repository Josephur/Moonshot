"""Tests for the weather.provider module.

Mocks requests to avoid real network calls.
"""

import sys
import os
from unittest.mock import patch, MagicMock, call

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from weather.provider import default_weather, fetch_weather, WeatherData


class TestDefaultWeather:
    """Default weather fallback."""

    def test_default_weather(self):
        """default_weather should return expected sensible defaults."""
        w = default_weather()
        assert w.temp_c == 15.0
        assert w.pressure_mbar == 1013.0
        assert w.humidity == 50.0
        assert w.cloud_cover_pct == 0.0
        assert w.visibility_km == 10.0
        assert w.conditions == "clear sky"
        assert w.wind_speed == 0.0

    def test_default_weather_type(self):
        """default_weather should return a WeatherData instance."""
        w = default_weather()
        assert isinstance(w, WeatherData)


class TestFetchWeather:
    """Weather fetching with mocked requests.

    The provider tries the One Call 3.0 endpoint first, then falls
    back to the free 2.5/weather endpoint on 401.
    """

    # ── One Call 3.0 response format ─────────────────────────────
    _ONECALL_DATA = {
        "lat": 39.7,
        "lon": -86.2,
        "timezone": "America/Indianapolis",
        "current": {
            "temp": 22.5,
            "pressure": 1015,
            "humidity": 60,
            "clouds": 25,
            "visibility": 15000,
            "wind_speed": 3.5,
            "weather": [{"description": "scattered clouds"}],
        },
    }

    # ── Free 2.5/weather response format ─────────────────────────
    _FREE_DATA = {
        "main": {"temp": 18.0, "pressure": 1012, "humidity": 55},
        "wind": {"speed": 2.0},
        "clouds": {"all": 40},
        "weather": [{"description": "broken clouds"}],
        "visibility": 10000,
    }

    def _make_mock_response(self, data, status_code=200):
        """Helper to create a mock requests.Response."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = data
        if status_code >= 400:
            mock_resp.raise_for_status.side_effect = __import__("requests").HTTPError(
                response=mock_resp
            )
        return mock_resp

    # ── One Call 3.0 succeeds (primary path) ─────────────────────
    @patch("weather.provider.requests.get")
    def test_oncall_success(self, mock_get):
        """One Call 3.0 response should be parsed correctly."""
        mock_resp = self._make_mock_response(self._ONECALL_DATA)
        mock_get.return_value = mock_resp

        w = fetch_weather(39.7, -86.2, "test-key")
        assert w is not None
        assert w.temp_c == pytest.approx(22.5)
        assert w.pressure_mbar == 1015.0
        assert w.humidity == 60.0
        assert w.cloud_cover_pct == 25.0
        assert w.visibility_km == 15.0
        assert w.conditions == "scattered clouds"
        assert w.wind_speed == 3.5

    @patch("weather.provider.requests.get")
    def test_oncall_passes_correct_params(self, mock_get):
        """One Call 3.0 should be called with lat, lon, appid, units, exclude."""
        mock_resp = self._make_mock_response(self._ONECALL_DATA)
        mock_get.return_value = mock_resp

        from weather.provider import OWM_ONECALL_URL
        fetch_weather(40.0, -75.0, "my-key")

        mock_get.assert_called_once_with(
            OWM_ONECALL_URL,
            params={
                "lat": 40.0,
                "lon": -75.0,
                "appid": "my-key",
                "units": "metric",
                "exclude": "minutely,hourly,daily,alerts",
            },
            timeout=10,
        )

    # ── On Call 3.0 401 → fallback to 2.5/weather ───────────────
    @patch("weather.provider.requests.get")
    def test_fallback_on_401(self, mock_get):
        """When One Call 3.0 returns 401, fall back to 2.5/weather."""
        mock_401 = self._make_mock_response({"error": "unauthorized"}, status_code=401)
        mock_ok = self._make_mock_response(self._FREE_DATA)

        mock_get.side_effect = [mock_401, mock_ok]

        w = fetch_weather(39.7, -86.2, "my-key")
        assert w is not None
        assert w.temp_c == pytest.approx(18.0)
        assert w.cloud_cover_pct == 40.0
        assert w.conditions == "broken clouds"
        assert w.wind_speed == 2.0

        # Should have called both endpoints
        assert mock_get.call_count == 2

    @patch("weather.provider.requests.get")
    def test_fallback_uses_free_endpoint_without_exclude(self, mock_get):
        """The free 2.5/weather fallback should not include 'exclude' param."""
        mock_401 = self._make_mock_response({"error": "unauthorized"}, status_code=401)
        mock_ok = self._make_mock_response(self._FREE_DATA)
        mock_get.side_effect = [mock_401, mock_ok]

        from weather.provider import OWM_FREE_URL
        fetch_weather(40.0, -75.0, "my-key")

        # Second call should use FREE_URL without exclude
        second_call = mock_get.call_args_list[1]
        assert second_call[0][0] == OWM_FREE_URL
        assert "exclude" not in second_call[1]["params"]

    # ── Both endpoints fail 401 ─────────────────────────────────
    @patch("weather.provider.requests.get")
    def test_both_endpoints_401(self, mock_get):
        """When both endpoints return 401, fetch_weather should return None."""
        mock_401a = self._make_mock_response({"error": "no"}, status_code=401)
        mock_401b = self._make_mock_response({"error": "no"}, status_code=401)
        mock_get.side_effect = [mock_401a, mock_401b]

        w = fetch_weather(39.7, -86.2, "bad-key")
        assert w is None
        assert mock_get.call_count == 2

    # ── Network error ───────────────────────────────────────────
    @patch("weather.provider.requests.get")
    def test_network_error(self, mock_get):
        """When a network error occurs, fetch_weather should return None."""
        mock_get.side_effect = __import__("requests").RequestException("Connection failed")

        w = fetch_weather(39.7, -86.2, "test-key")
        assert w is None

    # ── Missing fields ──────────────────────────────────────────
    @patch("weather.provider.requests.get")
    def test_missing_fields(self, mock_get):
        """When the response is missing fields, graceful defaults should be used."""
        mock_resp = self._make_mock_response({
            "current": {
                "temp": 20.0,  # missing pressure, humidity
                "weather": [],
            },
        })
        mock_get.return_value = mock_resp

        w = fetch_weather(39.7, -86.2, "test-key")
        assert w is not None
        assert w.temp_c == 20.0
        assert w.conditions == "unknown"
        assert w.pressure_mbar == 1013.0  # default
        assert w.humidity == 50.0  # default

