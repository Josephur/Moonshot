"""Tests for the weather.provider module.

Mocks requests to avoid real network calls.
"""

import sys
import os
from unittest.mock import patch, MagicMock

import pytest

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
    """Weather fetching with mocked requests."""

    def _make_mock_response(self, status_code=200, data=None):
        """Helper to create a mock requests.Response."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = data or {
            "main": {
                "temp": 22.5,
                "pressure": 1015,
                "humidity": 60,
            },
            "wind": {"speed": 3.5},
            "clouds": {"all": 25},
            "weather": [{"description": "scattered clouds"}],
            "visibility": 15000,
        }
        return mock_resp

    @patch("weather.provider.requests.get")
    def test_fetch_weather_mocked(self, mock_get):
        """Mock requests.get, verify correct parsing of a normal response."""
        mock_resp = self._make_mock_response()
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
    def test_fetch_weather_api_error(self, mock_get):
        """When the API returns an error, fetch_weather should return None."""
        mock_resp = self._make_mock_response(status_code=401)
        mock_resp.raise_for_status.side_effect = __import__("requests").HTTPError()
        mock_get.return_value = mock_resp

        w = fetch_weather(39.7, -86.2, "bad-key")
        assert w is None

    @patch("weather.provider.requests.get")
    def test_fetch_weather_network_error(self, mock_get):
        """When a network error occurs, fetch_weather should return None."""
        mock_get.side_effect = __import__("requests").RequestException("Connection failed")

        w = fetch_weather(39.7, -86.2, "test-key")
        assert w is None

    @patch("weather.provider.requests.get")
    def test_fetch_weather_missing_fields(self, mock_get):
        """When the response is missing fields, fetch_weather should return None gracefully."""
        mock_resp = self._make_mock_response(data={
            "main": {"temp": 20.0},  # missing pressure, humidity
            "wind": {},
            "clouds": {},
            "visibility": 5000,
        })
        mock_get.return_value = mock_resp

        # Should gracefully default missing fields
        w = fetch_weather(39.7, -86.2, "test-key")
        assert w is not None
        assert w.temp_c == 20.0
        assert w.conditions == "unknown"
        assert w.pressure_mbar == 1013.0  # default

    @patch("weather.provider.requests.get")
    def test_fetch_weather_calls_correct_url(self, mock_get):
        """fetch_weather should call the OWM endpoint with correct params."""
        mock_resp = self._make_mock_response()
        mock_get.return_value = mock_resp

        from weather.provider import OWM_BASE_URL
        fetch_weather(40.0, -75.0, "my-key")

        mock_get.assert_called_once()
        call_args, call_kwargs = mock_get.call_args
        assert call_args[0] == OWM_BASE_URL
        assert call_kwargs["params"]["lat"] == 40.0
        assert call_kwargs["params"]["lon"] == -75.0
        assert call_kwargs["params"]["appid"] == "my-key"
        assert call_kwargs["params"]["units"] == "metric"
