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


# ── W4: Weather Data Pipeline ────────────────────────────────
class TestWeatherDataPipeline:
    """W4: Weather data flows correctly to the render pipeline."""

    @patch("render.composite.render_clouds")
    @patch("render.composite.render_haze")
    @patch("render.composite.render_fog")
    def test_clear_activates_haze_only(self, mock_fog, mock_haze, mock_clouds):
        """Clear sky (0% clouds, 10km vis) triggers haze but not fog or clouds."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from render.composite import generate_moon_image

        w = WeatherData(
            temp_c=15.0, pressure_mbar=1013.0, humidity=50.0,
            cloud_cover_pct=0.0, visibility_km=10.0,
            conditions="clear sky", wind_speed=0.0,
        )
        dt = datetime(2026, 4, 28, 21, 0, tzinfo=ZoneInfo("America/Indiana/Indianapolis"))
        mock_clouds.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        mock_haze.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        mock_fog.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        generate_moon_image(
            lat=39.77, lon=-86.16, dt=dt, weather_data=w,
            image_w=100, image_h=100, fov_deg=90.0, api_key="",
        )
        # 0% clouds → no cloud render; 10km <20 → haze called; 50% humidity → no fog
        assert not mock_clouds.called, "render_clouds should NOT be called at 0% clouds"
        assert mock_haze.called, "render_haze should be called at 10km visibility"
        assert not mock_fog.called, "render_fog should NOT be called at 50% humidity"

    @patch("render.composite.render_clouds")
    @patch("render.composite.render_haze")
    @patch("render.composite.render_fog")
    def test_cloud_threshold_triggers_render_clouds(self, mock_fog, mock_haze, mock_clouds):
        """Cloud cover >1% triggers render_clouds."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from render.composite import generate_moon_image

        w = WeatherData(
            temp_c=15.0, pressure_mbar=1013.0, humidity=50.0,
            cloud_cover_pct=50.0, visibility_km=50.0,  # high vis to avoid haze
            conditions="scattered clouds", wind_speed=3.0,
        )
        dt = datetime(2026, 4, 28, 21, 0, tzinfo=ZoneInfo("America/Indiana/Indianapolis"))
        mock_clouds.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        mock_haze.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        mock_fog.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        generate_moon_image(
            lat=39.77, lon=-86.16, dt=dt, weather_data=w,
            image_w=100, image_h=100, fov_deg=90.0, api_key="",
        )
        assert mock_clouds.called, "render_clouds should be called when cloud cover >1%"

    @patch("render.composite.render_clouds")
    @patch("render.composite.render_haze")
    @patch("render.composite.render_fog")
    def test_visibility_threshold_triggers_haze(self, mock_fog, mock_haze, mock_clouds):
        """Visibility <20km triggers render_haze."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from render.composite import generate_moon_image

        w = WeatherData(
            temp_c=15.0, pressure_mbar=1013.0, humidity=50.0,
            cloud_cover_pct=0.0, visibility_km=5.0,  # <20km → haze
            conditions="haze", wind_speed=3.0,
        )
        dt = datetime(2026, 4, 28, 21, 0, tzinfo=ZoneInfo("America/Indiana/Indianapolis"))
        mock_clouds.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        mock_haze.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        mock_fog.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        generate_moon_image(
            lat=39.77, lon=-86.16, dt=dt, weather_data=w,
            image_w=100, image_h=100, fov_deg=90.0, api_key="",
        )
        assert mock_haze.called, "render_haze should be called when visibility <20km"

    @patch("render.composite.render_clouds")
    @patch("render.composite.render_haze")
    @patch("render.composite.render_fog")
    def test_humidity_threshold_triggers_fog(self, mock_fog, mock_haze, mock_clouds):
        """Humidity >=80% triggers render_fog."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from render.composite import generate_moon_image

        w = WeatherData(
            temp_c=15.0, pressure_mbar=1013.0, humidity=90.0,
            cloud_cover_pct=0.0, visibility_km=50.0,
            conditions="fog", wind_speed=2.0,
        )
        dt = datetime(2026, 4, 28, 21, 0, tzinfo=ZoneInfo("America/Indiana/Indianapolis"))
        mock_clouds.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        mock_haze.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        mock_fog.return_value = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        generate_moon_image(
            lat=39.77, lon=-86.16, dt=dt, weather_data=w,
            image_w=100, image_h=100, fov_deg=90.0, api_key="",
        )
        assert mock_fog.called, "render_fog should be called when humidity >=80%"


# ── W5: Weather Overlay Units ────────────────────────────────
class TestWeatherOverlayUnits:
    """W5: Direct pixel-level checks on weather overlay functions."""

    def test_zero_clouds_unchanged(self):
        """0% clouds → render_clouds returns image unchanged."""
        from render.weather_overlay import render_clouds
        img = Image.new("RGB", (50, 50), (100, 100, 100))
        result = render_clouds(img, 0.0)
        # Should be identity for <1% cloud
        assert list(img.convert("RGBA").tobytes()) == list(result.tobytes()), (
            "Image should be unchanged at 0% clouds"
        )

    def test_full_clouds_modifies_image(self):
        """100% clouds → render_clouds modifies the image pixels."""
        from render.weather_overlay import render_clouds
        img = Image.new("RGB", (50, 50), (100, 100, 100))
        result = render_clouds(img, 100.0)
        original_data = list(img.convert("RGBA").tobytes())
        result_data = list(result.tobytes())
        assert original_data != result_data, (
            "Image should be modified at 100% clouds"
        )

    def test_haze_identity_at_high_visibility(self):
        """50km visibility → render_haze returns image nearly unchanged."""
        from render.weather_overlay import render_haze
        img = Image.new("RGB", (50, 50), (100, 100, 100))
        result = render_haze(img, 50.0)
        # At 50km, haze strength is 0, so image should be unchanged
        original_data = list(img.convert("RGBA").tobytes())
        result_data = list(result.tobytes())
        assert original_data == result_data, (
            "Image should be unchanged at 50km visibility"
        )

    def test_haze_modifies_at_low_visibility(self):
        """1km visibility → render_haze modifies top rows more than bottom rows."""
        from render.weather_overlay import render_haze
        img = Image.new("RGB", (50, 50), (100, 100, 100))
        result = render_haze(img, 1.0)

        # Haze gradient goes top=1.0 → bottom=0.0, so top rows get more haze
        original_pixel = (100, 100, 100)

        top_pixel = result.getpixel((0, 0))
        bot_pixel = result.getpixel((0, 49))

        top_diff = sum(abs(top_pixel[i] - original_pixel[i]) for i in range(3))
        bot_diff = sum(abs(bot_pixel[i] - original_pixel[i]) for i in range(3))

        assert top_diff > 0, "Top of image should be modified by haze"
        assert top_diff >= bot_diff, (
            f"Top rows should be more modified than bottom rows "
            f"(top: {top_diff}, bot: {bot_diff})"
        )

    def test_fog_threshold_at_80_percent(self):
        """79% humidity → no fog; 85% humidity → fog modifies image."""
        from render.weather_overlay import render_fog
        img = Image.new("RGB", (50, 50), (100, 100, 100))

        # Below threshold: 79% humidity
        result_below = render_fog(img, 79.0)
        original_data = list(img.convert("RGBA").tobytes())
        below_data = list(result_below.tobytes())
        assert original_data == below_data, (
            "Image should be unchanged at 79% humidity"
        )

        # Above threshold: 85% humidity
        result_above = render_fog(img, 85.0)
        above_data = list(result_above.tobytes())
        assert original_data != above_data, (
            "Image should be modified at 85% humidity"
        )
