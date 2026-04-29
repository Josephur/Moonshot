"""Visual regression tests using local Ollama + moondream vision model.

These tests render Moonshot images with known parameters and use the
vision model to verify expected visual features are present.

Skipped automatically if Ollama / moondream is not available.
"""

import pytest

from tests.conftest import (
    ask_with_retry,
    check_yes,
    extract_choice,
    ollama_not_ready,
    render_image,
    render_weather_image,
    CLEAR_SKY,
    OVERCAST,
)

pytestmark = ollama_not_ready

# ── Acceptable answer sets ───────────────────────────────────
YES_SET = {"YES"}


# ── V1: Moon Visibility ─────────────────────────────────────
class TestMoonVisibility:
    """V1: The moon should be clearly visible."""

    def test_moon_visible(self):
        """Moon visible at FOV=10."""
        path = render_image("v1_moon.png", lat=39.77, lon=-86.16,
                            date="2026-04-28", time="21:00", fov=10)
        response = ask_with_retry(path, "Is there a moon visible in this image?")
        assert check_yes(response), f"Moon not detected.\nModel: {response!r}"


# ── V2: Horizon ───────────────────────────────────────────────
class TestHorizon:
    """V2: Horizon silhouette should always be present."""

    @pytest.mark.xfail(reason="moondream can't resolve horizon/moon contrast at center")
    def test_horizon_present(self):
        """Horizon (land/terrain at bottom) should be visible."""
        path = render_image("v2_horizon.png")
        response = ask_with_retry(path, "Is there a dark ground or shoreline at the bottom of this image?")
        assert check_yes(response), f"Horizon not detected.\nModel: {response!r}"


# ── V3: Sky Type (Night/Twilight) ─────────────────────────────
class TestSkyType:
    """V3: 9pm EDT should produce twilight or night sky."""

    def test_sky_is_night_or_twilight(self):
        """Sky should be described as night/dark at 9pm."""
        path = render_image("v3_skytype.png", time="21:00")
        response = ask_with_retry(path, "Describe the sky color at the top of this image.")
        match = extract_choice(response, {"DARK", "BLACK", "BLUE", "NIGHT"})
        assert match is not None, f"Sky not night-like.\nModel: {response!r}"


# ── V4: Annotations ────────────────────────────────────────
class TestAnnotations:
    """V4: Data overlay text should be visible."""

    @pytest.mark.xfail(reason="moondream can't resolve small annotation text")
    def test_annotations_present(self):
        """Text/numbers should be visible on the image."""
        path = render_image("v4_annotations.png")
        response = ask_with_retry(path, "Is there any text or numbers written on this image?")
        assert check_yes(response), f"Annotations not detected.\nModel: {response!r}"


# ── V5: Moon Texture — Surface Detail ──────────────────────
class TestMoonTexture:
    """V5: The moon should show surface detail (craters/mares)."""

    def test_moon_has_surface_detail(self):
        """Moon surface should not be a flat solid colour — verify craters/mares visible."""
        path = render_image("v5_moon_texture.png", lat=39.77, lon=-86.16,
                            date="2026-04-28", time="21:00", fov=10)
        response = ask_with_retry(path, "Does the moon in this image have visible surface features like craters or dark spots?")
        assert check_yes(response), f"Moon surface detail not detected.\nModel: {response!r}"

    def test_moon_not_solid_color(self):
        """The moon should not appear as a uniform solid disk."""
        path = render_image("v5_moon_not_solid.png", lat=39.77, lon=-86.16,
                            date="2026-04-28", time="21:00", fov=10)
        response = ask_with_retry(path, "Is the moon just a plain solid circle with no texture or detail?")
        assert not check_yes(response), f"Moon appears as solid disk.\nModel: {response!r}"


# ── V6: Stars — Visible in Deep Night ─────────────────────
class TestStars:
    """V6: Deep night (3am) should show real stars, not random dots."""

    def test_stars_visible_deep_night(self):
        """Stars should be visible at 3am."""
        path = render_image("v6_stars.png", time="03:00")
        response = ask_with_retry(path, "Are there stars or small white dots visible in the dark sky?")
        assert check_yes(response), f"Stars not detected.\nModel: {response!r}"

    def test_stars_have_varying_brightness(self):
        """Stars should not all be the same brightness."""
        path = render_image("v6_stars_bright.png", time="03:00")
        response = ask_with_retry(path, "Are there some stars that are noticeably brighter or larger than others?")
        assert check_yes(response), f"No brightness variation in stars.\nModel: {response!r}"


# ── V7: Constellation Pattern (Real Stars) ─────────────────
class TestConstellation:
    """V7: Real star data should produce recognisable constellations."""

    @pytest.mark.xfail(reason="moondream struggles to identify specific constellations at current resolution")
    def test_star_clustering_near_orion(self):
        """Stars should cluster in patterns, not be uniformly random."""
        path = render_image("v7_star_clusters.png", time="03:00", fov=90)
        response = ask_with_retry(
            path,
            "Are the stars in this image scattered randomly, or do they form patterns "
            "and clusters like actual constellations?"
        )
        assert "PATTERN" in response.upper() or "CLUSTER" in response.upper() or "CONSTELLATION" in response.upper(), \
            f"Stars appear random.\nModel: {response!r}"

    def test_real_stars_not_random_grid(self):
        """Stars should not look like a grid or regular pattern."""
        path = render_image("v7_stars_not_grid.png", time="03:00")
        response = ask_with_retry(path, "Do the stars in this image look like a regular grid or evenly spaced pattern?")
        assert not check_yes(response), f"Stars appear as regular grid.\nModel: {response!r}"


# ── V8: Sky Color Sanity (Wrong Sky Detection) ─────────────
class TestWrongSkyDetection:
    """V8: Negative tests — model must detect contradictions."""

    def test_daytime_sky_at_noon(self):
        """Noon render should be daytime."""
        path = render_image("v8a_noon.png", time="12:00")
        response = ask_with_retry(path, "Describe the sky in this image. Is it bright and daytime or dark and night?")
        assert "DAY" in response.upper() or "BRIGHT" in response.upper(), \
            f"Noon sky not daytime.\nModel: {response!r}"

    def test_daytime_sky_not_at_3am(self):
        """3am render should NOT be daytime."""
        path = render_image("v8b_3am_not_daytime.png", time="03:00")
        response = ask_with_retry(path, "Is this a daytime or night scene?")
        assert "NIGHT" in response.upper() or "DARK" in response.upper(), \
            f"3am incorrectly called daytime.\nModel: {response!r}"

    @pytest.mark.xfail(reason="moondream sees noon scene as 'sunrise/sunset' due to warm scattering tint")
    def test_night_sky_not_at_noon(self):
        """Noon render should NOT be night."""
        path = render_image("v8c_noon_not_night.png", time="12:00")
        response = ask_with_retry(path, "Is this a daytime or night scene?")
        assert "DAY" in response.upper() or "BRIGHT" in response.upper(), \
            f"Noon incorrectly called night.\nModel: {response!r}"


# ── V9: Earthshine Effect ─────────────────────────────────
class TestEarthshine:
    """V9: The dark side of the moon should have a subtle glow (earthshine)."""

    @pytest.mark.xfail(reason="moondream cannot resolve faint earthshine on the moon's dark side at current moon sizes")
    def test_earthshine_present_crescent(self):
        """A thin crescent moon should show faint illumination on the dark side."""
        path = render_image("v9_earthshine_crescent.png", lat=39.77, lon=-86.16,
                            date="2026-04-22", time="21:00", fov=10)
        response = ask_with_retry(
            path,
            "Look at the dark part of the moon. Is there a very faint glow or "
            "faint details visible on the shadowed side?"
        )
        assert check_yes(response), f"Earthshine not detected.\nModel: {response!r}"


# ── W1: Clear Sky Visual ────────────────────────────────────
class TestClearSkyVisual:
    """W1: Clear weather (0% clouds) should have no visible clouds."""

    def test_no_clouds_visible(self):
        """Clear sky render should not show any cloud formations."""
        path = render_weather_image("w1_clear.png", CLEAR_SKY, fov=10)
        response = ask_with_retry(
            path,
            "Are there any clouds visible in this image? Answer YES or NO.",
        )
        assert not check_yes(response), (
            f"Clouds detected in clear sky render.\nModel: {response!r}"
        )


# ── W2: Overcast Visual ────────────────────────────────────
class TestOvercastVisual:
    """W2: 100% cloud cover should show visible clouds."""

    def test_clouds_visible(self):
        """Overcast render should show obvious cloud cover."""
        path = render_weather_image("w2_overcast.png", OVERCAST, fov=10)
        response = ask_with_retry(
            path,
            "Are there visible clouds or a cloudy sky in this image? Answer YES or NO.",
        )
        assert check_yes(response), (
            f"Clouds not detected in overcast render.\nModel: {response!r}"
        )


# ── W3: Weather Annotations ─────────────────────────────────
class TestWeatherAnnotations:
    """W3: Weather data annotations should be visible on the image."""

    def test_clear_sky_text_visible(self):
        """Annotation should contain 'clear sky' or descriptive weather text."""
        path = render_weather_image("w3a_annotations.png", CLEAR_SKY)
        response = ask_with_retry(path, "Is there any text or data overlay visible in this image? Answer YES or NO.")
        assert check_yes(response), (
            f"Weather annotations not detected.\nModel: {response!r}"
        )

    def test_temperature_visible(self):
        """Annotation should show a temperature value."""
        path = render_weather_image("w3b_temperature.png", CLEAR_SKY)
        response = ask_with_retry(path, "Does the image contain a temperature number (like '15°C' or '15C')? Answer YES or NO.")
        assert check_yes(response), (
            f"Temperature text not detected in annotations.\nModel: {response!r}"
        )

