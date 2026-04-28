"""Visual regression tests using local Ollama + moondream vision model.

These tests render Moonshot images with known parameters and use the
vision model to verify expected visual features are present.

Skipped automatically if Ollama / moondream is not available.
"""

import pytest

from tests.conftest import (
    ask_model,
    ask_with_retry,
    check_yes,
    extract_choice,
    ollama_not_ready,
    render_image,
)

pytestmark = ollama_not_ready


# ── V1: Moon Visibility ─────────────────────────────────────
class TestMoonVisibility:
    """V1: At FOV=10, the moon should be clearly visible."""

    def test_moon_visible(self):
        path = render_image(
            "v1_moon.png",
            lat=39.77,
            lon=-86.16,
            date="2026-04-27",
            time="21:00",
            fov=10,
        )
        response = ask_with_retry(
            path,
            "Is there a visible moon or crescent in this image? Answer YES or NO.",
        )
        assert check_yes(response), (
            f"Moon not detected at FOV=10.\n"
            f"Model response: {response!r}"
        )


# ── V2: Horizon ───────────────────────────────────────────────
class TestHorizon:
    """V2: Horizon silhouette should always be present."""

    def test_horizon_present(self):
        path = render_image("v2_horizon.png")
        response = ask_with_retry(
            path,
            "Is there a dark horizon or terrain at the bottom of this image? Answer YES or NO.",
        )
        assert check_yes(response), (
            f"Horizon not detected.\n"
            f"Model response: {response!r}"
        )


# ── V3: Sky Type (Night/Twilight) ─────────────────────────────
class TestSkyType:
    """V3: 9pm EDT should produce twilight or night sky."""

    def test_sky_is_night_or_twilight(self):
        path = render_image("v3_skytype.png", time="21:00")
        prompt = (
            "What best describes the sky color in the upper portion of this image? "
            "Answer with one word: DAYTIME, TWILIGHT, or NIGHT."
        )
        response = ask_with_retry(path, prompt)
        match = extract_choice(response, {"TWILIGHT", "NIGHT"})
        assert match is not None, (
            f"Sky not identified as twilight/night at 9pm EDT.\n"
            f"Model response: {response!r}"
        )


# ── V4: Annotations ────────────────────────────────────────
class TestAnnotations:
    """V4: Data overlay text should be visible."""

    def test_annotations_present(self):
        path = render_image("v4_annotations.png")
        response = ask_with_retry(
            path,
            "Is there any visible text, numbers, or data overlay in this image? Answer YES or NO.",
        )
        assert check_yes(response), (
            f"Annotations not detected.\n"
            f"Model response: {response!r}"
        )


# ── V5: Stars ──────────────────────────────────────────────
class TestStars:
    """V5: Deep night (3am) should show stars."""

    def test_stars_visible_deep_night(self):
        path = render_image("v5_stars.png", time="03:00")
        response = ask_with_retry(
            path,
            "Are there small white dots (stars) visible in the darker parts of the sky? Answer YES or NO.",
        )
        assert check_yes(response), (
            f"Stars not detected at 3am.\n"
            f"Model response: {response!r}"
        )


# ── V6: Sky Color Sanity (Wrong Sky Detection) ─────────────
class TestWrongSkyDetection:
    """V6: Negative tests — model must detect contradictions."""

    def test_daytime_sky_at_noon(self):
        """V6a: Noon render should be identified as daytime."""
        path = render_image("v6a_noon.png", time="12:00")
        response = ask_with_retry(path, "Is this daytime sky? Answer YES or NO.")
        assert check_yes(response), (
            f"Noon sky not identified as daytime.\n"
            f"Model response: {response!r}"
        )

    def test_daytime_sky_not_at_3am(self):
        """V6b: 3am render should NOT be identified as daytime."""
        path = render_image("v6b_3am_not_daytime.png", time="03:00")
        response = ask_with_retry(path, "Is this daytime sky? Answer YES or NO.")
        assert not check_yes(response), (
            f"3am sky was incorrectly identified as daytime.\n"
            f"Model response: {response!r}"
        )

    def test_night_sky_not_at_noon(self):
        """V6c: Noon render should NOT be identified as night sky."""
        path = render_image("v6c_noon_not_night.png", time="12:00")
        response = ask_with_retry(path, "Is this a night sky? Answer YES or NO.")
        assert not check_yes(response), (
            f"Noon sky was incorrectly identified as night.\n"
            f"Model response: {response!r}"
        )
