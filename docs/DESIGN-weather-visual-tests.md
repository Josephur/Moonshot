# Weather Visual Test Design

## Overview

See [Issue #15](https://github.com/Josephur/Moonshot/issues/15) for full context.

## Test Strategy — Two Channels

**Channel A (Visual — moondream):** Test only extremes — clear sky vs. overcast — plus text-based annotation verification.

**Channel B (Unit — no moondream):** Test all non-visual weather data flow, intermediate cloud levels, haze, and fog via direct pixel comparison and function-level mocks.

## Test Classes

| ID | Class | Type | What it Tests |
|---|---|---|---|
| W1 | TestClearSkyVisual | moondream | Clear sky has no visible clouds |
| W2 | TestOvercastVisual | moondream | Overcast clouds visible, may obscure moon |
| W3 | TestWeatherAnnotations | moondream | Text like "clear sky" and temperature visible |
| W4 | TestWeatherDataPipeline | unit | Weather data flows correctly to render pipeline |
| W5 | TestWeatherOverlayUnits | unit | Direct pixel-level checks on overlay functions |

## Mock Weather Data

Use `render_weather_image()` helper in `conftest.py` that calls `generate_moon_image()` directly with a known `WeatherData`, bypassing CLI and real API. Predefined scenarios:

- `CLEAR_SKY` — 0% clouds, 10km visibility
- `OVERCAST` — 100% clouds, 5km visibility, 90% humidity
- `LIGHT_CLOUDS` — 30% clouds, 10km visibility
- `FOGGY` — 0% clouds, 1km visibility, 95% humidity
- `HAZY` — 30% clouds, 3km visibility, 70% humidity

## Files to Create/Modify

| File | Action |
|------|--------|
| `tests/conftest.py` | Add `render_weather_image()` helper |
| `tests/test_visual.py` | Add W1-W3 test classes |
| `tests/test_weather.py` | Add W4-W5 test classes |
| `VISUAL_TEST_ARCH.md` | Add weather test methodology |
