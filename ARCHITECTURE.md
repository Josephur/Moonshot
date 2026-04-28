# Moonshot — Architecture Document

## Overview

Moonshot generates scientifically-accurate PNG images of the moon as it would appear from any US location. It combines astronomical calculations, atmospheric physics, and current weather data to produce realistic horizon-to-sky renderings.

---

## 1. Project Structure

```
Moonshot/
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── main.py                  # CLI entry point
│   ├── moon/
│   │   ├── __init__.py
│   │   ├── position.py          # Moon altitude/azimuth calculations
│   │   ├── phase.py             # Moon phase, illumination, terminator
│   │   └── timeconv.py          # Julian date, sidereal time conversions
│   ├── atmosphere/
│   │   ├── __init__.py
│   │   ├── refraction.py        # Atmospheric refraction correction
│   │   ├── scattering.py        # Rayleigh & Mie scattering, hue calc
│   │   └── airmass.py           # Air mass calculations
│   ├── location/
│   │   ├── __init__.py
│   │   ├── geocode.py           # ZIP/City/State → lat/lon
│   │   └── timezone.py          # Timezone resolution from coords
│   ├── weather/
│   │   ├── __init__.py
│   │   └── provider.py          # Weather API client
│   ├── render/
│   │   ├── __init__.py
│   │   ├── sky.py               # Sky gradient generation
│   │   ├── moon_render.py       # Moon disk with phase rendering
│   │   ├── horizon.py           # Horizon line / terrain
│   │   ├── weather_overlay.py   # Clouds, haze, fog
│   │   ├── annotations.py       # Text/data overlay
│   │   └── composite.py         # Assembles final image
│   └── config.py                # Configuration handling
├── tests/
│   ├── __init__.py
│   ├── test_position.py
│   ├── test_phase.py
│   ├── test_timeconv.py
│   ├── test_refraction.py
│   ├── test_scattering.py
│   ├── test_airmass.py
│   ├── test_geocode.py
│   ├── test_render.py
│   └── test_integration.py
└── output/                      # Generated images land here
```

---

## 2. Module Breakdown

### 2.1 `moon.position` — Moon Position Calculator

**Responsibility:** Calculate the moon's apparent altitude and azimuth for a given observer location and time.

**Algorithm:**
- Compute Julian Date from Gregorian date/time
- Compute Greenwich Sidereal Time (GST) and Local Sidereal Time (LST)
- Compute moon ecliptic longitude, latitude, and distance using the ELP2000-82 analytical lunar ephemeris (or simplified: the low-precision IAU SOFA algorithm from the Astronomical Almanac)
- Convert ecliptic → equatorial coordinates (RA, Dec) accounting for obliquity of the ecliptic
- Convert equatorial → horizontal coordinates (altitude, azimuth) using hour angle
- Apply atmospheric refraction correction to altitude (delegates to `atmosphere.refraction`)

**Key Functions:**
- `julian_date(year, month, day, hour, minute, second) → float`
- `gmst(jd) → float`  — Greenwich Mean Sidereal Time in hours
- `lmst(gmst, longitude) → float` — Local Sidereal Time
- `sun_position(jd) → (ra_sun, dec_sun)` — Sun position for phase calculation
- `moon_position(jd) → (ra, dec, distance)` — Moon RA/Dec/distance
- `moon_alt_az(lat, lon, jd) → (altitude, azimuth)` — Topocentric horizontal coords
- `moon_apparent_alt_az(lat, lon, jd, temp_c, pressure_mbar) → (altitude, azimuth)` — Including refraction

### 2.2 `moon.phase` — Moon Phase Calculator

**Responsibility:** Calculate the moon's illumination percentage and phase name.

**Algorithm:**
- Compute the sun-moon-observer angle (phase angle) using the dot product of sun and moon position vectors
- Illumination fraction = (1 - cos(phase_angle)) / 2
- Determine phase name from illumination:
  - 0-1%: New Moon
  - 1-30%: Waxing/Waning Crescent
  - 30-50%: First/Last Quarter (interpolated)
  - 50-70%: Waxing/Waning Gibbous
  - 70-99%: Waxing/Waning Gibbous
  - 99-100%: Full Moon
- Terminator angle = position angle of the moon's bright limb

**Key Functions:**
- `illumination(sun_pos, moon_pos) → float`  — Fraction 0.0-1.0
- `phase_name(illumination, waxing) → str`
- `terminator_angle(sun_pos, moon_pos) → float` — Degrees from vertical
- `is_waxing(sun_ra, moon_ra) → bool`

### 2.3 `moon.timeconv` — Time Conversions

**Responsibility:** Time/date astronomy helpers.

**Key Functions:**
- `julian_date(y, m, d, h, mi, s) → float` — Julian Day Number
- `jd_to_mjd(jd) → float` — Modified Julian Date
- `gmst_from_jd(jd) → float` — Greenwich Mean Sidereal Time
- `lmst_from_jd(jd, lon_deg) → float` — Local Sidereal Time
- `delta_t(year) → float` — ΔT approximation (difference between TT and UT1)

### 2.4 `atmosphere.refraction` — Atmospheric Refraction

**Responsibility:** Correct apparent altitude of celestial objects for atmospheric refraction.

**Algorithm:** Use the Saemundsson (1986) formula for astronomical refraction:
- R = 1.02 / tan(h + 10.3/(h + 5.11))  (for altitude h in degrees, R in arcminutes)
- Apply temperature/pressure correction factor: P/(273+T) * 283/1013
- Works down to -5° altitude (atmospheric extinction becomes dominant below that)

**Key Functions:**
- `refraction_correction(apparent_altitude_deg, temp_c, pressure_mbar) → float` — Correction in degrees
- `true_from_apparent(alt_apparent, temp_c, pressure_mbar) → float` — Remove refraction
- `apparent_from_true(alt_true, temp_c, pressure_mbar) → float` — Add refraction

### 2.5 `atmosphere.scattering` — Atmospheric Scattering & Color

**Responsibility:** Calculate the perceived color/hue of the moon based on atmospheric conditions.

**Algorithm:**
- Compute air mass using Kasten & Young (1989) formula: X = 1/(sin(h) + 0.50572*(h+6.07995)^(-1.6364))
- Rayleigh optical depth = 0.008569 * λ^(-4) * (1 + 0.0113*λ^(-2) + 0.00013*λ^(-4)) * (P/P0)
- Mie scattering from aerosols (estimated from humidity/visibility)
- For each RGB wavelength (approx 650nm, 510nm, 475nm):
  - Calculate extinction via Beer-Lambert: I = I0 * exp(-τ * X * (P/P0))
  - Combine Rayleigh and Mie extinction
- Result: RGB tint factor — moon appears redder at low altitudes due to preferential scattering of blue

**Key Functions:**
- `air_mass(altitude_deg) → float`
- `rayleigh_optical_depth(pressure_mbar) → float`
- `mie_optical_depth(visibility_km_or_humidity) → float`
- `atmospheric_extinction_factor(altitude_deg, temp_c, pressure_mbar, humidity) → (r, g, b)` — Tint multipliers
- `moon_color_tint(altitude_deg, temp_c, pressure_mbar, humidity) → (r, g, b)` — Final color

### 2.6 `atmosphere.airmass` — Air Mass Calculator

**Responsibility:** Calculate relative air mass along the line of sight.

**Algorithm:** Kasten & Young (1989), valid down to zenith angle of ~90°.

**Key Functions:**
- `air_mass(zenith_angle_deg) → float`
- `relative_pressure(correction_altitude_m) → float` — Pressure correction for observer elevation

### 2.7 `location.geocode` — Location Resolver

**Responsibility:** Convert ZIP codes, city/state strings, or lat/lon to standardized coordinates.

**Approach:**
- Use `photon` (free Geocoding API based on OSM) or `geopy` with Nominatim
- ZIP code lookup via free Open Census Data API or Zippopotam.us
- Local fallback: bundled ZIP code → lat/lon database for offline use

**Key Functions:**
- `from_zip(zip_code) → (lat, lon, city, state)`
- `from_city_state(city, state) → (lat, lon)`
- `from_lat_lon(lat, lon) → (lat, lon)` — Validate and normalize
- `is_valid_us_zip(zip_code) → bool`

### 2.8 `location.timezone` — Timezone Resolution

**Responsibility:** Determine local timezone from lat/lon.

**Approach:** Use the `timezonefinder` library for offline timezone lookup from coordinates.

**Key Functions:**
- `timezone_at(lat, lon) → str` — e.g. "America/New_York"
- `local_time_from_utc(utc_dt, lat, lon) → datetime`

### 2.9 `weather.provider` — Weather API Client

**Responsibility:** Fetch current weather for the observed location.

**Approach:** OpenWeatherMap OneCall API (free tier).

**Output:**
- `temp_c: float`
- `pressure_mbar: float`
- `humidity: int` — percentage
- `cloud_cover: int` — percentage
- `visibility_km: float`
- `conditions: str` — description
- `weather_icon_code: str`
- `wind_speed: float`

**Key Functions:**
- `fetch_weather(lat, lon, api_key) → WeatherData`
- `WeatherData` — NamedTuple/dataclass with all fields above

### 2.10 `render.sky` — Sky Gradient Rendering

**Responsibility:** Generate the sky background with realistic color gradient.

**Algorithm:**
- Daytime: blue sky gradient based on solar altitude and Rayleigh scattering
- Sunset/twilight: sunset colors (red/orange/purple) based on solar depression angle
- Nighttime: deep blue-black gradient with LRGB color temperature shift
- If moon is present: subtle sky glow near the moon's position (atmospheric backscatter)

**Key Functions:**
- `sky_gradient(sun_altitude, moon_altitude, image_width, image_height) → Image`

### 2.11 `render.moon_render` — Moon Disk Rendering

**Responsibility:** Render the moon disk with correct phase, size, and color.

**Algorithm:**
- Calculate angular diameter of moon from distance (~0.49° to ~0.56° apparent)
- Render circle with correct number of pixels based on image FOV
- Phase rendering: dark side vs illuminated side with terminator line
- Apply color tint from atmospheric scattering calculations
- Add subtle brightness gradient (limb darkening on illuminated side)

**Key Functions:**
- `moon_size_pixels(angular_diameter, fov_deg, image_width) → int`
- `render_moon_disk(phase, terminator_angle, tint_color, pixel_radius) → Image`
- `moon_position_on_image(altitude, azimuth, viewer_direction, image_w, image_h, fov) → (x, y)`

### 2.12 `render.horizon` — Horizon Rendering

**Responsibility:** Render the horizon/landscape at the bottom.

**Algorithm:**
- Simple: flat horizon line with terrain silhouette
- Draw a flat or gently undulating horizon based on observer altitude
- Color: dark silhouette against sky, lighter if daytime
- Horizon dip: h_arcmin = 1.93 * sqrt(observer_height_m) — Earth curvature correction

**Key Functions:**
- `horizon_line(image, observer_height_m, sun_altitude) → Image`
- `horizon_dip_degrees(observer_height_m) → float`

### 2.13 `render.weather_overlay` — Weather Visual Effects

**Responsibility:** Overlay weather effects on the sky.

**Effects:**
- **Clouds:** Semi-transparent cloud layers at varying altitudes, rendered as fractals or noise-based shapes
- **Haze:** Horizontal gradient of reduced contrast near horizon
- **Rain/Snow:** Subtle vertical streaks (if applicable)
- **Fog:** Uniform reduced visibility gradient

**Key Functions:**
- `render_clouds(image, cloud_cover_pct, moon_visible) → Image`
- `render_haze(image, visibility_km, altitude_deg) → Image`

### 2.14 `render.annotations` — Data Overlay

**Responsibility:** Add data annotations to the final image.

**Content:**
- Header: Location, Date/Time
- Left panel: Moon details (phase %, altitude, azimuth)
- Right panel: Weather (temp, conditions, cloud cover)
- Footer: "Generated by Moonshot"

### 2.15 `render.composite` — Image Assembly

**Responsibility:** Orchestrate all render layers into final image.

**Data Flow:**
1. Start with sky gradient (full canvas)
2. Draw moon disk at calculated position
3. Draw horizon at bottom
4. Overlay weather effects (clouds on top of sky, in front of moon if covering)
5. Annotations on top
6. Save as PNG

### 2.16 `main.py` — CLI Entry Point

**Interface:**
```
moonshot --zip 46201 | --city "Indianapolis" --state "IN" | --lat 39.7 --lon -86.2
         [--date 2026-04-27] [--time 21:00] [--fov 90]
         [--width 1920] [--height 1080]
         [--output moon.png]
         [--weather-api-key KEY]
```

---

## 3. Data Flow

```
User Input (location, time, options)
    │
    ▼
┌─────────────────┐
│ location.geocode │──→ (lat, lon, city, state)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────┐
│ location.timezone│──→  │ Timezone-aware       │
└─────────────────┘     │ datetime object      │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
        ┌─────────────────┐ ┌──────────┐ ┌──────────────────┐
        │ moon.position    │ │ weather  │ │ moon.phase       │
        │ moon.timeconv    │ │ provider │ │ (uses sun pos)   │
        └────────┬────────┘ └─────┬────┘ └────────┬─────────┘
                 │                │               │
                 ▼                ▼               ▼
    ┌──────────────────────┐ ┌──────────┐ ┌──────────────────┐
    │ atmosphere.refraction │ │ temp,    │ │ illumination     │
    │ atmosphere.scattering │ │ pressure │ │ phase name       │
    │ atmosphere.airmass    │ │ humidity │ │ terminator angle │
    └────────┬─────────────┘ │ cloud    │ └────────┬─────────┘
             │               │ vis.     │          │
             ▼               └────┬─────┘          │
    ┌─────────────────────┐       │                │
    │ Corrected altitude  │       │                │
    │ Azimuth             │       │                │
    │ Apparent magnitude  │       │                │
    │ Color tint (RGB)    │       │                │
    └────────┬────────────┘       │                │
             │                    │                │
             ▼                    ▼                ▼
    ┌─────────────────────────────────────────────────────┐
    │                   render.composite                  │
    │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐ │
    │  │ sky  │  │ moon │  │horiz.│  │wthr  │  │annot.│ │
    │  │grad. │  │render│  │render│  │over. │  │      │ │
    │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘ │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼
                     ┌─────────────┐
                     │  output.png │
                     └─────────────┘
```

---

## 4. Dependencies

### Runtime
- `Pillow>=10.0` — Image generation
- `numpy>=1.24` — Numerical calculations
- `requests>=2.31` — HTTP client for weather & geocoding APIs
- `geopy>=2.4` — Geocoding (Nominatim/Photon)
- `timezonefinder>=6.5` — Offline timezone lookup

### Development
- `pytest>=7.0` — Testing framework
- `pytest-cov>=4.1` — Coverage reporting

### Optional (enhanced accuracy)
- `skyfield>=1.49` — Alternative high-precision ephemeris (NASA JPL)

---

## 5. Testing Strategy

- **Unit tests** for every public function in:
  - `moon.position`, `moon.phase`, `moon.timeconv`
  - `atmosphere.refraction`, `atmosphere.scattering`, `atmosphere.airmass`
  - `location.geocode`, `location.timezone`
  - `render.sky`, `render.moon_render`, `render.horizon`, `render.annotations`
- **Mocking** for weather API and geocoding API calls (no network in tests)
- **Fixture data** for known moon positions (cross-checked against JPL Horizons or USNO)
  - Test against known values for specific dates/locations
  - e.g., Full Moon on 2026-04-27, altitude at Indianapolis at 21:00 EDT
- **Image validation** tests:
  - Output is valid PNG
  - Image dimensions correct
  - Moon center is within expected pixel region
  - Colors are within valid ranges
- **Integration test** end-to-end: mock input → PNG output exists

---

## 6. Weather API Integration

- **Provider:** OpenWeatherMap OneCall 3.0 (free tier: 1000 calls/day)
- **Key management:** Read from environment variable `MOONSHOT_WEATHER_API_KEY` or `.env` file
- **Graceful degradation:** If API key not set or API unavailable, proceed with clear weather defaults (temperature: 15°C, pressure: 1013mbar, humidity: 50%, cloud: 0%, visibility: 10km)

---

## 7. Location Resolution Strategy

1. **Priority order:**
   - Direct lat/lon if provided
   - ZIP code (5 digits) → geocode lookup
   - City/State → geocode lookup
2. **Caching:** Cache geocode results in memory to reduce API calls during a session
3. **Fallback:** If all lookups fail, default to Indianapolis, IN

---

## 8. Build/Run Instructions

```bash
# Setup
cd Moonshot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python -m src.main --zip 46201

# Run with weather
export MOONSHOT_WEATHER_API_KEY=your_key
python -m src.main --city "Indianapolis" --state "IN" --date 2026-04-27 --time 21:00

# Tests
pytest tests/ -v
```
