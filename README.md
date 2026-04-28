# Moonshot 🌙

Generate scientifically-accurate PNG images of the moon as it would appear from **anywhere on Earth**.

## Features

- **Precise moon position** — altitude & azimuth based on IAU-standard algorithms
- **Realistic phase rendering** — illumination percentage, terminator angle, phase name
- **Real moon surface texture** — texture-mapped lunar surface from NASA LRO data with earthshine effect on the dark side
- **Real star data** — stars rendered from the HYG Database v3 (41K+ stars, precessed to observation epoch) with spectral colors and atmospheric extinction
- **Atmospheric effects** — refraction correction, Rayleigh/Mie scattering for accurate color
- **Weather integration** — current conditions affect moon visibility and sky rendering
- **Worldwide location support** — by ZIP/postal code, City/State/Country, or direct lat/lon coordinates
- **Beautiful PNG output** — configurable resolution, horizon, annotations

## Gallery

Moonshots generated on **April 28, 2026** — a Waning Gibbous moon (∼95% illuminated):

| New York City (8:30pm EDT) | Los Angeles (8:30pm PDT) | Chicago (8:30pm CDT) |
|---|---|---|
| ![NYC](https://raw.githubusercontent.com/Josephur/Moonshot/main/output/gallery/moonshot_new_york_city.png) | ![LA](https://raw.githubusercontent.com/Josephur/Moonshot/main/output/gallery/moonshot_los_angeles.png) | ![Chicago](https://raw.githubusercontent.com/Josephur/Moonshot/main/output/gallery/moonshot_chicago.png) |

| Indianapolis (9:00pm EDT) | Miami (9:00pm EDT) | Denver (8:30pm MDT) |
|---|---|---|
| ![Indianapolis](https://raw.githubusercontent.com/Josephur/Moonshot/main/output/gallery/moonshot_indianapolis.png) | ![Miami](https://raw.githubusercontent.com/Josephur/Moonshot/main/output/gallery/moonshot_miami.png) | ![Denver](https://raw.githubusercontent.com/Josephur/Moonshot/main/output/gallery/moonshot_denver.png) |

| Sydney, Australia (7:30pm AEST) | Buenos Aires, Argentina (7:30pm ART) | Cape Town, South Africa (7:30pm SAST) |
|---|---|---|
| ![Sydney](https://raw.githubusercontent.com/Josephur/Moonshot/main/output/gallery/moonshot_sydney.png) | ![Buenos Aires](https://raw.githubusercontent.com/Josephur/Moonshot/main/output/gallery/moonshot_buenos_aires.png) | ![Cape Town](https://raw.githubusercontent.com/Josephur/Moonshot/main/output/gallery/moonshot_cape_town.png) |

*Colors and moon positions vary by latitude, longitude, and atmospheric conditions.*

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate moon image for Indianapolis
python -m src.main --zip 46201 --date 2026-04-28 --time 21:00

# With weather data
export MOONSHOT_WEATHER_API_KEY=your_openweather_api_key
python -m src.main --city "Indianapolis" --state "IN" --output moon.png

# International locations
python -m src.main --city "Paris" --country "France"
python -m src.main --city "Tokyo"
python -m src.main --city "Melbourne" --state "Victoria" --country "Australia"

# Full options
python -m src.main --help
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--zip` | ZIP / postal code | — |
| `--city` | City name | — |
| `--state` | State / region / province | — |
| `--country` | Country name | — (defaults to USA if --state given) |
| `--lat` | Latitude | — |
| `--lon` | Longitude | — |
| `--date` | Date (YYYY-MM-DD) | Today |
| `--time` | Time (HH:MM, local) | Now |
| `--fov` | Field of view (degrees) | 90 |
| `--width` | Image width (px) | 1920 |
| `--height` | Image height (px) | 1080 |
| `--output` | Output filename | moon_<timestamp>.png |
| `--weather-api-key` | OpenWeatherMap API key | `MOONSHOT_WEATHER_API_KEY` env |

## Updating Gallery Images

Whenever rendering code changes, regenerate the gallery before committing:

```bash
# Northern Hemisphere (US)
python -m src.main --zip 10001 --date 2026-04-28 --time 20:30 --output output/gallery/moonshot_new_york_city.png
python -m src.main --zip 90001 --date 2026-04-28 --time 20:30 --output output/gallery/moonshot_los_angeles.png
python -m src.main --city Chicago --state IL --date 2026-04-28 --time 20:30 --output output/gallery/moonshot_chicago.png
python -m src.main --city Indianapolis --state IN --date 2026-04-28 --time 21:00 --output output/gallery/moonshot_indianapolis.png
python -m src.main --city Miami --state FL --date 2026-04-28 --time 21:00 --output output/gallery/moonshot_miami.png
python -m src.main --city Denver --state CO --date 2026-04-28 --time 20:30 --output output/gallery/moonshot_denver.png

# Southern Hemisphere
python -m src.main --city "Sydney" --country "Australia" --date 2026-04-28 --time 19:30 --output output/gallery/moonshot_sydney.png
python -m src.main --city "Buenos Aires" --country "Argentina" --date 2026-04-28 --time 19:30 --output output/gallery/moonshot_buenos_aires.png
python -m src.main --city "Cape Town" --country "South Africa" --date 2026-04-28 --time 19:30 --output output/gallery/moonshot_cape_town.png
```

## Technical Details

- **Moon position:** IAU-standard algorithms via ELP2000-82 analytical lunar ephemeris
- **Phase:** Sun-Moon-Observer angle with precise terminator
- **Moon surface:** Texture-mapped from LRO WAC data (512x256 equirectangular), with earthshine on the dark side
- **Stars:** HYG Database v3 (41K+ stars, Vmag < 8.0), precessed to observation epoch with spectral colors and atmospheric extinction
- **Refraction:** Saemundsson (1986) formula
- **Scattering:** Rayleigh & Mie with air mass via Kasten & Young (1989)
- **Weather:** OpenWeatherMap API (free tier)

## Testing

```bash
pytest tests/ -v
```

Built with ❤️ by the Crustacean Dev Squad 🦀🎯🏗️🦐🦞
