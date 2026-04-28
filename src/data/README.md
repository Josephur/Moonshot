# Star Catalog Data

## File: `hyg_v3_mag8.csv.gz`

**Provenance:** Filtered subset of the HYG Database v3.7 (Hipparcos, Yale Bright Star, Gliese merged catalog).

- **Source:** <https://astronexus.com/downloads/catalogs/hygdata_v37.csv.gz>
- **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Original catalog description:** <https://astronexus.com/projects/hyg>

### Filter applied

- Stars with `mag` (V-band magnitude) < 8.0
- Excluded id=0 (the Sun, mag=-26.7)
- Result: **41,148 stars** (~840 KB gzipped)

### Column descriptions

| Column  | Description                                                      |
|---------|------------------------------------------------------------------|
| `id`    | HYG sequential star identifier                                   |
| `ra`    | Right ascension (J2000.0), decimal degrees                       |
| `dec`   | Declination (J2000.0), decimal degrees                           |
| `mag`   | V-band apparent magnitude                                        |
| `ci`    | Color index (B–V) — may be empty for some stars                  |
| `pmra`  | Proper motion in right ascension (mas/yr)                        |
| `pmdec` | Proper motion in declination (mas/yr)                            |

### Usage

This file is loaded by `src/render/stars.py` at render time via lazy
cached numpy arrays.  See that module for the coordinate pipeline:
proper motion correction → precession → equatorial-to-horizontal →
filtering → extinction → pixel mapping → rendering.
