"""
Real star rendering pipeline for Moonshot.

Loads the filtered HYG Database v3 (Vmag < 8.0) and renders stars onto
the sky background using a fully vectorised numpy pipeline:

  1. Load catalog (lazy cached)
  1b. Convert RA from hours to degrees (HYG convention: ×15)
  2. Proper-motion correction (J2000 → observation epoch)
  3. Precession (IAU 1976, J2000.0 → observation epoch)
  4. Equatorial → horizontal (vectorised)
  5. Filter: altitude > 0°, within FOV, brightest N stars
  6. Atmospheric extinction
  7. Pixel mapping
  8. Rasterisation with magnitude-aware size and B-V colour
"""

from __future__ import annotations

__all__ = [
    "render_stars_to_sky",
    "load_catalog",
    "bv_to_rgb",
]

import csv
import gzip
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_J2000_JD = 2451545.0
_MAX_STARS = 2000


# ---------------------------------------------------------------------------
# Lazy-loaded catalog
# ---------------------------------------------------------------------------

_CATALOG: Optional[dict] = None

_CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "hyg_v3_mag8.csv.gz",
)


def load_catalog() -> dict:
    """Load (and cache) the filtered HYG catalog.

    Returns a dict with numpy arrays:
        id, ra, dec, mag, ci, pmra, pmdec
    all as float64 arrays.

    Note: RA is stored in decimal hours (0-24) per HYG convention.
    Callers must multiply by 15 to convert to degrees before
    passing to coordinate transforms.
    """
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    ids: list[float] = []
    ra: list[float] = []
    dec: list[float] = []
    mag: list[float] = []
    ci: list[float] = []
    pmra: list[float] = []
    pmdec: list[float] = []

    with gzip.open(_CATALOG_PATH, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(float(row["id"]))
            ra.append(float(row["ra"]))
            dec.append(float(row["dec"]))
            mag.append(float(row["mag"]))
            ci.append(float(row["ci"]) if row["ci"].strip() else np.nan)
            pmra.append(float(row["pmra"]) if row["pmra"].strip() else 0.0)
            pmdec.append(float(row["pmdec"]) if row["pmdec"].strip() else 0.0)

    _CATALOG = {
        "id": np.array(ids, dtype=np.float64),
        "ra": np.array(ra, dtype=np.float64),
        "dec": np.array(dec, dtype=np.float64),
        "mag": np.array(mag, dtype=np.float64),
        "ci": np.array(ci, dtype=np.float64),
        "pmra": np.array(pmra, dtype=np.float64),
        "pmdec": np.array(pmdec, dtype=np.float64),
    }
    return _CATALOG


# ---------------------------------------------------------------------------
# Proper motion
# ---------------------------------------------------------------------------


def proper_motion(ra_deg: np.ndarray, dec_deg: np.ndarray,
                  pmra: np.ndarray, pmdec: np.ndarray,
                  jd: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply proper-motion correction from J2000.0 to the given Julian Date.

    Proper motions are in milliarcseconds per year.
    Returns corrected (ra, dec) in degrees.
    """
    delta_t = (jd - _J2000_JD) / 365.25  # years from J2000.0

    dec_rad = np.radians(dec_deg)
    cos_dec = np.cos(dec_rad)

    # ΔRA = pmra * ΔT / (3600000 × cos(Dec))  [degrees]
    # ΔDec = pmdec * ΔT / 3600000              [degrees]
    dra = pmra * delta_t / (3600000.0 * cos_dec)
    ddec = pmdec * delta_t / 3600000.0

    return ra_deg + dra, dec_deg + ddec


# ---------------------------------------------------------------------------
# Precession (IAU 1976)
# ---------------------------------------------------------------------------


def _precession_angles(T: float) -> Tuple[float, float, float]:
    """Compute IAU 1976 precession parameters zeta, z, theta (arcseconds)."""
    zeta = (2306.2181 * T + 0.30188 * T ** 2 + 0.017998 * T ** 3)
    z = (2306.2181 * T + 1.09468 * T ** 2 + 0.018203 * T ** 3)
    theta = (2004.3109 * T - 0.42665 * T ** 2 - 0.041833 * T ** 3)
    return zeta, z, theta


def precess_j2000_to_epoch(ra_deg: np.ndarray, dec_deg: np.ndarray,
                           jd: float) -> Tuple[np.ndarray, np.ndarray]:
    """Precess equatorial coordinates from J2000.0 to observation epoch.

    Uses the IAU 1976 precession model with the standard three-rotation
    matrix::

        R = Rz(−z) × Ry(θ) × Rz(−ζ)

    where Rz(α) is a rotation about the z-axis and Ry(θ) about the y-axis.

    Args:
        ra_deg: Right ascension in degrees (J2000.0).
        dec_deg: Declination in degrees (J2000.0).
        jd: Target Julian Date.

    Returns:
        (ra_epoch_deg, dec_epoch_deg) — precessed coordinates in degrees.
    """
    T = (jd - _J2000_JD) / 36525.0  # Julian centuries from J2000.0

    if abs(T) < 1e-10:
        return ra_deg.copy(), dec_deg.copy()

    zeta_arcsec, z_arcsec, theta_arcsec = _precession_angles(T)
    zeta = np.radians(zeta_arcsec / 3600.0)
    z = np.radians(z_arcsec / 3600.0)
    theta = np.radians(theta_arcsec / 3600.0)

    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    # Convert to Cartesian unit vectors
    cos_dec = np.cos(dec)
    x0 = cos_dec * np.cos(ra)
    y0 = cos_dec * np.sin(ra)
    z0 = np.sin(dec)

    # Rz(-ζ): rotate about z by -ζ
    cos_zeta = np.cos(-zeta)
    sin_zeta = np.sin(-zeta)
    x1 = x0 * cos_zeta - y0 * sin_zeta
    y1 = x0 * sin_zeta + y0 * cos_zeta
    z1 = z0

    # Ry(θ): rotate about y by θ
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x2 = x1 * cos_theta + z1 * sin_theta
    z2 = -x1 * sin_theta + z1 * cos_theta
    y2 = y1

    # Rz(-z): rotate about z by -z
    cos_z = np.cos(-z)
    sin_z = np.sin(-z)
    x3 = x2 * cos_z - y2 * sin_z
    y3 = x2 * sin_z + y2 * cos_z
    z3 = z2

    # Convert back to equatorial
    ra_out = np.degrees(np.arctan2(y3, x3)) % 360.0
    dec_out = np.degrees(np.arcsin(z3))

    return ra_out, dec_out


# ---------------------------------------------------------------------------
# Equatorial → Horizontal (vectorised)
# ---------------------------------------------------------------------------


def equatorial_to_horizontal_vectorised(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    jd: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert equatorial to horizontal (alt, az) — vectorised over arrays.

    Args:
        ra_deg: Right ascension(s) in degrees.
        dec_deg: Declination(s) in degrees.
        lat_deg: Observer latitude in degrees (north positive).
        lon_deg: Observer longitude in degrees (east positive).
        jd: Julian Date.

    Returns:
        (altitude_deg, azimuth_deg) — both as numpy arrays in degrees.
    """
    # Compute Local Sidereal Time (vectorised)
    gmst_hours = 18.697374558 + 24.06570982441908 * (jd - _J2000_JD)
    gmst_hours = gmst_hours % 24.0
    lst_hours = (gmst_hours + lon_deg / 15.0) % 24.0
    lst_deg = lst_hours * 15.0

    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    lat = np.radians(lat_deg)
    ha = np.radians(lst_deg - ra_deg)  # hour angle

    # Altitude
    alt = np.arcsin(
        np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    )

    # Azimuth (measured from north through east)
    az = np.arctan2(
        -np.sin(ha),
        np.tan(dec) * np.cos(lat) - np.sin(lat) * np.cos(ha),
    )

    alt_deg = np.degrees(alt)
    az_deg = np.degrees(az) % 360.0

    return alt_deg, az_deg


# ---------------------------------------------------------------------------
# Atmospheric extinction
# ---------------------------------------------------------------------------


def _airmass(alt_deg: np.ndarray) -> np.ndarray:
    """Compute approximate airmass for given altitudes in degrees.

    Uses the simple 1/sin(h) model with a cap at 10 airmasses
    (corresponding to ~5.7° altitude) to prevent unrealistic
    extinction at very low altitudes.
    """
    alt_rad = np.radians(np.maximum(alt_deg, 0.1))
    am = 1.0 / np.sin(alt_rad)
    return np.clip(am, 1.0, 10.0)


def apply_extinction(mag: np.ndarray, alt_deg: np.ndarray) -> np.ndarray:
    """Apply atmospheric extinction to apparent magnitudes."""
    return mag + 0.28 * _airmass(alt_deg)


# ---------------------------------------------------------------------------
# B-V → RGB colour conversion
# ---------------------------------------------------------------------------


def bv_to_rgb(bv: np.ndarray) -> np.ndarray:
    """Convert B-V colour index to RGB values (0–255).

    Uses linear interpolation over spectral-type colour control points.
    NaN values are mapped to white (255, 255, 255).
    """
    table = np.array([
        (-0.45, 155, 175, 255),   # O-type: blue
        (-0.30, 175, 195, 255),   # B-type
        (-0.15, 200, 210, 250),   # late B / early A
        ( 0.00, 220, 225, 245),   # A-type: white-blue
        ( 0.15, 235, 230, 230),   # F-type: white
        ( 0.30, 245, 235, 210),   # late F / early G
        ( 0.45, 250, 240, 185),   # G-type: yellow
        ( 0.60, 250, 235, 155),   # early K
        ( 0.80, 245, 210, 120),   # K-type: orange
        ( 1.00, 240, 180,  90),   # late K / early M
        ( 1.30, 230, 150,  80),   # M-type: red
        ( 2.00, 200, 100,  70),   # very red (carbon stars etc)
    ])
    bv_pts = table[:, 0]

    r = np.interp(bv, bv_pts, table[:, 1], left=table[0, 1], right=table[-1, 1])
    g = np.interp(bv, bv_pts, table[:, 2], left=table[0, 2], right=table[-1, 2])
    b = np.interp(bv, bv_pts, table[:, 3], left=table[0, 3], right=table[-1, 3])

    # NaN → white
    nan_mask = np.isnan(bv)
    r = np.where(nan_mask, 255.0, r)
    g = np.where(nan_mask, 255.0, g)
    b = np.where(nan_mask, 255.0, b)

    rgb = np.stack([
        np.clip(np.round(r), 0, 255),
        np.clip(np.round(g), 0, 255),
        np.clip(np.round(b), 0, 255),
    ], axis=1).astype(np.uint8)

    return rgb


# ---------------------------------------------------------------------------
# Star properties from magnitude
# ---------------------------------------------------------------------------


def _star_size(mag: np.ndarray) -> np.ndarray:
    """Map apparent magnitude to star pixel size."""
    s = np.ones_like(mag, dtype=np.int32)
    s[mag < 0.5] = 3
    s[(mag >= 0.5) & (mag < 2.5)] = 2
    return s


def _star_intensity(mag: np.ndarray) -> np.ndarray:
    """Compute rendering intensity (0–1) from magnitude using Pogson's law.

    mag=0 → 1.0, mag=5 → ~0.01.
    """
    return np.clip(2.512 ** (-mag) * 1.2, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Pixel mapping
# ---------------------------------------------------------------------------


def _map_to_pixels(alt_deg: np.ndarray, az_deg: np.ndarray,
                   width: int, height: int, fov_deg: float,
                   az_center: float = 0.0,
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map (alt, az) to pixel (x, y). Returns (px_x, px_y, visible_mask).

    ``az_center`` is the azimuth (in degrees) that maps to the centre
    of the image.  Default: 0 (north).
    """
    # Vertical: alt=0 → bottom, alt=fov → top
    px_y = height * (1.0 - alt_deg / fov_deg)
    px_y = np.round(px_y).astype(np.int32)

    # Horizontal: centre on az_center
    hfov = fov_deg * width / height
    az_off = (az_deg - az_center + 180.0) % 360.0 - 180.0
    px_x = width // 2 + (az_off / hfov * width)
    px_x = np.round(px_x).astype(np.int32)

    visible = (
        (alt_deg > 0.0) &
        (alt_deg <= fov_deg) &
        (px_x >= 0) & (px_x < width) &
        (px_y >= 0) & (px_y < height)
    )
    return px_x, px_y, visible


# ---------------------------------------------------------------------------
# Main rendering entry point
# ---------------------------------------------------------------------------


def render_stars_to_sky(
    sky_image: Image.Image,
    lat: float,
    lon: float,
    jd: float,
    fov_deg: float,
) -> Image.Image:
    """Render real stars from the HYG catalog onto a sky background image.

    The full pipeline:

    1. Load catalog (lazy cached)
    1b. Convert RA from decimal hours to degrees (×15)
    2. Proper-motion correct RA/Dec from J2000.0 to observation epoch
    3. Precess coordinates from J2000.0 to observation epoch
    4. Convert equatorial to horizontal (altitude, azimuth)
    5. Filter: altitude > 0°, within vertical FOV, pick brightest stars
    6. Apply atmospheric extinction
    7. Map to pixel coordinates
    8. Render stars with magnitude-aware size and B-V colour

    Args:
        sky_image: Sky background image (RGB) to draw stars onto.
        lat: Observer latitude in decimal degrees.
        lon: Observer longitude in decimal degrees.
        jd: Julian Date of observation.
        fov_deg: Vertical field-of-view in degrees.

    Returns:
        Updated sky image with stars rendered.
    """
    w, h = sky_image.size
    pixels = np.array(sky_image, dtype=np.float32)

    # 1. Load catalog
    cat = load_catalog()
    N = len(cat["ra"])

    # HYG catalog stores RA in decimal hours (0-24); convert to degrees (0-360)
    # before passing to proper-motion, precession, and coordinate transforms.
    ra_deg = cat["ra"] * 15.0

    # 2. Proper motion
    ra_pm, dec_pm = proper_motion(ra_deg, cat["dec"],
                                  cat["pmra"], cat["pmdec"], jd)

    # 3. Precession J2000.0 → observation epoch
    ra_ep, dec_ep = precess_j2000_to_epoch(ra_pm, dec_pm, jd)

    # 4. Equatorial → horizontal (vectorised)
    alt, az = equatorial_to_horizontal_vectorised(ra_ep, dec_ep, lat, lon, jd)

    # 5. Filter: above horizon & in FOV
    keep = (alt > 0.0) & (alt <= fov_deg)

    if keep.sum() == 0:
        return sky_image

    vis_mag = cat["mag"][keep]
    vis_alt = alt[keep]
    vis_az = az[keep]
    vis_ci = cat["ci"][keep]

    # Sort by brightness (ascending mag = brighter), take top N
    order = np.argsort(vis_mag)[:_MAX_STARS]

    final_mag = vis_mag[order]
    final_alt = vis_alt[order]
    final_az = vis_az[order]
    final_ci = vis_ci[order]

    # 6. Extinction
    ext_mag = apply_extinction(final_mag, final_alt)

    # 7. Pixel mapping — centre azimuth on the median of the brightest
    #    third of visible stars so the densest star field is in-frame.
    n_for_center = max(1, len(final_az) // 3)
    top_third_az = final_az[np.argsort(final_mag)[:n_for_center]]
    # Use circular median (handle 0/360 wrap)
    az_sin = np.sin(np.radians(top_third_az))
    az_cos = np.cos(np.radians(top_third_az))
    az_center = float(np.degrees(np.arctan2(az_sin.mean(), az_cos.mean())) % 360.0)

    px_x, px_y, px_vis = _map_to_pixels(
        final_alt, final_az, w, h, fov_deg, az_center=az_center,
    )

    star_ok = px_vis
    if star_ok.sum() == 0:
        return sky_image

    ext_mag = ext_mag[star_ok]
    final_ci = final_ci[star_ok]
    xs = px_x[star_ok]
    ys = px_y[star_ok]

    # 8. Render stars
    sizes = _star_size(ext_mag)
    intensities = _star_intensity(ext_mag)
    colors = bv_to_rgb(final_ci)  # (N, 3)

    # Draw stars directly onto pixels (simple additive approach).
    # Colors are stored pre-multiplied; the overlay alpha tracks
    # total star intensity per pixel to avoid over-brightening.
    for i in range(len(xs)):
        cx, cy = int(xs[i]), int(ys[i])
        sz = int(sizes[i])
        intensity = float(intensities[i])

        r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])

        if sz == 1:
            x0, y0 = cx, cy
            x1, y1 = cx + 1, cy + 1
        else:
            half = sz // 2
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(w, cx - half + sz)
            y1 = min(h, cy - half + sz)

        # Direct pixel blending (avoids alpha-multiply traps)
        dst = pixels[y0:y1, x0:x1]
        col_arr = np.array([r, g, b], dtype=np.float32)
        blend = intensity * 0.7
        pixels[y0:y1, x0:x1] = dst * (1.0 - blend) + col_arr * blend

    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")
