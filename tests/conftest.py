"""Shared fixtures and helpers for Moonshot tests."""

import base64
import logging
import subprocess
from pathlib import Path

import pytest
import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
MOONSHOT_DIR = Path(__file__).resolve().parents[1]
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OUTPUT_DIR = MOONSHOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Ollama availability check ─────────────────────────────────
def ollama_available() -> bool:
    """Return True if Ollama is running and moondream model is available."""
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        models = {m["name"].split(":")[0] for m in r.json().get("models", [])}
        return "moondream" in models
    except (requests.ConnectionError, requests.Timeout):
        return False


# Module-level skipif condition — evaluated at collection time
_ollama_ready = ollama_available()
ollama_not_ready = pytest.mark.skipif(
    not _ollama_ready,
    reason="Ollama/moondream not available (run: ollama pull moondream && ollama serve)",
)


# ── Vision model helpers ──────────────────────────────────────
def ask_model(image_path: str | Path, prompt: str) -> str:
    """Send an image to local moondream and return the raw response text."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": "moondream",
        "prompt": prompt,
        "images": [b64],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    text = resp.json().get("response", "")
    logger.info("Model prompt: %s", prompt)
    logger.info("Model response: %s", text)
    return text


def ask_with_retry(image_path: str | Path, prompt: str, retries: int = 2) -> str:
    """Ask the model, retrying on empty response."""
    for attempt in range(retries + 1):
        response = ask_model(image_path, prompt)
        if response.strip():
            return response
        logger.warning("Empty response on attempt %d/%d", attempt + 1, retries + 1)
    return response


def check_yes(response: str) -> bool:
    """Return True if response contains YES (case-insensitive)."""
    return "YES" in response.upper()


def extract_choice(response: str, options: set[str]) -> str | None:
    """Return the first option found in the response (case-insensitive)."""
    upper = response.upper()
    for option in options:
        if option in upper:
            return option
    return None


# ── Image rendering helper ────────────────────────────────────
def render_image(
    output_name: str,
    lat: float | None = None,
    lon: float | None = None,
    city: str | None = None,
    state: str | None = None,
    date: str | None = None,
    time: str | None = None,
    fov: float | None = None,
) -> Path:
    """Run Moonshot CLI and return the output image path.

    By default uses lat/lon for Indianapolis. Override with keyword args.
    """
    out_path = OUTPUT_DIR / output_name
    cmd = ["python3", "-m", "src.main", "--output", str(out_path)]

    if city and state:
        cmd.extend(["--city", city, "--state", state])
    else:
        cmd.extend([
            "--lat", str(lat if lat is not None else 39.77),
            "--lon", str(lon if lon is not None else -86.16),
        ])

    if date:
        cmd.extend(["--date", date])
    if time:
        cmd.extend(["--time", time])
    if fov is not None:
        cmd.extend(["--fov", str(fov)])

    logger.info("Rendering: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=MOONSHOT_DIR, capture_output=True, check=True, text=True)
    return out_path
