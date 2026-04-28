# Visual Test Architecture

## Overview

This document describes how to use a **local Ollama + moondream** vision model to automatically verify Moonshot rendered images contain expected visual features. The vision model acts as a programmatic "eyeball" — it does not compare against a reference image (though that could be layered in later), but answers targeted yes/no/multiple-choice questions about image content.

---

## 1. Test Design Patterns

Moondream is a 1.76B-parameter vision-language model. It works well for simple visual Q&A but is not a fine-grained classifier. Design every test with these constraints in mind.

### Pattern A: Binary Approval (recommended for most tests)

Ask a concrete yes/no question. Moondream tends to answer "Yes" or "No" succinctly when prompted directly.

**Template:**
```
Is/Are there <visible feature> in this image? Answer YES or NO.
```

**Examples:**
- "Is there a visible moon or crescent in this image? Answer YES or NO."
- "Are there stars visible in the sky? Answer YES or NO."
- "Is there any text or data overlay in this image? Answer YES or NO."

**Why this works:** The explicit instruction "Answer YES or NO." suppresses free-form narration. Moondream complies reliably with binary choices when they use a short, simple question with an explicit answer format constraint.

### Pattern B: Multiple-Choice Attribute (for gradations)

For attributes that aren't binary (sky color, moon phase), give a closed set of choices and ask for one.

**Template:**
```
What color is the <region>? Choose one: <option A>, <option B>, <option C>.
```

**Examples:**
- "What color is the sky at the top of the image? Choose one: deep blue, black, twilight orange, daytime blue."
- "Is the moon in this image: full, gibbous, quarter, crescent, or not visible?"

**Why this works:** Closed sets reduce hallucination. Moondream will pick from the list instead of inventing its own values. Keep options to 5 or fewer.

### Pattern C: Contradiction Check (for wrong-sky detection)

Generate an image with deliberately wrong parameters (e.g., daytime params when expecting night) and ask the opposite question. The model MUST detect the contradiction.

**Template:**
Generate image A (should be night) → ask binary: "Is this a night sky?"
Generate image B (daytime params) → ask: "Is this a daytime sky?"
Then invert: generate daytime scene, ask "Is this a night sky?" → expected NO.

### Anti-Patterns (what to avoid)

| Anti-Pattern | Why |
|---|---|
| Open-ended questions ("Describe what you see") | Non-deterministic, hard to parse |
| Counting questions ("How many stars?") | Moondream is not accurate at counting |
| Negative framing ("Is there no moon?") | Creates double-negatives, confuses model |
| Multiple questions in one prompt | Model may skip some; ask one question per API call |
| Long context before the question | Keep the prompt under 200 tokens total |

---

## 2. Pass / Fail Criteria

### Tier 1: Exact Keyword Match (on YES/NO prompts)

For binary questions with the "Answer YES or NO" instruction:

| Response contains | Interpretation | Action |
|---|---|---|
| `"YES"` (uppercase) | Feature detected → **PASS** | Accept |
| `"NO"` (uppercase) | Feature not detected → **FAIL** | Log response |
| `"yes"` / `"no"` (lowercase) | Same, but model didn't follow instruction → still match | Accept via `upper()` |
| Neither | Model diverged; log raw response | **FAIL** with diagnostic |

**Implementation:** Always call `.upper()` on the response. Then `"YES" in response.upper()`. This catches both "YES." and "I see YES a moon is there."

### Tier 2: Keyword Set Match (on multiple-choice prompts)

For multiple-choice questions, extract the chosen value and compare against **any** acceptable answer:

| Prompt choice | Acceptable values |
|---|---|
| Sky color | `{"DEEP BLUE", "BLACK", "TWILIGHT ORANGE"}` |
| Moon phase | `{"FULL", "GIBBOUS", "QUARTER", "CRESCENT"}` |

**Implementation:**
```python
def _extract_choice(response: str, choices: set[str]) -> str | None:
    """Return the first choice found in the response text."""
    upper = response.upper()
    for choice in choices:
        if choice in upper:
            return choice
    return None
```

### Tier 3: Fuzzy Failure (graceful degradation)

If the model returns nonsense / hallucinated content, the test should **FAIL** but include the full response in the error message so a human can inspect.

```python
assert detected is not None, (
    f"Model response did not contain any expected answer.\n"
    f"Prompt: {prompt!r}\n"
    f"Response: {response!r}"
)
```

### Expected Answer Sets Per Test

| Test ID | Test Name | Expected Answer(s) |
|---|---|---|
| V1 | Moon visibility | `"YES"` |
| V2 | Horizon present | `"YES"` |
| V3 | Night sky color | `"DEEP BLUE"`, `"BLACK"`, `"TWILIGHT ORANGE"` |
| V4 | Annotations present | `"YES"` |
| V5 | Stars visible (deep night) | `"YES"` |
| V6 | Wrong sky detection | Varies — see §6 |

---

## 3. Handling Model Non-Determinism

Moondream does not always emit a perfect "YES" / the exact expected word. Handle this with multiple strategies in order of preference.

### Strategy 1: Prefix/Substring Match (default)

Always check `value.upper() in response.upper()` rather than `response == "YES"`. This catches:
- `"Yes."`
- `"yes, there is a moon"`
- `"The answer is YES"`
- `"YES, I see it"`

### Strategy 2: Strict Fallback After Insertion

If no expected value is found, re-ask with a corrected prompt that adds an explicit instruction on the **second line**:

```python
if not _response_matches(response, expected):
    # Retry once with stricter prompt
    retry_prompt = prompt + "\n" + "Only write YES or NO. Do not write anything else."
    response = _ask_model(image_path, retry_prompt)
```

### Strategy 3: Two-Test Majority

For flaky tests (especially V3 sky color, V5 stars), run the same prompt **3 times** and pass if ≥2 of 3 match. Only do this for high-flake tests, not all of them, to keep test runtime acceptable.

```python
@pytest.mark.flaky(reruns=0)  # manual retry
def _majority_vote(image_path, prompt, expected_set, n_trials=3):
    votes = []
    for _ in range(n_trials):
        resp = _ask_model(image_path, prompt)
        votes.append(_extract_choice(resp, expected_set))
    return max(set(votes), key=votes.count)
```

### Strategy 4: Logging for Debuggability

Every test must log the **raw model response** so failures are diagnosable without re-running.

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Model response: %s", response)
```

### Decision Tree

```
Raw response from model
├── Contains expected keyword (case-insensitive) → PASS
├── Does not contain expected keyword
│   ├── Repeat with strict prompt → contains keyword? → PASS
│   ├── Repeat with strict prompt → still no keyword? → FAIL (log response)
│   └── (For sky-color only) → majority vote → ≥2/3 match? → PASS : FAIL
└── Empty response → FAIL (Ollama error or timeout)
```

---

## 4. Graceful Skip When Ollama Is Unavailable

Ollama may not be running on every developer machine, CI runner, or production deployment. Tests must **skip**, not fail.

### Health Check Pattern

Check the Ollama health endpoint before any test runs. If it is unreachable, mark all visual tests as skipped with a clear message.

```python
# tests/conftest.py

import pytest
import requests


def ollama_available(model="moondream") -> bool:
    """Return True if Ollama is running and the given model is pulled."""
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        models = {m["name"].split(":")[0] for m in r.json().get("models", [])}
        return model in models
    except (requests.ConnectionError, requests.Timeout):
        return False


def pytest_configure(config):
    """Register the custom marker."""
    config.addinivalue_line("markers", "visual: mark test as requiring Ollama + moondream vision model")


# Global skip condition — evaluated at collection time
_ollama_ready = ollama_available()
ollama_not_ready = pytest.mark.skipif(
    not _ollama_ready,
    reason="Ollama/moondream not available (start with: ollama serve && ollama pull moondream)"
)
```

### Usage

```python
@ollama_not_ready
def test_moon_visible():
    ...
```

### Message to User on Skip

When skipped, pytest prints:

```
tests/test_visual.py::test_moon_visible SKIPPED (Ollama/moondream not available)
```

Include a `pytest.skip` suggestion pointing them to the fix:

```
Run:  ollama pull moondream
      ollama serve   # (stays running in background)
Then: pytest tests/test_visual.py -v
```

### Alternative: Module-Level Skip

If all tests in `test_visual.py` need Ollama, put the skip at the top of the file:

```python
import pytest
from conftest import ollama_not_ready

pytestmark = ollama_not_ready  # applied to every test in this module
```

---

## 5. pytest Integration

### File Layout

```
Moonshot/
  tests/
    __init__.py
    conftest.py                  ← Ollama health check + shared fixtures
    test_visual.py               ← ALL visual regression tests
    test_geocode.py              (existing)
    test_integration.py          (existing)
    ...
  output/                        ← rendered test images land here
  src/
    ...
```

### Fixtures (in `conftest.py`)

```python
import os
import subprocess
import base64
import logging
import pytest
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
MOONSHOT_DIR = Path(__file__).resolve().parents[1]
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OUTPUT_DIR = MOONSHOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Ollama availability ─────────────────────────────────────
@pytest.fixture(scope="session")
def ollama_available():
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        models = {m["name"].split(":")[0] for m in r.json().get("models", [])}
        return "moondream" in models
    except (requests.ConnectionError, requests.Timeout):
        return False


# ── Image rendering helper ──────────────────────────────────
def render_image(output_name: str, extra_args: list[str] | None = None) -> Path:
    """Run Moonshot CLI and return the output image path."""
    out_path = OUTPUT_DIR / output_name
    cmd = [
        "python3", "-m", "src.main",
        "--lat", "39.77",
        "--lon", "-86.16",
        "--date", "2026-04-27",
        "--time", "21:00",
        "--output", str(out_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, cwd=MOONSHOT_DIR, capture_output=True, check=True, text=True)
    return out_path


# ── Vision model helper ─────────────────────────────────────
def ask_model(image_path: Path, prompt: str) -> str:
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


def check_yes(response: str) -> bool:
    """Flexible YES/NO check."""
    return "YES" in response.upper() and "NO" not in response.upper()


def extract_choice(response: str, choices: set[str]) -> str | None:
    """Return first matching choice from response (case-insensitive)."""
    upper = response.upper()
    for choice in choices:
        if choice in upper:
            return choice
    return None


# ── Retry with strict fallback ────────────────────────────
def ask_with_fallback(image_path: Path, prompt: str, expected_set: set[str], strict_suffix: str | None = None) -> str | None:
    """Ask model, retry once with strict instruction if first answer doesn't match."""
    strict_suffix = strict_suffix or "Only write YES or NO. Do not write anything else."

    response = ask_model(image_path, prompt)
    match = extract_choice(response, expected_set)
    if match is not None:
        return match

    # Retry with stricter instruction
    response2 = ask_model(image_path, prompt + "\n" + strict_suffix)
    return extract_choice(response2, expected_set)
```

### Marker Registration

In `conftest.py` or `pyproject.toml`:

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
markers = [
    "visual: Tests requiring Ollama + moondream vision model for visual regression detection.",
]
```

### Running Visual Tests

```bash
# All visual tests
pytest tests/test_visual.py -v

# Single test with debug logging
pytest tests/test_visual.py::test_moon_visible -v --log-cli-level=INFO

# Skip visual tests entirely
pytest tests/ -v --ignore=tests/test_visual.py

# Or via marker
pytest tests/ -v -m "not visual"
```

### CI Integration

In CI, visual tests should be a separate job:

```yaml
visual-tests:
  steps:
    - name: Install Ollama
      run: |
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull moondream &
    - name: Run visual tests
      run: pytest tests/test_visual.py -v --junitxml=report-visual.xml
```

---

## 6. Complete Test Suite

Below is the full spec for each test. Barnaby should implement these exactly.

### V1 — Moon Visibility

- **Args:** `--fov 10` (narrow field of view ensures moon is large enough)
- **Prompt:** `"Is there a visible moon or crescent in this image? Answer YES or NO."`
- **Expected:** `"YES"`
- **Flakiness:** Low (moon is large at FOV=10)
- **Fallback:** One retry with stricter prompt

### V2 — Horizon Present

- **Args:** default (broad FOV with horizon)
- **Prompt:** `"Is there a dark horizon or terrain visible at the bottom of this image? Answer YES or NO."`
- **Expected:** `"YES"`
- **Flakiness:** Very low (almost every Moonshot image has a horizon)
- **Fallback:** One retry with stricter prompt

### V3 — Night/Twilight Sky Color

- **Args:** `--time 21:00` (9pm EDT, twilight/night)
- **Prompt:** `"What color is the sky at the top of this image? Choose one: deep blue, black, twilight orange, daytime blue."`
- **Expected:** One of `{"DEEP BLUE", "BLACK", "TWILIGHT ORANGE"}`
- **Flakiness:** **Moderate.** Moondream sometimes mislabels sky color.
- **Fallback:** Majority vote over 3 attempts. Pass if ≥2/3 match an acceptable color.

### V4 — Annotations Present

- **Args:** default
- **Prompt:** `"Is there any text or data overlay visible in this image? Answer YES or NO."`
- **Expected:** `"YES"`
- **Flakiness:** Low (text overlay is obvious to the model)
- **Fallback:** One retry with stricter prompt
- **Note:** If annotations are behind `--no-annotations`, this test will fail. That's intentional — skip or invert when that flag is used.

### V5 — Stars Visible (Deep Night)

- **Args:** `--time 03:00` (deep night)
- **Prompt:** `"Are there stars visible in the sky in this image? Answer YES or NO."`
- **Expected:** `"YES"`
- **Flakiness:** **Moderate-High.** Moondream may miss faint stars against a dark background. If the render uses very faint star points at default brightness, the model may not see them.
- **Mitigation:** Increase star brightness in the renderer for test purposes (optional), or use majority vote.
- **Fallback:** Majority vote over 3 attempts. Also log star count from the renderer metadata (if available) to cross-reference.

### V6 — Wrong Sky Detection (Daytime Sky When It Should Be Night)

This is the **negative test** — it proves the model can detect a contradiction.

**Test V6a — Daytime parameters produce daytime sky:**
- **Args:** `--time 14:00` (2pm EDT, broad daylight)
- **Prompt:** `"What color is the sky at the top of this image? Choose one: deep blue, black, twilight orange, daytime blue."`
- **Expected:** `"DAYTIME BLUE"`
- **Goal:** Verify the renderer correctly produces a light blue daytime sky.

**Test V6b — Daytime sky *rejected* when night expected:**
- **Args:** Same as V6a (daylight render)
- **Prompt:** `"Is this a night sky? Answer YES or NO."`
- **Expected:** `"NO"`
- **Goal:** Model recognizes that a bright daytime-blue sky is *not* night.

**Test V6c — Night sky *rejected* when daytime expected:**
- **Args:** `--time 03:00` (deep night render)
- **Prompt:** `"Is this a daytime sky? Answer YES or NO."`
- **Expected:** `"NO"`
- **Goal:** Model recognizes that a dark starry sky is *not* daytime.

**Test V6d — Wrong date produces wrong sky:**
- **Args:** `--time 14:00` for a high-latitude winter location where the sun is actually below the horizon at 2pm (e.g., `--lat 65.0 --lon -150.0 --date 2026-01-15 --time 14:00`)
- **Prompt:** `"What color is the sky at the top of this image? Choose one: deep blue, black, twilight orange, daytime blue."`
- **Expected:** `"DEEP BLUE"` or `"BLACK"` or `"TWILIGHT ORANGE"` (not "daytime blue" — polar winter afternoon is dark)
- **Goal:** Verify location/season produce correct sky type, not just time of day.

**Why V6 is important:** This is the strongest signal that the render pipeline (atmosphere model, scattering, sun position) is working correctly. A render that always produces "twilight blue" regardless of time/location would pass V3 but fail V6.

---

## 7. Test Code Structure (for Barnaby)

```python
"""Visual regression tests using local Ollama + moondream vision model.

These tests render Moonshot images with known parameters and use the
vision model to verify expected visual features are present.

Skipped automatically if Ollama / moondream is not available.
"""

import pytest

from conftest import (
    MOONSHOT_DIR, OUTPUT_DIR, render_image, ask_model,
    ask_with_fallback, check_yes, extract_choice,
    ollama_not_ready,
)

pytestmark = ollama_not_ready

# ── Acceptable answer sets ───────────────────────────────────
NIGHT_SKY_COLORS = {"DEEP BLUE", "BLACK", "TWILIGHT ORANGE"}
DAY_SKY_COLORS = {"DAYTIME BLUE"}
YES_SET = {"YES"}
NO_SET = {"NO"}


# ── V1: Moon Visibility ─────────────────────────────────────
class TestMoonVisibility:
    def test_moon_visible(self):
        path = render_image("v1_moon.png", ["--fov", "10"])
        match = ask_with_fallback(path, "Is there a visible moon or crescent in this image? Answer YES or NO.", YES_SET)
        assert match == "YES", f"Moon not detected. Raw response logged above."


# ── V2: Horizon ───────────────────────────────────────────────
class TestHorizon:
    def test_horizon_present(self):
        path = render_image("v2_horizon.png")
        match = ask_with_fallback(path, "Is there a dark horizon or terrain visible at the bottom of this image? Answer YES or NO.", YES_SET)
        assert match == "YES", f"Horizon not detected."


# ── V3: Night Sky Color ────────────────────────────────────
class TestNightSkyColor:
    @pytest.mark.flaky(reruns=0)  # handled via majority vote
    def test_sky_is_night(self):
        path = render_image("v3_skycolor.png", ["--time", "21:00"])
        # Majority vote: 3 attempts, pass if ≥2 match
        votes = []
        for _ in range(3):
            resp = ask_model(path, "What color is the sky at the top of this image? Choose one: deep blue, black, twilight orange, daytime blue.")
            choice = extract_choice(resp, NIGHT_SKY_COLORS | DAY_SKY_COLORS)
            votes.append(choice)
        night_votes = sum(1 for v in votes if v in NIGHT_SKY_COLORS)
        assert night_votes >= 2, (
            f"Sky not identified as night in ≥2/3 attempts. "
            f"Votes: {votes}"
        )


# ── V4: Annotations ────────────────────────────────────────
class TestAnnotations:
    def test_annotations_present(self):
        path = render_image("v4_annotations.png")
        match = ask_with_fallback(path, "Is there any text or data overlay visible in this image? Answer YES or NO.", YES_SET)
        assert match == "YES", f"Annotations not detected."


# ── V5: Stars ──────────────────────────────────────────────
class TestStars:
    @pytest.mark.flaky(reruns=0)
    def test_stars_visible_deep_night(self):
        path = render_image("v5_stars.png", ["--time", "03:00"])
        votes = []
        for _ in range(3):
            resp = ask_model(path, "Are there stars visible in the sky in this image? Answer YES or NO.")
            choice = extract_choice(resp, YES_SET | NO_SET)
            votes.append(choice)
        yes_votes = sum(1 for v in votes if v == "YES")
        assert yes_votes >= 2, (
            f"Stars not detected in ≥2/3 attempts. "
            f"Votes: {votes}"
        )


# ── V6: Wrong Sky Detection ────────────────────────────────
class TestWrongSkyDetection:
    def test_daytime_produces_blue_sky(self):
        """Daylight render should produce daytime blue sky."""
        path = render_image("v6a_daytime.png", ["--time", "14:00"])
        match = ask_with_fallback(path, "What color is the sky at the top of this image? Choose one: deep blue, black, twilight orange, daytime blue.", DAY_SKY_COLORS | NIGHT_SKY_COLORS)
        assert match in DAY_SKY_COLORS, f"Daytime render produced night sky: {match}"

    def test_daytime_sky_not_night(self):
        """Daylight render should NOT be identified as night sky."""
        path = render_image("v6b_day_not_night.png", ["--time", "14:00"])
        match = ask_with_fallback(path, "Is this a night sky? Answer YES or NO.", {"NO"})
        assert match == "NO", f"Daytime render was identified as night sky."

    def test_night_sky_not_daytime(self):
        """Night render should NOT be identified as daytime sky."""
        path = render_image("v6c_night_not_day.png", ["--time", "03:00"])
        match = ask_with_fallback(path, "Is this a daytime sky? Answer YES or NO.", {"NO"})
        assert match == "NO", f"Night render was identified as daytime sky."

    def test_polar_winter_afternoon_is_dark(self):
        """High-latitude winter afternoon should produce dark sky, not daytime blue."""
        path = render_image("v6d_polar_winter.png", [
            "--lat", "65.0", "--lon", "-150.0",
            "--date", "2026-01-15", "--time", "14:00",
        ])
        votes = []
        for _ in range(3):
            resp = ask_model(path, "What color is the sky at the top of this image? Choose one: deep blue, black, twilight orange, daytime blue.")
            choice = extract_choice(resp, NIGHT_SKY_COLORS | DAY_SKY_COLORS)
            votes.append(choice)
        night_votes = sum(1 for v in votes if v in NIGHT_SKY_COLORS)
        assert night_votes >= 2, (
            f"Polar winter afternoon sky should be dark, got: {votes}"
        )
```

---

## 8. Summary of Trade-offs

| Concern | Decision | Rationale |
|---|---|---|
| **Prompt format** | Explicit "Answer YES or NO" / "Choose one:" suffix | Forces deterministic outputs from a non-deterministic model |
| **Flakiness** | Acceptable on V3/V5 via majority vote | Vision models are not 100% reliable on subtle features (sky gradient, faint stars) |
| **Test speed** | ~5-10s per prompt (model inference) | Expect ~1 min for full suite (8 tests × ~8s avg) |
| **False positives** | Low for YES/NO, moderate for sky-color | V6 contradictions catch render pipeline bugs more reliably than V3 alone |
| **Maintenance** | Low — only needs updating if prompt patterns change | New tests follow the same templates |
| **Hardware** | Needs Ollama + moondream (4GB VRAM or CPU 8GB RAM) | Skip on machines without it; no hard dependency |
