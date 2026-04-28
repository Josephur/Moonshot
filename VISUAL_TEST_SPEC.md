# Visual Regression Testing with Local Vision Model

## Concept
Use the local Ollama + moondream vision model as an automated "eyeball" to verify rendered images contain expected visual features. This catches rendering regressions (broken compositing, missing elements, wrong colors) before images reach a human reviewer.

## How It Works
1. Generate a test image with known parameters (date, location, FOV)
2. Send it to the local Ollama API with targeted prompts
3. Parse the response to verify expected features
4. Pass/fail based on feature detection

## Test Scenarios

### Test 1: Moon Visibility
- Generate image with `--fov 10` (narrow enough for moon to be >50px)
- Prompt: "Is there a visible moon or crescent in this image? Answer YES or NO."
- Expected: YES (moon should be visible at this FOV)

### Test 2: Horizon
- Prompt: "Is there a dark horizon or terrain visible at the bottom of this image? Answer YES or NO."
- Expected: YES

### Test 3: Sky Type
- Generate image for 9pm EDT (twilight/night)
- Prompt: "What color is the sky at the top of the image? Answer with: deep blue, black, twilight orange, or daytime blue."
- Expected: "deep blue" or "twilight" or "black"

### Test 4: Annotations
- Prompt: "Is there any text or data overlay present in the image? Answer YES or NO."
- Expected: YES

### Test 5: Stars (Night/Twilight)
- Generate image for deep night (3am)
- Prompt: "Are there stars visible in the sky? Answer YES or NO."
- Expected: YES

## Implementation

### File: tests/test_visual.py

```python
import pytest
import subprocess
import base64
import json
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MOONSHOT_DIR = "/home/Brachyura/.openclaw/workspace/Moonshot"

def _ask_model(image_path, prompt):
    """Send image to local vision model and return response text."""
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(OLLAMA_URL, json={
        'model': 'moondream',
        'prompt': prompt,
        'images': [b64],
        'stream': False,
    }, timeout=120)
    return resp.json().get('response', '')

def test_moon_visible():
    """At FOV=10, the moon should be clearly visible."""
    out = f"{MOONSHOT_DIR}/output/test_visual_moon.png"
    subprocess.run([
        "python3", "-m", "src.main",
        "--lat", "39.77", "--lon", "-86.16",
        "--date", "2026-04-27", "--time", "21:00",
        "--fov", "10",
        "--output", out,
    ], cwd=MOONSHOT_DIR, check=True)
    
    response = _ask_model(out, "Is there a visible moon or crescent in this image? Answer YES or NO.")
    assert "YES" in response.upper(), f"Moon not detected: {response}"

def test_horizon_present():
    """Horizon silhouette should always be present."""
    out = f"{MOONSHOT_DIR}/output/test_visual_horizon.png"
    subprocess.run([
        "python3", "-m", "src.main",
        "--lat", "39.77", "--lon", "-86.16",
        "--output", out,
    ], cwd=MOONSHOT_DIR, check=True)
    
    response = _ask_model(out, "Is there a dark horizon or terrain at the bottom? Answer YES or NO.")
    assert "YES" in response.upper(), f"Horizon not detected: {response}"

def test_annotations_present():
    """Data overlay text should be visible."""
    out = f"{MOONSHOT_DIR}/output/test_visual_annotations.png"
    subprocess.run([
        "python3", "-m", "src.main",
        "--lat", "39.77", "--lon", "-86.16",
        "--output", out,
    ], cwd=MOONSHOT_DIR, check=True)
    
    response = _ask_model(out, "Is there any text or data overlay in this image? Answer YES or NO.")
    assert "YES" in response.upper(), f"Annotations not detected: {response}"
```

## Requirements
- Ollama service running with moondream model pulled
- Test skipped if Ollama not available (pytest.skip)

## Future Enhancements
- Reference image comparison (generate once, compare against it)
- Multi-model voting (use both moondream and another model for reliability)
- Pixel-level checks mixed with model-based checks
