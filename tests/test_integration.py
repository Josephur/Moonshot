"""End-to-end integration tests for the Moonshot project.

Tests that all modules import correctly and the CLI pipeline runs
without errors when given lat/lon arguments.
"""

import sys
import os
import importlib

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestImports:
    """All source modules should import without ImportError."""

    def test_main_imports(self):
        """All public-facing modules import successfully."""
        # Core moon modules
        import moon.timeconv
        import moon.position
        import moon.phase

        # Atmosphere modules
        import atmosphere.airmass
        import atmosphere.refraction
        import atmosphere.scattering

        # Location module
        import location.geocode
        import location.timezone

        # Weather module
        import weather.provider

        # Config
        import config

        # Main entry point
        import main

        # Check that key symbols exist
        assert hasattr(moon.timeconv, 'julian_date')
        assert hasattr(moon.timeconv, 'gmst')
        assert hasattr(moon.timeconv, 'lmst')
        assert hasattr(moon.position, 'sun_position')
        assert hasattr(moon.position, 'moon_position')
        assert hasattr(moon.position, 'moon_alt_az')
        assert hasattr(moon.phase, 'illumination')
        assert hasattr(moon.phase, 'phase_name')
        assert hasattr(atmosphere.refraction, 'refraction_correction')
        assert hasattr(atmosphere.scattering, 'moon_color_tint')
        assert hasattr(location.geocode, 'from_lat_lon')
        assert hasattr(location.geocode, 'is_valid_us_zip')
        assert hasattr(weather.provider, 'fetch_weather')
        assert hasattr(weather.provider, 'default_weather')
        assert hasattr(config, 'Config')
        assert hasattr(config, 'get_api_key')
        assert hasattr(main, 'main')
        assert hasattr(main, 'build_parser')

    def test_imports_no_errors(self):
        """All modules should import cleanly with no exceptions."""
        # Verify that all the critical ones import cleanly in sequence
        modules = [
            'moon.timeconv',
            'moon.position',
            'moon.phase',
            'atmosphere.airmass',
            'atmosphere.refraction',
            'atmosphere.scattering',
            'location.geocode',
            'location.timezone',
            'weather.provider',
            'config',
        ]
        for mod_name in modules:
            # Clear any cached import and re-import
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            mod = importlib.import_module(mod_name)
            assert mod is not None


class TestMainWithLatLon:
    """Simulate CLI execution with lat/lon arguments."""

    def test_main_with_lat_lon(self, monkeypatch):
        """Running main with --lat and --lon should produce a summary."""
        from main import main

        # Mock sys.argv
        test_args = [
            "src/main.py",
            "--lat", "39.7",
            "--lon", "-86.2",
            "--date", "2026-04-27",
            "--time", "21:00",
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        # Mock timezone_at to avoid timezonefinder dependency
        from location import timezone
        monkeypatch.setattr(timezone, 'timezone_at', lambda lat, lon: "America/New_York")

        # Run main — should not raise
        try:
            main()
        except SystemExit as e:
            # main() uses sys.exit(1) on location resolution failure,
            # but our args should resolve successfully.
            pytest.fail(f"main() exited with code {e.code}")

    def test_main_without_location(self, monkeypatch):
        """Running main without location args should exit with code 1."""
        from main import main

        test_args = ["src/main.py"]
        monkeypatch.setattr(sys, 'argv', test_args)

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_build_parser(self):
        """build_parser() should return a configured ArgumentParser."""
        from main import build_parser
        parser = build_parser()
        assert parser is not None
        assert parser.prog is not None

        # Verify expected arguments exist
        args = parser.parse_args([
            "--lat", "39.7", "--lon", "-86.2",
            "--date", "2026-04-27", "--time", "21:00",
        ])
        assert args.lat == pytest.approx(39.7)
        assert args.lon == pytest.approx(-86.2)
        assert args.date == "2026-04-27"
        assert args.time == "21:00"
