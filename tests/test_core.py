"""
test_core.py
Basic unit tests for observer, catalog, and ephemeris modules.
Run with: python -m pytest tests/ -v
"""

import pytest
from astropy.time import Time
import astropy.units as u

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.observer import ObserverSite, KNOWN_SITES
from core.catalog import RadioSource, RadioCatalog, default_catalog, BRIGHT_SOURCES
from core.ephemeris import Ephemeris


# ---------------------------------------------------------------------------
# ObserverSite tests
# ---------------------------------------------------------------------------

class TestObserverSite:
    def setup_method(self):
        self.site = ObserverSite("Test", 40.0, -75.0, 100.0)

    def test_beam_fwhm(self):
        """Beam width should be positive and reasonable for a 3m dish at 1420 MHz."""
        self.site.dish_diameter = 3.0
        self.site.frequency_mhz = 1420.0
        fwhm = self.site.beam_fwhm_deg
        assert 5.0 < fwhm < 15.0, f"Unexpected FWHM: {fwhm}°"

    def test_sefd_positive(self):
        assert self.site.sefd_jy > 0

    def test_sensitivity_decreases_with_time(self):
        rms10 = self.site.sensitivity_mjy(10.0, 10.0)
        rms1000 = self.site.sensitivity_mjy(10.0, 1000.0)
        assert rms1000 < rms10

    def test_known_sites_loaded(self):
        for key in ["green_bank", "vla", "parkes"]:
            assert key in KNOWN_SITES
            s = KNOWN_SITES[key]
            assert s.latitude != 0.0

    def test_is_visible_returns_bool(self):
        from astropy.coordinates import SkyCoord
        coord = SkyCoord(ra=83.8 * u.deg, dec=22.0 * u.deg, frame="icrs")
        t = Time("2024-06-01 00:00:00", scale="utc")
        result = self.site.is_visible(coord, t)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# RadioCatalog tests
# ---------------------------------------------------------------------------

class TestRadioCatalog:
    def setup_method(self):
        self.cat = default_catalog()

    def test_default_catalog_not_empty(self):
        assert len(self.cat) > 0

    def test_bright_sources_have_coords(self):
        for src in BRIGHT_SOURCES[:5]:
            assert 0.0 <= src.ra_deg <= 360.0
            assert -90.0 <= src.dec_deg <= 90.0

    def test_flux_at_different_frequency(self):
        src = BRIGHT_SOURCES[0]  # Cas A
        f1 = src.flux_at(1000.0)
        f2 = src.flux_at(408.0)
        # At lower freq, steeper-spectrum source should be brighter
        assert f2 > f1

    def test_filter_by_flux(self):
        bright = self.cat.by_flux_min(100.0)
        for src in bright:
            assert src.flux_at(1400.0) >= 100.0 or src.flux_jy >= 100.0

    def test_search(self):
        results = self.cat.search("Cas")
        assert any("Cas" in s.name for s in results)

    def test_csv_roundtrip(self, tmp_path):
        path = tmp_path / "catalog.csv"
        self.cat.save_csv(path)
        loaded = RadioCatalog.from_csv(path)
        assert len(loaded) == len(self.cat)
        assert loaded.sources[0].name == self.cat.sources[0].name


# ---------------------------------------------------------------------------
# Ephemeris tests
# ---------------------------------------------------------------------------

class TestEphemeris:
    def setup_method(self):
        self.site = KNOWN_SITES["green_bank"]
        self.eph = Ephemeris(self.site, time_step_min=5.0)
        self.cas_a = BRIGHT_SOURCES[0]   # Cas A — circumpolar from Green Bank
        self.start = Time("2024-06-01 00:00:00", scale="utc")

    def test_elevation_track_shape(self):
        hours, elevs = self.eph.elevation_track(
            self.cas_a.coord, self.start, 24.0)
        assert len(hours) == len(elevs)
        assert len(hours) > 100

    def test_elevation_in_range(self):
        _, elevs = self.eph.elevation_track(
            self.cas_a.coord, self.start, 24.0)
        assert -90 <= elevs.min() <= 90
        assert -90 <= elevs.max() <= 90

    def test_visibility_window_cas_a(self):
        # Cas A is circumpolar from Green Bank (lat ~38°, dec ~58°)
        w = self.eph.visibility_window(
            self.cas_a.coord, self.cas_a.name, self.start, 24.0)
        assert w.peak_elevation_deg > 0
        # Should be always up or at least have a transit
        assert w.transit_time is not None or w.always_up

    def test_multi_source_schedule(self):
        sources = BRIGHT_SOURCES[:5]
        windows = self.eph.multi_source_schedule(sources, self.start, 24.0)
        assert len(windows) == 5
        # No duplicates
        names = [w.source_name for w in windows]
        assert len(set(names)) == len(names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
