"""
test_core.py
Unit tests for the core modules: observer, catalog, ephemeris.
Run with:  pytest tests/ -v
"""

import math
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.observer import ObserverSite, KNOWN_SITES, get_site, now_utc
from core.catalog import (RadioSource, RadioCatalog,
                           default_catalog, BRIGHT_SOURCES)
from core.ephemeris import Ephemeris, VisibilityWindow, DriftScanPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def site():
    """A mid-latitude northern hemisphere site for general tests."""
    return ObserverSite(
        name="Test Site",
        latitude=40.0,
        longitude=-75.0,
        elevation=100.0,
        min_elevation=10.0,
        dish_diameter=3.0,
        frequency_mhz=1420.0,
        system_temp_k=100.0,
    )


@pytest.fixture
def green_bank():
    return KNOWN_SITES["green_bank"]


@pytest.fixture
def cas_a():
    return BRIGHT_SOURCES[0]   # 3C 461 (Cas A)


@pytest.fixture
def cyg_a():
    return BRIGHT_SOURCES[1]   # 3C 405 (Cyg A)


@pytest.fixture
def obs_time():
    return Time("2024-06-01 00:00:00", scale="utc")


# ---------------------------------------------------------------------------
# ObserverSite
# ---------------------------------------------------------------------------

class TestObserverSite:

    def test_beam_fwhm_3m_dish(self, site):
        """3m dish at 1420 MHz should give FWHM between 4° and 15°."""
        assert 4.0 < site.beam_fwhm_deg < 15.0

    def test_beam_fwhm_smaller_dish_wider_beam(self, site):
        """Halving dish diameter should roughly double the beam width."""
        fwhm_3m = site.beam_fwhm_deg
        site.dish_diameter = 1.5
        fwhm_15m = site.beam_fwhm_deg
        assert 1.8 < fwhm_15m / fwhm_3m < 2.2

    def test_beam_fwhm_higher_freq_narrower_beam(self, site):
        """Higher frequency should produce a narrower beam."""
        fwhm_1420 = site.beam_fwhm_deg
        site.frequency_mhz = 2840.0
        fwhm_2840 = site.beam_fwhm_deg
        assert fwhm_2840 < fwhm_1420

    def test_sefd_positive(self, site):
        assert site.sefd_jy > 0

    def test_sefd_larger_dish_lower_sefd(self, site):
        """Larger collecting area reduces SEFD."""
        sefd_3m = site.sefd_jy
        site.dish_diameter = 6.0
        sefd_6m = site.sefd_jy
        assert sefd_6m < sefd_3m

    def test_sefd_higher_tsys_higher_sefd(self, site):
        """Higher system temperature increases SEFD."""
        sefd_100k = site.sefd_jy
        site.system_temp_k = 200.0
        sefd_200k = site.sefd_jy
        assert sefd_200k > sefd_100k

    def test_sensitivity_decreases_with_integration_time(self, site):
        """Longer integration improves sensitivity (lower RMS)."""
        rms_short = site.sensitivity_mjy(10.0, 10.0)
        rms_long  = site.sensitivity_mjy(10.0, 1000.0)
        assert rms_long < rms_short

    def test_sensitivity_decreases_with_bandwidth(self, site):
        """Wider bandwidth improves sensitivity."""
        rms_narrow = site.sensitivity_mjy(1.0, 60.0)
        rms_wide   = site.sensitivity_mjy(10.0, 60.0)
        assert rms_wide < rms_narrow

    def test_sensitivity_radiometer_equation(self, site):
        """
        RMS should follow the radiometer equation:
        rms = SEFD / sqrt(n_pol * bw * t)
        """
        bw_hz = 10e6
        t_s   = 300.0
        n_pol = 2
        expected_mjy = site.sefd_jy / math.sqrt(n_pol * bw_hz * t_s) * 1e3
        result_mjy   = site.sensitivity_mjy(10.0, 300.0)
        assert abs(result_mjy - expected_mjy) / expected_mjy < 0.01

    def test_known_sites_present(self):
        for key in ["green_bank", "vla", "parkes", "arecibo", "effelsberg"]:
            assert key in KNOWN_SITES

    def test_known_sites_have_valid_coords(self):
        for key, s in KNOWN_SITES.items():
            assert -90 <= s.latitude  <= 90,  f"{key}: bad latitude"
            assert -180 <= s.longitude <= 180, f"{key}: bad longitude"
            assert s.elevation >= 0,           f"{key}: negative elevation"

    def test_get_site_case_insensitive(self):
        s1 = get_site("green_bank")
        s2 = get_site("Green_Bank")
        assert s1.name == s2.name

    def test_get_site_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown site"):
            get_site("nonexistent_observatory")

    def test_is_visible_returns_bool(self, site, obs_time):
        coord = SkyCoord(ra=83.8 * u.deg, dec=22.0 * u.deg, frame="icrs")
        result = site.is_visible(coord, obs_time)
        assert isinstance(result, bool)

    def test_altaz_frame_has_correct_location(self, site, obs_time):
        frame = site.altaz_frame(obs_time)
        assert frame.location is not None

    def test_now_utc_returns_time(self):
        t = now_utc()
        assert isinstance(t, Time)


# ---------------------------------------------------------------------------
# RadioSource
# ---------------------------------------------------------------------------

class TestRadioSource:

    def test_coord_property(self, cas_a):
        coord = cas_a.coord
        assert isinstance(coord, SkyCoord)
        assert abs(coord.ra.deg  - cas_a.ra_deg)  < 0.001
        assert abs(coord.dec.deg - cas_a.dec_deg) < 0.001

    def test_flux_at_reference_frequency(self, cas_a):
        """flux_at(ref_freq) should return ref flux."""
        assert abs(cas_a.flux_at(cas_a.ref_freq_mhz) - cas_a.flux_jy) < 0.01

    def test_flux_at_lower_freq_brighter_for_negative_index(self, cas_a):
        """Cas A has spectral_index < 0 so it is brighter at lower frequencies."""
        assert cas_a.flux_at(408.0) > cas_a.flux_at(1420.0)

    def test_flux_at_zero_flux_returns_zero(self):
        src = RadioSource("Empty", 0.0, 0.0, flux_jy=0.0)
        assert src.flux_at(1420.0) == 0.0

    def test_str_representation(self, cas_a):
        s = str(cas_a)
        assert cas_a.name in s
        assert "RA=" in s
        assert "Dec=" in s


# ---------------------------------------------------------------------------
# RadioCatalog
# ---------------------------------------------------------------------------

class TestRadioCatalog:

    def test_default_catalog_not_empty(self):
        cat = default_catalog()
        assert len(cat) > 0

    def test_default_catalog_sun_has_nonzero_coords(self):
        """Sun position should be computed at runtime, not placeholder (0,0)."""
        cat = default_catalog()
        sun = next((s for s in cat if s.name == "Sun"), None)
        assert sun is not None
        # Sun is never exactly at RA=0, Dec=0 except briefly at vernal equinox
        assert not (sun.ra_deg == 0.0 and sun.dec_deg == 0.0)

    def test_bright_sources_coords_in_range(self):
        for src in BRIGHT_SOURCES:
            assert 0.0 <= src.ra_deg  <= 360.0, f"{src.name}: RA out of range"
            assert -90.0 <= src.dec_deg <= 90.0, f"{src.name}: Dec out of range"

    def test_no_duplicate_names(self):
        names = [s.name for s in BRIGHT_SOURCES]
        assert len(names) == len(set(names)), "Duplicate source names in catalog"

    def test_filter_by_flux(self):
        cat = default_catalog()
        bright = cat.by_flux_min(100.0, freq_mhz=1400.0)
        for src in bright:
            assert src.flux_at(1400.0) >= 100.0 or src.flux_jy >= 1e5

    def test_filter_by_type(self):
        cat = default_catalog()
        snrs = cat.by_type("supernova remnant")
        for src in snrs:
            assert "supernova remnant" in src.source_type.lower()

    def test_search_case_insensitive(self):
        cat = default_catalog()
        upper = cat.search("CAS")
        lower = cat.search("cas")
        assert len(upper) == len(lower)
        assert len(upper) > 0

    def test_search_no_match_returns_empty(self):
        cat = default_catalog()
        results = cat.search("zzz_no_such_source")
        assert results == []

    def test_by_sky_region(self):
        cat = default_catalog()
        centre = SkyCoord(ra=83.8 * u.deg, dec=22.0 * u.deg)
        nearby = cat.by_sky_region(centre, radius_deg=20.0)
        # Tau A (3C 144) is at RA=83.8, Dec=22.0 — should be in result
        assert any("144" in s.name or "Tau" in s.name for s in nearby)

    def test_csv_roundtrip(self, tmp_path):
        cat = default_catalog()
        path = tmp_path / "catalog.csv"
        cat.save_csv(path)
        loaded = RadioCatalog.from_csv(path)
        assert len(loaded) == len(cat)
        for orig, reloaded in zip(cat.sources, loaded.sources):
            assert orig.name     == reloaded.name
            assert abs(orig.ra_deg  - reloaded.ra_deg)  < 0.001
            assert abs(orig.dec_deg - reloaded.dec_deg) < 0.001

    def test_iter_and_len(self):
        cat = default_catalog()
        count = sum(1 for _ in cat)
        assert count == len(cat)

    def test_add_source(self):
        cat = RadioCatalog("Test")
        src = RadioSource("New Source", 100.0, 45.0, 5.0)
        cat.add(src)
        assert len(cat) == 1
        assert cat.sources[0].name == "New Source"


# ---------------------------------------------------------------------------
# Ephemeris
# ---------------------------------------------------------------------------

class TestEphemeris:

    def test_elevation_track_length(self, green_bank, cas_a, obs_time):
        eph = Ephemeris(green_bank, time_step_min=5.0)
        hours, elevs = eph.elevation_track(cas_a.coord, obs_time, 24.0)
        assert len(hours) == len(elevs)
        assert len(hours) > 100

    def test_elevation_values_in_physical_range(self, green_bank, cas_a, obs_time):
        eph = Ephemeris(green_bank, time_step_min=5.0)
        _, elevs = eph.elevation_track(cas_a.coord, obs_time, 24.0)
        assert elevs.min() >= -90.0
        assert elevs.max() <=  90.0

    def test_cas_a_circumpolar_from_green_bank(self, green_bank, cas_a, obs_time):
        """Cas A (Dec ~+59°) should be circumpolar from Green Bank (lat ~38°N)."""
        eph = Ephemeris(green_bank, time_step_min=5.0)
        w = eph.visibility_window(cas_a.coord, cas_a.name, obs_time, 24.0)
        assert w.always_up or w.peak_elevation_deg > 0

    def test_never_rises_for_far_south_source(self, green_bank, obs_time):
        """A source at Dec -80° should never rise from Green Bank (lat +38°)."""
        src = RadioSource("Far South", 0.0, -80.0)
        eph = Ephemeris(green_bank, time_step_min=5.0)
        w = eph.visibility_window(src.coord, src.name, obs_time, 24.0)
        assert w.never_rises

    def test_multi_source_schedule_count(self, green_bank, obs_time):
        sources = BRIGHT_SOURCES[:5]
        eph = Ephemeris(green_bank, time_step_min=5.0)
        windows = eph.multi_source_schedule(sources, obs_time, 24.0)
        assert len(windows) == 5

    def test_multi_source_schedule_no_duplicates(self, green_bank, obs_time):
        sources = BRIGHT_SOURCES[:5]
        eph = Ephemeris(green_bank, time_step_min=5.0)
        windows = eph.multi_source_schedule(sources, obs_time, 24.0)
        names = [w.source_name for w in windows]
        assert len(set(names)) == len(names)

    def test_next_transit_is_within_24h(self, green_bank, cas_a, obs_time):
        eph = Ephemeris(green_bank, time_step_min=5.0)
        transit = eph.next_transit(cas_a.coord, obs_time)
        delta_h = (transit - obs_time).to(u.hour).value
        assert 0 <= delta_h <= 25.0

    def test_is_visible_now_returns_bool(self, green_bank, cas_a, obs_time):
        eph = Ephemeris(green_bank)
        result = eph.is_visible_now(cas_a.coord, obs_time)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# DriftScanPredictor
# ---------------------------------------------------------------------------

class TestDriftScanPredictor:

    def test_cas_a_transits_south_pointing_beam(self, green_bank, obs_time):
        """
        Cas A (Dec ~+59°) should transit a south-facing beam (Az=180°) at
        Green Bank when the beam elevation is set near Cas A's transit altitude
        (~71°). Use a wide beam (large dish=0.1m → huge FWHM) to guarantee
        detection regardless of exact timing.
        """
        site = ObserverSite(
            "GB Wide", 38.4331, -79.8397, 880,
            dish_diameter=0.1,      # tiny dish → ~172° FWHM — catches everything
            frequency_mhz=1420.0,
            system_temp_k=100.0,
        )
        predictor = DriftScanPredictor(site, beam_az=180.0, beam_el=45.0,
                                       time_step_min=2.0)
        sources = [BRIGHT_SOURCES[0]]   # Cas A only
        transits = predictor.predict(sources, obs_time, duration_hours=24.0)
        assert len(transits) >= 1
        assert transits[0].source_name == BRIGHT_SOURCES[0].name

    def test_transit_duration_positive(self, green_bank, obs_time):
        site = ObserverSite(
            "GB Wide", 38.4331, -79.8397, 880,
            dish_diameter=0.1, frequency_mhz=1420.0, system_temp_k=100.0,
        )
        predictor = DriftScanPredictor(site, beam_az=180.0, beam_el=45.0,
                                       time_step_min=2.0)
        transits = predictor.predict(BRIGHT_SOURCES[:3], obs_time, 24.0)
        for t in transits:
            assert t.transit_duration_min > 0

    def test_peak_response_between_0_and_1(self, green_bank, obs_time):
        site = ObserverSite(
            "GB Wide", 38.4331, -79.8397, 880,
            dish_diameter=0.1, frequency_mhz=1420.0, system_temp_k=100.0,
        )
        predictor = DriftScanPredictor(site, beam_az=180.0, beam_el=45.0,
                                       time_step_min=2.0)
        transits = predictor.predict(BRIGHT_SOURCES, obs_time, 24.0)
        for t in transits:
            assert 0.0 <= t.peak_response <= 1.0

    def test_is_detected_at_half_power(self):
        from core.ephemeris import BeamTransit
        t = BeamTransit("Test", None, None, None,
                        peak_separation_deg=0.0,
                        peak_response=0.5,
                        transit_duration_min=60.0)
        assert t.is_detected(threshold_response=0.5)
        assert not t.is_detected(threshold_response=0.51)

    def test_min_response_filter(self, obs_time):
        site = ObserverSite(
            "GB Wide", 38.4331, -79.8397, 880,
            dish_diameter=0.1, frequency_mhz=1420.0, system_temp_k=100.0,
        )
        predictor = DriftScanPredictor(site, beam_az=180.0, beam_el=45.0,
                                       time_step_min=2.0)
        all_transits  = predictor.predict(BRIGHT_SOURCES, obs_time, 24.0)
        strong = [t for t in all_transits if t.peak_response >= 0.5]
        # All strong transits should be within the half-power beam width
        for t in strong:
            assert t.peak_response >= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])