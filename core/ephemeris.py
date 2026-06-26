"""
ephemeris.py
Rise, set, transit calculations and visibility window generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u

from .observer import ObserverSite


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VisibilityWindow:
    source_name: str
    rise_time: Optional[Time]
    transit_time: Optional[Time]
    set_time: Optional[Time]
    peak_elevation_deg: float
    always_up: bool = False
    never_rises: bool = False

    def duration_hours(self) -> float:
        if self.always_up:
            return 24.0
        if self.never_rises or self.rise_time is None or self.set_time is None:
            return 0.0
        dt = (self.set_time - self.rise_time).to(u.hour).value
        return float(dt % 24)

    def __str__(self) -> str:
        if self.always_up:
            return f"{self.source_name:20s}  Always above horizon  peak={self.peak_elevation_deg:.1f}°"
        if self.never_rises:
            return f"{self.source_name:20s}  Never rises above horizon"
        rise = self.rise_time.iso[:16] if self.rise_time else "---"
        tran = self.transit_time.iso[:16] if self.transit_time else "---"
        sett = self.set_time.iso[:16] if self.set_time else "---"
        return (f"{self.source_name:20s}  rise={rise}  "
                f"transit={tran}  set={sett}  "
                f"peak={self.peak_elevation_deg:.1f}°  "
                f"up={self.duration_hours():.1f}h")


# ---------------------------------------------------------------------------
# Core ephemeris engine
# ---------------------------------------------------------------------------

class Ephemeris:
    """
    Computes rise/transit/set times and elevation tracks for radio sources.
    Uses a sampling approach for reliability across all dec values.
    """

    def __init__(self, site: ObserverSite, time_step_min: float = 2.0):
        self.site = site
        self.time_step = time_step_min / (24 * 60)  # days

    def elevation_track(
        self,
        coord: SkyCoord,
        start_time: Time,
        duration_hours: float = 24.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute elevation (degrees) at regular intervals.

        Returns
        -------
        times : np.ndarray  — UTC hours from start_time
        elevations : np.ndarray — degrees
        """
        n_steps = int(duration_hours / (self.time_step * 24))
        offsets = np.linspace(0, duration_hours / 24, n_steps)
        times = start_time + offsets * u.day
        frame = AltAz(obstime=times, location=self.site.location)
        altaz = coord.transform_to(frame)
        hours = np.linspace(0, duration_hours, n_steps)
        return hours, altaz.alt.deg

    def visibility_window(
        self,
        coord: SkyCoord,
        source_name: str,
        start_time: Time,
        duration_hours: float = 24.0,
    ) -> VisibilityWindow:
        """Find rise, transit, and set times within a time window."""
        hours, elevs = self.elevation_track(coord, start_time, duration_hours)
        min_el = self.site.min_elevation
        above = elevs >= min_el
        peak_el = float(np.max(elevs))

        if np.all(above):
            transit_idx = int(np.argmax(elevs))
            transit_t = start_time + float(hours[transit_idx]) / 24 * u.day
            return VisibilityWindow(source_name, None, transit_t, None,
                                    peak_el, always_up=True)

        if not np.any(above):
            return VisibilityWindow(source_name, None, None, None,
                                    peak_el, never_rises=True)

        # Find rise transition (False -> True)
        transitions = np.diff(above.astype(int))
        rise_idx = np.where(transitions == 1)[0]
        set_idx = np.where(transitions == -1)[0]

        rise_t = (start_time + float(hours[rise_idx[0]]) / 24 * u.day
                  if len(rise_idx) else None)
        set_t = (start_time + float(hours[set_idx[-1]]) / 24 * u.day
                 if len(set_idx) else None)

        # Transit = maximum elevation
        transit_idx = int(np.argmax(elevs))
        transit_t = start_time + float(hours[transit_idx]) / 24 * u.day

        return VisibilityWindow(source_name, rise_t, transit_t, set_t, peak_el)

    def is_visible_now(self, coord: SkyCoord, time: Optional[Time] = None) -> bool:
        """Return True if the source is above the horizon limit at the given time."""
        t = time or Time.now()
        return self.site.is_visible(coord, t)

    def next_transit(self, coord: SkyCoord, start_time: Time) -> Optional[Time]:
        """Find the next transit (upper culmination) within 24 hours."""
        hours, elevs = self.elevation_track(coord, start_time, 25.0)
        transit_idx = int(np.argmax(elevs))
        return start_time + float(hours[transit_idx]) / 24 * u.day

    def multi_source_schedule(
        self,
        sources: List,  # List[RadioSource]
        start_time: Time,
        duration_hours: float = 24.0,
    ) -> List[VisibilityWindow]:
        """Compute visibility windows for a list of RadioSource objects."""
        windows = []
        for src in sources:
            w = self.visibility_window(src.coord, src.name,
                                       start_time, duration_hours)
            windows.append(w)
        # Sort: always-up first, then by rise time, never-rises last
        def sort_key(w: VisibilityWindow):
            if w.always_up:
                return (0, -w.peak_elevation_deg)
            if w.never_rises:
                return (2, 0)
            rise_hr = (w.rise_time - start_time).to(u.hour).value if w.rise_time else 999
            return (1, rise_hr)
        return sorted(windows, key=sort_key)


# ---------------------------------------------------------------------------
# Drift-scan beam transit engine
# ---------------------------------------------------------------------------

@dataclass
class BeamTransit:
    """
    Describes a single source crossing through a drift-scan beam.

    The beam is fixed at (beam_az, beam_el).  As Earth rotates the beam
    centre traces a path in RA/Dec.  A source 'transits' when its angular
    separation from that instantaneous beam centre falls below FWHM/2.
    """
    source_name: str
    enter_time:  Optional[Time]   # separation first drops below FWHM/2
    peak_time:   Optional[Time]   # moment of closest approach
    exit_time:   Optional[Time]   # separation rises back above FWHM/2
    peak_separation_deg: float    # angular distance at closest approach
    peak_response: float          # Gaussian beam response 0-1 at peak (1 = beam centre)
    transit_duration_min: float   # time inside the FWHM (minutes)
    source_flux_jy: float = 0.0   # catalog flux for reference

    def is_detected(self, threshold_response: float = 0.5) -> bool:
        """True if peak response exceeds threshold (default: half-power)."""
        return self.peak_response >= threshold_response

    def __str__(self) -> str:
        enter = self.enter_time.iso[:16] if self.enter_time else "---"
        peak  = self.peak_time.iso[:16]  if self.peak_time  else "---"
        det   = "YES" if self.is_detected() else "weak"
        return (
            f"{self.source_name:20s}  "
            f"enter={enter}  peak={peak}  "
            f"dur={self.transit_duration_min:5.1f}min  "
            f"sep={self.peak_separation_deg:5.2f}°  "
            f"resp={self.peak_response:.2f}  "
            f"flux={self.source_flux_jy:.1f}Jy  [{det}]"
        )


class DriftScanPredictor:
    """
    Predicts which catalog sources will pass through a stationary beam
    and computes their crossing times, duration, and beam response.

    Parameters
    ----------
    site        : ObserverSite
    beam_az     : dish azimuth (degrees, 0=N 90=E)
    beam_el     : dish elevation (degrees)
    time_step_min : sampling resolution (default 0.5 min for accuracy)
    """

    def __init__(self, site: ObserverSite, beam_az: float, beam_el: float,
                 time_step_min: float = 0.5):
        self.site          = site
        self.beam_az       = beam_az
        self.beam_el       = beam_el
        self.time_step_min = time_step_min

    def _beam_centre_track(self, start_time: Time,
                           duration_hours: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return arrays of (time_offsets_hours, ra_deg, dec_deg) for the
        beam centre as Earth rotates over duration_hours.
        """
        n      = int(duration_hours * 60 / self.time_step_min)
        offsets = np.linspace(0, duration_hours, n)           # hours
        times   = start_time + offsets * u.hour
        frame   = AltAz(obstime=times, location=self.site.location)
        centre  = SkyCoord(az=self.beam_az * u.deg,
                           alt=self.beam_el * u.deg,
                           frame=frame).icrs
        return offsets, centre.ra.deg, centre.dec.deg

    def _gaussian_response(self, separation_deg: float) -> float:
        """
        Gaussian beam response as a function of angular separation.
        Normalised so response = 1.0 at separation = 0,
        response = 0.5 at separation = FWHM/2.
        """
        fwhm  = self.site.beam_fwhm_deg
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return float(np.exp(-0.5 * (separation_deg / sigma) ** 2))

    def predict(self, sources: list,   # List[RadioSource]
                start_time: Time,
                duration_hours: float = 24.0) -> List[BeamTransit]:
        """
        Find all sources that pass within FWHM/2 of the beam centre
        during the observation window.

        Returns a list of BeamTransit objects sorted by peak_time.
        """
        fwhm   = self.site.beam_fwhm_deg
        radius = fwhm / 2.0

        offsets, bc_ra, bc_dec = self._beam_centre_track(start_time, duration_hours)
        n = len(offsets)

        # Pre-build SkyCoords for beam centres (vectorised separation calc)
        beam_centres = SkyCoord(ra=bc_ra * u.deg, dec=bc_dec * u.deg)

        transits = []
        for src in sources:
            # Angular separation at every time step
            seps = src.coord.separation(beam_centres).deg  # shape (n,)

            inside = seps <= radius
            if not np.any(inside):
                continue

            # Find contiguous windows where the source is inside the beam
            transitions = np.diff(inside.astype(int))
            enter_idxs  = list(np.where(transitions ==  1)[0] + 1)
            exit_idxs   = list(np.where(transitions == -1)[0] + 1)

            # Handle edge cases: already inside at start / still inside at end
            if inside[0]:
                enter_idxs.insert(0, 0)
            if inside[-1]:
                exit_idxs.append(n - 1)

            for enter_i, exit_i in zip(enter_idxs, exit_idxs):
                window_seps = seps[enter_i:exit_i + 1]
                peak_i      = enter_i + int(np.argmin(window_seps))
                peak_sep    = float(seps[peak_i])
                peak_resp   = self._gaussian_response(peak_sep)

                enter_t = start_time + float(offsets[enter_i]) * u.hour
                peak_t  = start_time + float(offsets[peak_i])  * u.hour
                exit_t  = start_time + float(offsets[exit_i])  * u.hour

                dur_min = float((offsets[exit_i] - offsets[enter_i]) * 60)

                transits.append(BeamTransit(
                    source_name          = src.name,
                    enter_time           = enter_t,
                    peak_time            = peak_t,
                    exit_time            = exit_t,
                    peak_separation_deg  = peak_sep,
                    peak_response        = peak_resp,
                    transit_duration_min = dur_min,
                    source_flux_jy       = src.flux_jy,
                ))

        transits.sort(key=lambda t: t.peak_time.unix if t.peak_time else 0)
        return transits

    def print_transits(self, sources: list, start_time: Time,
                       duration_hours: float = 24.0,
                       min_response: float = 0.0) -> List[BeamTransit]:
        """Predict and pretty-print the transit table. Returns the list."""
        transits = self.predict(sources, start_time, duration_hours)
        if min_response > 0:
            transits = [t for t in transits if t.peak_response >= min_response]

        fwhm = self.site.beam_fwhm_deg
        print(f"\n{'='*90}")
        print(f"  Drift-Scan Beam Transit Predictions")
        print(f"  Site: {self.site.name}  |  Az={self.beam_az:.1f}°  El={self.beam_el:.1f}°  "
              f"|  FWHM={fwhm:.2f}°")
        print(f"  Window: {start_time.iso[:16]} UTC  +{duration_hours:.0f}h")
        print(f"{'='*90}")
        hdr = (f"  {'Source':<20}  {'Enter (UTC)':<16}  {'Peak (UTC)':<16}  "
               f"{'Exit (UTC)':<16}  {'Dur':>6}  {'Sep':>6}  {'Resp':>5}  "
               f"{'Flux':>7}  Det?")
        print(hdr)
        print(f"  {'-'*86}")

        if not transits:
            print("  No sources transit the beam in this window.")
        for t in transits:
            enter = t.enter_time.iso[11:16] if t.enter_time else "---  "
            enter_d = t.enter_time.iso[:10] if t.enter_time else ""
            peak  = t.peak_time.iso[:16]  if t.peak_time  else "---"
            exit_ = t.exit_time.iso[11:16]  if t.exit_time  else "---  "
            det   = "  ✓" if t.is_detected() else "  ~"
            print(f"  {t.source_name:<20}  "
                  f"{enter_d} {enter}  "
                  f"{peak}  "
                  f"{enter_d} {exit_}  "
                  f"{t.transit_duration_min:>5.1f}m  "
                  f"{t.peak_separation_deg:>5.2f}°  "
                  f"{t.peak_response:>5.2f}  "
                  f"{t.source_flux_jy:>6.1f}Jy"
                  f"{det}")

        print(f"  {'-'*86}")
        print(f"  {len(transits)} transit(s) found.  "
              f"✓ = peak response ≥ 0.5 (within half-power beam width)")
        print(f"{'='*90}\n")
        return transits