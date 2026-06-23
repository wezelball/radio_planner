"""
ephemeris.py
Rise, set, transit calculations and visibility window generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
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
        t = time or Time.now()
        return self.site.is_visible(coord, t)

    def next_transit(self, coord: SkyCoord, start_time: Time) -> Optional[Time]:
        """Find the next transit (upper culmination) within 24 hours."""
        hours, elevs = self.elevation_track(coord, start_time, 25.0)
        transit_idx = int(np.argmax(elevs))
        return start_time + float(hours[transit_idx]) / 24 * u.day

    def multi_source_schedule(
        self,
        sources: list,  # List[RadioSource]
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
