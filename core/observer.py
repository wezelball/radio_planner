"""
observer.py
Manages the observer's site location, current time, and coordinate transforms.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from astropy.coordinates import EarthLocation, AltAz, ICRS, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u


@dataclass
class ObserverSite:
    """Represents a radio telescope / observation site."""
    name: str
    latitude: float          # degrees, +N
    longitude: float         # degrees, +E
    elevation: float = 0.0   # meters above sea level
    min_elevation: float = 10.0  # degrees — antenna horizon limit

    # Antenna parameters
    dish_diameter: float = 3.0    # meters
    frequency_mhz: float = 1420.0 # MHz (default: HI line)
    system_temp_k: float = 100.0  # system noise temperature (K)

    _location: Optional[EarthLocation] = field(default=None, init=False, repr=False)

    @property
    def location(self) -> EarthLocation:
        if self._location is None:
            self._location = EarthLocation(
                lat=self.latitude * u.deg,
                lon=self.longitude * u.deg,
                height=self.elevation * u.m,
            )
        return self._location

    def altaz_frame(self, time: Time) -> AltAz:
        """Return an AltAz frame for this site at the given time."""
        return AltAz(obstime=time, location=self.location)

    def to_altaz(self, coord: SkyCoord, time: Time) -> SkyCoord:
        """Convert an ICRS SkyCoord to AltAz at this site and time."""
        return coord.transform_to(self.altaz_frame(time))

    def is_visible(self, coord: SkyCoord, time: Time) -> bool:
        """Return True if the source is above the antenna horizon limit."""
        altaz = self.to_altaz(coord, time)
        return float(altaz.alt.deg) >= self.min_elevation

    @property
    def beam_fwhm_deg(self) -> float:
        """Approximate half-power beam width (HPBW) in degrees."""
        wavelength_m = 3e8 / (self.frequency_mhz * 1e6)
        return float(np.degrees(1.22 * wavelength_m / self.dish_diameter))

    @property
    def sefd_jy(self) -> float:
        """System Equivalent Flux Density in Jansky."""
        wavelength_m = 3e8 / (self.frequency_mhz * 1e6)
        area_m2 = np.pi * (self.dish_diameter / 2) ** 2
        aperture_eff = 0.6
        aeff = aperture_eff * area_m2
        k_b = 1.38e-23
        return float(2 * k_b * self.system_temp_k / aeff * 1e26)  # Jy

    def sensitivity_mjy(self, bandwidth_mhz: float, integration_s: float,
                        n_pol: int = 2) -> float:
        """Minimum detectable flux density (mJy, 1-sigma)."""
        delta_nu = bandwidth_mhz * 1e6
        rms = self.sefd_jy / np.sqrt(n_pol * delta_nu * integration_s)
        return float(rms * 1e3)  # mJy


# ---------------------------------------------------------------------------
# Convenience factory for common sites
# ---------------------------------------------------------------------------

KNOWN_SITES = {
    "green_bank": ObserverSite("Green Bank", 38.4331, -79.8397, 880),
    "vla": ObserverSite("VLA", 34.0784, -107.6184, 2124),
    "parkes": ObserverSite("Parkes", -32.9983, 148.2635, 415),
    "arecibo": ObserverSite("Arecibo", 18.3464, -66.7528, 497),
    "effelsberg": ObserverSite("Effelsberg", 50.5247, 6.8828, 369),
    "twomice": ObserverSite(name="TwoMice", latitude=37.7906, longitude=-77.9242, elevation=116, min_elevation=30.0, dish_diameter=0.7, system_temp_k=150.0),
    "custom": ObserverSite("Custom Site", 0.0, 0.0, 0),
}


def get_site(name: str) -> ObserverSite:
    key = name.lower().replace(" ", "_")
    if key not in KNOWN_SITES:
        raise ValueError(f"Unknown site '{name}'. Known: {list(KNOWN_SITES)}")
    return KNOWN_SITES[key]


def now_utc() -> Time:
    """Current UTC time as an astropy Time object."""
    return Time(datetime.now(timezone.utc))
