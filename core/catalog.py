"""
catalog.py
Radio source catalog management. Supports built-in bright sources,
CSV catalogs, and live queries via astroquery.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RadioSource:
    """A single radio source entry."""
    name: str
    ra_deg: float          # J2000 right ascension, degrees
    dec_deg: float         # J2000 declination, degrees
    flux_jy: float = 0.0   # flux density at reference frequency (Jy)
    ref_freq_mhz: float = 1400.0
    spectral_index: float = -0.7  # S ∝ ν^α
    source_type: str = "continuum"  # continuum | spectral | pulsar | galaxy
    notes: str = ""

    @property
    def coord(self) -> SkyCoord:
        return SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg,
                        frame="icrs")

    def flux_at(self, freq_mhz: float) -> float:
        """Extrapolate flux to a given frequency using a power-law spectrum."""
        if self.flux_jy <= 0:
            return 0.0
        return float(self.flux_jy * (freq_mhz / self.ref_freq_mhz) ** self.spectral_index)

    def __str__(self) -> str:
        return (f"{self.name:20s}  RA={self.ra_deg:8.3f}°  "
                f"Dec={self.dec_deg:+8.3f}°  S={self.flux_jy:.1f} Jy")


# ---------------------------------------------------------------------------
# Built-in bright radio source catalog (subset of the 3C / bright sources)
# ---------------------------------------------------------------------------

BRIGHT_SOURCES: List[RadioSource] = [
    RadioSource("3C 461 (Cas A)",     350.8667,  58.8117, 2720.0, 1000, -0.77, "supernova remnant"),
    RadioSource("3C 405 (Cyg A)",     299.8672,  40.7339, 1590.0, 1000, -1.00, "radio galaxy"),
    RadioSource("3C 144 (Tau A)",      83.8221,  22.0144,  955.0, 1000, -0.30, "supernova remnant"),
    RadioSource("3C 274 (Vir A/M87)", 187.7059,  12.3911,  861.0, 1000, -0.86, "radio galaxy"),
    RadioSource("Sgr A*",             266.4168, -29.0078,  370.0,  10e3, -0.60, "galactic center"),
    RadioSource("3C 123",              69.2683,  29.6706,   48.0, 1000, -0.85, "radio galaxy"),
    RadioSource("3C 218 (Hydra A)",   139.5237, -12.0956,   40.0, 1000, -0.91, "radio galaxy"),
    RadioSource("Sun",                  0.0,      0.0,      1e6,   1e3,  2.0,  "solar"),  # coords set at runtime via get_sun()
    RadioSource("Orion A",             83.8221,  -5.3911,   50.0, 1000, -0.1,  "hii region"),
    RadioSource("W3",                  35.5833,  61.9,      30.0, 1000, -0.5,  "hii region"),
    RadioSource("Puppis A",           125.7143, -42.9983,   60.0, 1000, -0.5,  "supernova remnant"),
]


# ---------------------------------------------------------------------------
# Catalog class
# ---------------------------------------------------------------------------

class RadioCatalog:
    """Container for a list of RadioSource objects with filtering helpers."""

    def __init__(self, name: str = "Custom", sources: Optional[List[RadioSource]] = None):
        self.name = name
        self.sources: List[RadioSource] = sources or []

    def __len__(self) -> int:
        return len(self.sources)

    def __iter__(self):
        return iter(self.sources)

    def add(self, source: RadioSource) -> None:
        self.sources.append(source)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def by_type(self, source_type: str) -> "RadioCatalog":
        filtered = [s for s in self.sources
                    if source_type.lower() in s.source_type.lower()]
        return RadioCatalog(f"{self.name}/{source_type}", filtered)

    def by_flux_min(self, min_jy: float, freq_mhz: float = 1400.0) -> "RadioCatalog":
        filtered = [s for s in self.sources if s.flux_at(freq_mhz) >= min_jy]
        return RadioCatalog(f"{self.name}/flux>{min_jy}Jy", filtered)

    def by_sky_region(self, center: SkyCoord, radius_deg: float) -> "RadioCatalog":
        coords = SkyCoord(
            ra=[s.ra_deg for s in self.sources] * u.deg,
            dec=[s.dec_deg for s in self.sources] * u.deg,
        )
        seps = center.separation(coords).deg
        filtered = [s for s, sep in zip(self.sources, seps) if sep <= radius_deg]
        return RadioCatalog(f"{self.name}/region", filtered)

    def search(self, query: str) -> List[RadioSource]:
        q = query.lower()
        return [s for s in self.sources if q in s.name.lower()]

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["name", "ra_deg", "dec_deg", "flux_jy",
                         "ref_freq_mhz", "spectral_index", "source_type", "notes"])
        for s in self.sources:
            writer.writerow([s.name, s.ra_deg, s.dec_deg, s.flux_jy,
                             s.ref_freq_mhz, s.spectral_index,
                             s.source_type, s.notes])
        return buf.getvalue()

    @classmethod
    def from_csv(cls, path: str | Path, name: str = "") -> "RadioCatalog":
        sources = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sources.append(RadioSource(
                    name=row["name"],
                    ra_deg=float(row["ra_deg"]),
                    dec_deg=float(row["dec_deg"]),
                    flux_jy=float(row.get("flux_jy", 0)),
                    ref_freq_mhz=float(row.get("ref_freq_mhz", 1400)),
                    spectral_index=float(row.get("spectral_index", -0.7)),
                    source_type=row.get("source_type", "continuum"),
                    notes=row.get("notes", ""),
                ))
        return cls(name or Path(path).stem, sources)

    def save_csv(self, path: str | Path) -> None:
        Path(path).write_text(self.to_csv())


# ---------------------------------------------------------------------------
# Live query via astroquery (optional — requires internet)
# ---------------------------------------------------------------------------

def query_nvss(ra_deg: float, dec_deg: float, radius_deg: float = 1.0,
               min_flux_mjy: float = 100.0) -> RadioCatalog:
    """
    Query the NRAO VLA Sky Survey (NVSS) catalog via astroquery.
    Returns a RadioCatalog with matched sources.
    Requires: astroquery, internet connection.
    """
    try:
        from astroquery.vizier import Vizier
        center = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        v = Vizier(columns=["NVSS", "RAJ2000", "DEJ2000", "S1.4"],
                   column_filters={"S1.4": f">{min_flux_mjy}"},
                   row_limit=500)
        result = v.query_region(center, radius=radius_deg * u.deg,
                                catalog="VIII/65/nvss")
        if not result:
            return RadioCatalog("NVSS (empty)")
        tbl = result[0]
        sources = []
        for row in tbl:
            sources.append(RadioSource(
                name=str(row["NVSS"]),
                ra_deg=float(row["RAJ2000"]),
                dec_deg=float(row["DEJ2000"]),
                flux_jy=float(row["S1.4"]) / 1000.0,  # mJy -> Jy
                ref_freq_mhz=1400.0,
                source_type="continuum",
            ))
        return RadioCatalog("NVSS", sources)
    except Exception as exc:
        print(f"[catalog] NVSS query failed: {exc}")
        return RadioCatalog("NVSS (failed)")


# ---------------------------------------------------------------------------
# Default catalog singleton
# ---------------------------------------------------------------------------

def default_catalog(time=None) -> RadioCatalog:
    """
    Return the built-in bright source catalog.
    If time is provided (astropy Time), the Sun's position is updated
    to its actual RA/Dec at that time using astropy.coordinates.get_sun().
    """
    from astropy.coordinates import get_sun
    from astropy.time import Time as ATime
    import astropy.units as u

    sources = list(BRIGHT_SOURCES)

    t = time if time is not None else ATime.now()
    sun_coord = get_sun(t)
    for i, src in enumerate(sources):
        if src.name == "Sun":
            sources[i] = RadioSource(
                name="Sun",
                ra_deg=float(sun_coord.ra.deg),
                dec_deg=float(sun_coord.dec.deg),
                flux_jy=src.flux_jy,
                ref_freq_mhz=src.ref_freq_mhz,
                spectral_index=src.spectral_index,
                source_type=src.source_type,
                notes="Position computed for observation time",
            )
            break

    return RadioCatalog("Bright Sources", sources)