"""
skymap.py
Interactive sky map using matplotlib.
Rectangular RA/Dec projection with straight grid lines, source overlays,
galactic plane, visibility shading, and drift-scan antenna beam footprint.

For a stationary (drift-scan) dish the beam is fixed in Az/El.  On the
RA/Dec map this appears as a curved trail — the locus of RA/Dec
coordinates that pass through the beam over 24 hours as Earth rotates.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from astropy.coordinates import SkyCoord, AltAz, Galactic
from astropy.time import Time
import astropy.units as u

from core.observer import ObserverSite
from core.catalog import RadioCatalog


# ---------------------------------------------------------------------------
# Source style lookup
# ---------------------------------------------------------------------------

SOURCE_STYLES = {
    "supernova remnant": dict(color="#FF6B6B", marker="*",  size=14),
    "radio galaxy":      dict(color="#4ECDC4", marker="D",  size=10),
    "galactic center":   dict(color="#FFE66D", marker="X",  size=14),
    "hii region":        dict(color="#A8E6CF", marker="^",  size=10),
    "pulsar":            dict(color="#FF8B94", marker="p",  size=12),
    "galaxy":            dict(color="#C3A6FF", marker="o",  size=9),
    "solar":             dict(color="#FFD700", marker="o",  size=18),
    "continuum":         dict(color="#FFFFFF", marker="o",  size=8),
    "galactic":          dict(color="#98D8C8", marker="s",  size=8),
    "default":           dict(color="#AAAAAA", marker="o",  size=7),
}

def _source_style(source_type: str) -> dict:
    for key, style in SOURCE_STYLES.items():
        if key in source_type.lower():
            return style
    return SOURCE_STYLES["default"]


# ---------------------------------------------------------------------------
# SkyMap
# ---------------------------------------------------------------------------

class SkyMap:
    """
    Rectangular RA/Dec sky map with drift-scan beam overlay.

    Parameters
    ----------
    site        : ObserverSite
    catalog     : RadioCatalog, optional
    mode        : 'altaz' shades sources by current visibility; 'radec' shows all
    beam_az     : fixed azimuth of the dish (degrees, 0=N 90=E)
    beam_el     : fixed elevation of the dish (degrees above horizon)
    drift_hours : how many hours of drift trail to draw (default 24)
    figsize     : figure size in inches
    """

    DARK_BG    = "#0D1117"
    GRID_COLOR = "#2D3748"
    TEXT_COLOR = "#E2E8F0"
    BEAM_COLOR = "#00BFFF"

    def __init__(
        self,
        site: ObserverSite,
        catalog: Optional[RadioCatalog] = None,
        mode: str = "altaz",
        beam_az: Optional[float] = None,
        beam_el: Optional[float] = None,
        drift_hours: float = 24.0,
        figsize: Tuple[float, float] = (14, 7),
    ):
        self.site        = site
        self.catalog     = catalog
        self.mode        = mode
        self.beam_az     = beam_az
        self.beam_el     = beam_el
        self.drift_hours = drift_hours
        self.figsize     = figsize
        self.fig: Optional[Figure] = None
        self.ax:  Optional[Axes]   = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, time: Optional[Time] = None) -> Figure:
        t = time or Time.now()
        self.fig = plt.figure(figsize=self.figsize, facecolor=self.DARK_BG)
        self._render_radec(t, show_visibility=(self.mode == "altaz"))
        self._add_title(t)
        self._add_legend()
        plt.tight_layout()
        return self.fig

    def show(self, time: Optional[Time] = None) -> None:
        self.render(time)
        plt.show()

    def save(self, path: str, time: Optional[Time] = None, dpi: int = 150) -> None:
        self.render(time)
        self.fig.savefig(path, dpi=dpi, facecolor=self.DARK_BG, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Helpers: Az/El  <->  RA/Dec
    # ------------------------------------------------------------------

    def _altaz_to_radec(self, az_deg: float, el_deg: float, time: Time) -> SkyCoord:
        """Convert a fixed Az/El to RA/Dec at a given time."""
        altaz_frame = AltAz(obstime=time, location=self.site.location)
        altaz_coord = SkyCoord(az=az_deg * u.deg, alt=el_deg * u.deg,
                               frame=altaz_frame)
        return altaz_coord.icrs

    def _drift_trail(self, az_deg: float, el_deg: float,
                     start_time: Time, hours: float = 24.0,
                     n: int = 720) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trace the RA/Dec locus of a fixed Az/El pointing over `hours` hours.
        Returns (ra_deg array, dec_deg array).
        """
        offsets = np.linspace(0, hours, n) * u.hour
        times   = start_time + offsets
        altaz_frame = AltAz(obstime=times, location=self.site.location)
        coords  = SkyCoord(az=az_deg * u.deg, alt=el_deg * u.deg,
                           frame=altaz_frame).icrs
        return coords.ra.deg, coords.dec.deg

    def _beam_boundary_radec(self, az_deg: float, el_deg: float,
                             time: Time, n: int = 360,
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample the FWHM beam circle (in Az/El) and convert each point
        to RA/Dec at the given instant, giving the true beam footprint
        on the sky.
        """
        fwhm  = self.site.beam_fwhm_deg
        r     = fwhm / 2.0
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

        # Offset azimuths and elevations around the beam centre.
        # Az offset is stretched by 1/cos(el) so the circle is correct on sky.
        cos_el = np.cos(np.radians(el_deg))
        if abs(cos_el) < 0.01:
            cos_el = 0.01
        az_offsets  = az_deg  + r / cos_el * np.cos(theta)
        el_offsets  = el_deg  + r           * np.sin(theta)

        # Clip elevations to physical range
        el_offsets = np.clip(el_offsets, -89.9, 89.9)

        altaz_frame = AltAz(obstime=time, location=self.site.location)
        coords = SkyCoord(az=az_offsets * u.deg, alt=el_offsets * u.deg,
                          frame=altaz_frame).icrs
        return coords.ra.deg, coords.dec.deg

    # ------------------------------------------------------------------
    # Core rectangular RA/Dec renderer
    # ------------------------------------------------------------------

    def _render_radec(self, time: Time, show_visibility: bool = True) -> None:
        self.ax = self.fig.add_subplot(111, facecolor=self.DARK_BG)
        ax = self.ax

        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Right Ascension (°)", color=self.TEXT_COLOR, fontsize=10)
        ax.set_ylabel("Declination (°)",     color=self.TEXT_COLOR, fontsize=10)
        ax.tick_params(colors=self.TEXT_COLOR, which="both")
        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

        # Straight RA grid lines every 30°
        for ra in range(0, 361, 30):
            ax.axvline(x=ra, color=self.GRID_COLOR, linewidth=0.5,
                       linestyle="--", alpha=0.7)
            if 0 < ra < 360:
                ax.text(ra + 1, -88, f"{ra}°",
                        color=self.TEXT_COLOR, fontsize=7, alpha=0.5, va="bottom")

        # Straight Dec grid lines every 30°
        for dec in range(-90, 91, 30):
            ax.axhline(y=dec, color=self.GRID_COLOR, linewidth=0.5,
                       linestyle="--", alpha=0.7)

        # Celestial equator
        ax.axhline(y=0, color="#4A6A8A", linewidth=0.9, linestyle="-", alpha=0.6)

        # Observable dec band
        if show_visibility:
            min_dec = self.site.latitude + self.site.min_elevation - 90
            max_dec = self.site.latitude - self.site.min_elevation + 90
            ax.axhspan(min_dec, max_dec, color="#1A2A1A", alpha=0.25, zorder=0)
            ax.axhline(y=min_dec, color="#4A7A4A", linewidth=1,
                       linestyle="--", alpha=0.85)
            ax.axhline(y=max_dec, color="#4A7A4A", linewidth=1,
                       linestyle="--", alpha=0.85)
            ax.text(2, min_dec + 1, f"Min dec {min_dec:.0f}°",
                    color="#4A7A4A", fontsize=7, alpha=0.85)
            ax.text(2, max_dec + 1, f"Max dec {max_dec:.0f}°",
                    color="#4A7A4A", fontsize=7, alpha=0.85)

        self._draw_galactic_plane(ax)

        if self.catalog:
            self._plot_sources(ax, time, show_visibility)

        self._draw_drift_beam(ax, time)

    # ------------------------------------------------------------------
    # Source plotting
    # ------------------------------------------------------------------

    def _plot_sources(self, ax: Axes, time: Time, show_visibility: bool) -> None:
        frame = self.site.altaz_frame(time)
        for src in self.catalog:
            altaz = src.coord.transform_to(frame)
            el    = float(altaz.alt.deg)
            above = el >= self.site.min_elevation
            style = _source_style(src.source_type)

            alpha = (1.0  if above else 0.20) if show_visibility else 0.9
            fc    = (style["color"] if above else "#3A3A4A") if show_visibility else style["color"]

            ax.scatter(src.ra_deg, src.dec_deg,
                       c=fc, marker=style["marker"],
                       s=style["size"] ** 2, alpha=alpha,
                       zorder=3, edgecolors="none")

            if not show_visibility or above:
                ax.annotate(src.name, (src.ra_deg, src.dec_deg),
                            fontsize=7, color=style["color"],
                            xytext=(5, 3), textcoords="offset points",
                            alpha=0.9)

    # ------------------------------------------------------------------
    # Galactic plane
    # ------------------------------------------------------------------

    def _draw_galactic_plane(self, ax: Axes) -> None:
        gal_l  = np.linspace(0, 360, 1440) * u.deg
        gal_b  = np.zeros(1440) * u.deg
        coords = SkyCoord(l=gal_l, b=gal_b, frame=Galactic).icrs
        ra     = coords.ra.deg
        dec    = coords.dec.deg

        order    = np.argsort(ra)
        ra_s     = ra[order];  dec_s = dec[order]
        gaps     = np.where(np.diff(ra_s) > 10)[0] + 1
        segments = np.split(np.column_stack([ra_s, dec_s]), gaps)

        first = True
        for seg in segments:
            if len(seg) < 2:
                continue
            ax.plot(seg[:, 0], seg[:, 1],
                    color="#7B68EE", linewidth=1.2, alpha=0.6,
                    linestyle=":", zorder=2,
                    label="Galactic plane" if first else "_nolegend_")
            first = False

    # ------------------------------------------------------------------
    # Drift-scan beam overlay
    # ------------------------------------------------------------------

    def _draw_drift_beam(self, ax: Axes, time: Time) -> None:
        """
        For a stationary dish fixed at (beam_az, beam_el):

        1. Draw the 24-hour drift trail — the curved line tracing which
           RA/Dec coordinates pass through the beam pointing over a full day.

        2. Draw the instantaneous beam footprint (FWHM circle in Az/El
           space, projected into RA/Dec) at the given time.

        3. Mark the current pointing centre with a crosshair.
        """
        if self.beam_az is None or self.beam_el is None:
            return

        az  = self.beam_az
        el  = self.beam_el
        fwhm = self.site.beam_fwhm_deg

        # ------------------------------------------------------------------
        # 1. Drift trail over self.drift_hours
        # ------------------------------------------------------------------
        ra_trail, dec_trail = self._drift_trail(az, el, time, self.drift_hours)

        # Split trail at RA wraparound (>300° jump)
        gaps     = np.where(np.abs(np.diff(ra_trail)) > 300)[0] + 1
        ra_segs  = np.split(ra_trail,  gaps)
        dec_segs = np.split(dec_trail, gaps)

        first = True
        for ra_s, dec_s in zip(ra_segs, dec_segs):
            if len(ra_s) < 2:
                continue
            ax.plot(ra_s, dec_s,
                    color=self.BEAM_COLOR, linewidth=1.5,
                    linestyle="--", alpha=0.55, zorder=4,
                    label=f"Drift trail (Az={az:.1f}° El={el:.1f}°, {self.drift_hours:.0f}h)"
                          if first else "_nolegend_")
            first = False

        # ------------------------------------------------------------------
        # 2. Instantaneous beam footprint at `time`
        # ------------------------------------------------------------------
        ra_bnd, dec_bnd = self._beam_boundary_radec(az, el, time)

        # Handle wraparound in the boundary polygon by splitting and filling
        # each continuous segment separately
        jumps = np.where(np.abs(np.diff(ra_bnd)) > 300)[0] + 1
        if len(jumps) == 0:
            # Simple case — no wraparound, fill as polygon
            ax.fill(ra_bnd, dec_bnd,
                    color=self.BEAM_COLOR, alpha=0.12, zorder=5)
            ax.plot(np.append(ra_bnd, ra_bnd[0]),
                    np.append(dec_bnd, dec_bnd[0]),
                    color=self.BEAM_COLOR, linewidth=1.5,
                    linestyle="-", alpha=0.85, zorder=6,
                    label=f"Beam FWHM {fwhm:.2f}°")
        else:
            # Wraparound: just draw the outline without fill to avoid artefacts
            ax.plot(np.append(ra_bnd, ra_bnd[0]),
                    np.append(dec_bnd, dec_bnd[0]),
                    color=self.BEAM_COLOR, linewidth=1.5,
                    linestyle="-", alpha=0.85, zorder=6,
                    label=f"Beam FWHM {fwhm:.2f}°")

        # ------------------------------------------------------------------
        # 3. Current pointing centre + crosshair
        # ------------------------------------------------------------------
        centre = self._altaz_to_radec(az, el, time)
        ra_c   = float(centre.ra.deg)
        dec_c  = float(centre.dec.deg)

        cos_dec = max(abs(np.cos(np.radians(dec_c))), 0.01)
        ch = fwhm * 0.2
        ax.plot([ra_c - ch / cos_dec, ra_c + ch / cos_dec],
                [dec_c, dec_c],
                color=self.BEAM_COLOR, linewidth=1.2, alpha=0.95, zorder=7)
        ax.plot([ra_c, ra_c],
                [dec_c - ch, dec_c + ch],
                color=self.BEAM_COLOR, linewidth=1.2, alpha=0.95, zorder=7)
        ax.scatter([ra_c], [dec_c],
                   c=self.BEAM_COLOR, s=18, zorder=8, edgecolors="none")

        # Annotation
        ax.text(ra_c + fwhm / cos_dec * 0.55, dec_c,
                f" Az={az:.1f}°  El={el:.1f}°\n"
                f" RA={ra_c:.1f}°  Dec={dec_c:+.1f}°\n"
                f" FWHM={fwhm:.2f}°",
                color=self.BEAM_COLOR, fontsize=7,
                va="center", alpha=0.9, zorder=7)

    # ------------------------------------------------------------------
    # Decorations
    # ------------------------------------------------------------------

    def _add_title(self, time: Time) -> None:
        beam_str = ""
        if self.beam_az is not None and self.beam_el is not None:
            beam_str = (f"  |  Dish: Az={self.beam_az:.1f}°  El={self.beam_el:.1f}°"
                        f"  FWHM={self.site.beam_fwhm_deg:.2f}°  (drift scan)")
        title = (
            f"{self.site.name}  |  {time.iso[:16]} UTC  |  "
            f"Lat={self.site.latitude:+.2f}°  Lon={self.site.longitude:+.2f}°  |  "
            f"{self.site.frequency_mhz:.1f} MHz  |  Dish={self.site.dish_diameter:.1f} m"
            f"{beam_str}"
        )
        self.fig.suptitle(title, color=self.TEXT_COLOR, fontsize=8,
                          y=0.99, fontfamily="monospace")

    def _add_legend(self) -> None:
        handles = []
        seen    = set()
        if self.catalog:
            for src in self.catalog:
                t = src.source_type
                if t not in seen:
                    style = _source_style(t)
                    handles.append(mpatches.Patch(color=style["color"], label=t.title()))
                    seen.add(t)

        handles.append(mpatches.Patch(color="#7B68EE", label="Galactic Plane", alpha=0.6))

        if self.beam_az is not None and self.beam_el is not None:
            fwhm = self.site.beam_fwhm_deg
            handles.append(mpatches.Patch(
                color=self.BEAM_COLOR, alpha=0.7,
                label=f"Beam FWHM {fwhm:.2f}° (Az={self.beam_az:.1f}° El={self.beam_el:.1f}°)"
            ))
            handles.append(mpatches.Patch(
                color=self.BEAM_COLOR, alpha=0.4,
                label=f"Drift trail ({self.drift_hours:.0f}h)"
            ))

        if handles:
            self.ax.legend(
                handles=handles, loc="lower right", fontsize=7,
                framealpha=0.3, labelcolor=self.TEXT_COLOR,
                facecolor=self.DARK_BG, edgecolor=self.GRID_COLOR,
            )