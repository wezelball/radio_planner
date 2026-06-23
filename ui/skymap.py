"""
skymap.py
Interactive sky map using matplotlib.
Rectangular RA/Dec projection with straight grid lines, source overlays,
galactic plane, visibility shading, and antenna beam footprint.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from astropy.coordinates import SkyCoord, AltAz, Galactic
from astropy.time import Time
import astropy.units as u

from core.observer import ObserverSite
from core.catalog import RadioSource, RadioCatalog


# ---------------------------------------------------------------------------
# Color / marker scheme for source types
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
    Rectangular RA/Dec sky map.

    Parameters
    ----------
    site         : ObserverSite
    catalog      : RadioCatalog, optional
    mode         : 'altaz' (shade sources by current visibility) or 'radec'
    beam_target  : SkyCoord, optional — centre of the antenna beam overlay
    figsize      : figure size in inches
    """

    DARK_BG     = "#0D1117"
    GRID_COLOR  = "#2D3748"
    TEXT_COLOR  = "#E2E8F0"

    def __init__(
        self,
        site: ObserverSite,
        catalog: Optional[RadioCatalog] = None,
        mode: str = "altaz",
        beam_target: Optional[SkyCoord] = None,
        figsize: Tuple[float, float] = (14, 7),
    ):
        self.site        = site
        self.catalog     = catalog
        self.mode        = mode
        self.beam_target = beam_target   # None → no beam drawn
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
        self.fig.savefig(path, dpi=dpi, facecolor=self.DARK_BG,
                         bbox_inches="tight")

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

        # --- Straight RA grid lines every 30° ---
        for ra in range(0, 361, 30):
            ax.axvline(x=ra, color=self.GRID_COLOR, linewidth=0.5,
                       linestyle="--", alpha=0.7)
            if 0 < ra < 360:
                ax.text(ra + 1, -88, f"{ra}°",
                        color=self.TEXT_COLOR, fontsize=7, alpha=0.5, va="bottom")

        # --- Straight Dec grid lines every 30° ---
        for dec in range(-90, 91, 30):
            ax.axhline(y=dec, color=self.GRID_COLOR, linewidth=0.5,
                       linestyle="--", alpha=0.7)

        # Celestial equator
        ax.axhline(y=0, color="#4A6A8A", linewidth=0.9,
                   linestyle="-", alpha=0.6)

        # --- Observable dec band for this site ---
        if show_visibility:
            min_dec = self.site.latitude + self.site.min_elevation - 90
            max_dec = self.site.latitude - self.site.min_elevation + 90
            ax.axhspan(min_dec, max_dec,
                       color="#1A2A1A", alpha=0.25, zorder=0)
            ax.axhline(y=min_dec, color="#4A7A4A", linewidth=1,
                       linestyle="--", alpha=0.85)
            ax.axhline(y=max_dec, color="#4A7A4A", linewidth=1,
                       linestyle="--", alpha=0.85)
            ax.text(2, min_dec + 1, f"Min dec {min_dec:.0f}°",
                    color="#4A7A4A", fontsize=7, alpha=0.85)
            ax.text(2, max_dec + 1, f"Max dec {max_dec:.0f}°",
                    color="#4A7A4A", fontsize=7, alpha=0.85)

        # --- Galactic plane ---
        self._draw_galactic_plane(ax)

        # --- Sources ---
        if self.catalog:
            self._plot_sources(ax, time, show_visibility)

        # --- Antenna beam ---
        self._draw_beam(ax)

    # ------------------------------------------------------------------
    # Source plotting
    # ------------------------------------------------------------------

    def _plot_sources(self, ax: Axes, time: Time,
                      show_visibility: bool) -> None:
        frame = self.site.altaz_frame(time)
        for src in self.catalog:
            altaz  = src.coord.transform_to(frame)
            el     = float(altaz.alt.deg)
            above  = el >= self.site.min_elevation
            style  = _source_style(src.source_type)

            if show_visibility:
                alpha = 1.0  if above else 0.20
                fc    = style["color"] if above else "#3A3A4A"
            else:
                alpha = 0.9
                fc    = style["color"]

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

        order  = np.argsort(ra)
        ra_s   = ra[order]
        dec_s  = dec[order]

        # Split at RA wraparound gaps > 10°
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
    # Antenna beam footprint
    # ------------------------------------------------------------------

    def _draw_beam(self, ax: Axes) -> None:
        """
        Draw the antenna FWHM beam as an ellipse on the RA/Dec plot.

        In a rectangular (plate carrée) projection the Dec axis is linear
        in degrees, but the RA axis is compressed by cos(Dec) — a circle
        on the sky appears as an ellipse whose RA width is
            fwhm / cos(dec_centre).
        We draw:
          • a filled ellipse at ~5 % opacity (beam footprint)
          • a solid ellipse border (FWHM contour)
          • a small cross-hair at the pointing centre
          • a text label showing the FWHM
        """
        if self.beam_target is None:
            return

        fwhm_deg = self.site.beam_fwhm_deg
        ra_c     = float(self.beam_target.ra.deg)
        dec_c    = float(self.beam_target.dec.deg)

        # RA width is stretched by 1/cos(dec) in plate-carrée coordinates
        cos_dec  = np.cos(np.radians(dec_c))
        # Guard against poles
        if abs(cos_dec) < 0.01:
            cos_dec = 0.01
        ra_width  = fwhm_deg / cos_dec   # degrees along RA axis
        dec_height = fwhm_deg            # degrees along Dec axis

        # Filled beam area
        beam_fill = Ellipse(
            xy=(ra_c, dec_c),
            width=ra_width,
            height=dec_height,
            facecolor="#00BFFF",
            edgecolor="none",
            alpha=0.10,
            zorder=4,
        )
        ax.add_patch(beam_fill)

        # FWHM contour
        beam_edge = Ellipse(
            xy=(ra_c, dec_c),
            width=ra_width,
            height=dec_height,
            facecolor="none",
            edgecolor="#00BFFF",
            linewidth=1.5,
            linestyle="-",
            alpha=0.85,
            zorder=5,
            label=f"Beam FWHM {fwhm_deg:.2f}°",
        )
        ax.add_patch(beam_edge)

        # Cross-hair at pointing centre
        ch_size = fwhm_deg * 0.18
        ax.plot([ra_c - ch_size / cos_dec, ra_c + ch_size / cos_dec],
                [dec_c, dec_c],
                color="#00BFFF", linewidth=1.0, alpha=0.9, zorder=6)
        ax.plot([ra_c, ra_c],
                [dec_c - ch_size, dec_c + ch_size],
                color="#00BFFF", linewidth=1.0, alpha=0.9, zorder=6)

        # Label
        ax.text(ra_c + ra_width / 2 + 1, dec_c,
                f" FWHM {fwhm_deg:.2f}°\n RA={ra_c:.1f}°  Dec={dec_c:+.1f}°",
                color="#00BFFF", fontsize=7, va="center", alpha=0.9, zorder=6)

    # ------------------------------------------------------------------
    # Decorations
    # ------------------------------------------------------------------

    def _add_title(self, time: Time) -> None:
        beam_str = ""
        if self.beam_target is not None:
            beam_str = (f"  |  Beam FWHM={self.site.beam_fwhm_deg:.2f}°"
                        f"  pointing RA={self.beam_target.ra.deg:.1f}°"
                        f"  Dec={self.beam_target.dec.deg:+.1f}°")
        title = (
            f"{self.site.name}  |  {time.iso[:16]} UTC  |  "
            f"Lat={self.site.latitude:+.2f}°  Lon={self.site.longitude:+.2f}°  |  "
            f"{self.site.frequency_mhz:.1f} MHz  |  "
            f"Dish={self.site.dish_diameter:.1f} m"
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
                    handles.append(
                        mpatches.Patch(color=style["color"], label=t.title())
                    )
                    seen.add(t)

        # Galactic plane entry
        handles.append(
            mpatches.Patch(color="#7B68EE", label="Galactic Plane", alpha=0.6)
        )

        # Beam entry
        if self.beam_target is not None:
            handles.append(
                mpatches.Patch(color="#00BFFF",
                               label=f"Beam FWHM {self.site.beam_fwhm_deg:.2f}°",
                               alpha=0.7)
            )

        if handles:
            self.ax.legend(
                handles=handles,
                loc="lower right",
                fontsize=7,
                framealpha=0.3,
                labelcolor=self.TEXT_COLOR,
                facecolor=self.DARK_BG,
                edgecolor=self.GRID_COLOR,
            )