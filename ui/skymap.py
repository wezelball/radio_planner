"""
skymap.py
Interactive sky map using matplotlib. Supports AltAz (horizon) and
RA/Dec (equatorial) projections with source overlays.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from astropy.coordinates import SkyCoord, AltAz, ICRS
from astropy.time import Time
import astropy.units as u

from core.observer import ObserverSite
from core.catalog import RadioSource, RadioCatalog


# ---------------------------------------------------------------------------
# Color / marker scheme for source types
# ---------------------------------------------------------------------------

SOURCE_STYLES = {
    "supernova remnant": dict(color="#FF6B6B", marker="*", size=14),
    "radio galaxy":      dict(color="#4ECDC4", marker="D", size=10),
    "galactic center":   dict(color="#FFE66D", marker="X", size=14),
    "hii region":        dict(color="#A8E6CF", marker="^", size=10),
    "pulsar":            dict(color="#FF8B94", marker="p", size=12),
    "galaxy":            dict(color="#C3A6FF", marker="o", size=9),
    "solar":             dict(color="#FFD700", marker="o", size=18),
    "continuum":         dict(color="#FFFFFF", marker="o", size=8),
    "galactic":          dict(color="#98D8C8", marker="s", size=8),
    "default":           dict(color="#AAAAAA", marker="o", size=7),
}


def _source_style(source_type: str) -> dict:
    for key, style in SOURCE_STYLES.items():
        if key in source_type.lower():
            return style
    return SOURCE_STYLES["default"]


# ---------------------------------------------------------------------------
# SkyMap class
# ---------------------------------------------------------------------------

class SkyMap:
    """
    Renders an interactive matplotlib sky map.

    Modes
    -----
    'altaz'  — horizon view (azimuth vs elevation, polar projection)
    'radec'  — equatorial view (RA vs Dec, rectangular or Mollweide)
    """

    DARK_BG = "#0D1117"
    GRID_COLOR = "#2D3748"
    HORIZON_COLOR = "#2A4A2A"
    TEXT_COLOR = "#E2E8F0"

    def __init__(
        self,
        site: ObserverSite,
        catalog: Optional[RadioCatalog] = None,
        mode: str = "altaz",
        figsize: Tuple[float, float] = (10, 9),
    ):
        self.site = site
        self.catalog = catalog
        self.mode = mode
        self.figsize = figsize
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self._click_callback = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, time: Optional[Time] = None) -> Figure:
        """Render the sky map for the given time (defaults to now)."""
        t = time or Time.now()
        self.fig = plt.figure(figsize=self.figsize, facecolor=self.DARK_BG)

        if self.mode == "altaz":
            self._render_altaz(t)
        else:
            self._render_radec(t)

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

    def on_click(self, callback) -> None:
        """Register a callback: callback(ra_deg, dec_deg, source_name_or_None)"""
        self._click_callback = callback

    # ------------------------------------------------------------------
    # AltAz (polar horizon) view
    # ------------------------------------------------------------------

    def _render_altaz(self, time: Time) -> None:
        self.ax = self.fig.add_subplot(111, projection="polar",
                                       facecolor=self.DARK_BG)
        ax = self.ax

        # Polar: θ = azimuth (N=0, E=90), r = zenith angle (90-elevation)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)          # clockwise like compass
        ax.set_rlim(0, 90)
        ax.set_yticks([0, 30, 60, 90])
        ax.set_yticklabels(["90°", "60°", "30°", "0°"],
                           color=self.TEXT_COLOR, fontsize=8)
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                           color=self.TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=self.TEXT_COLOR)
        ax.grid(color=self.GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)

        # Horizon minimum elevation line
        min_el = self.site.min_elevation
        theta = np.linspace(0, 2 * np.pi, 360)
        r_limit = 90 - min_el
        ax.fill_between(theta, r_limit, 90,
                        color=self.HORIZON_COLOR, alpha=0.3, zorder=1)
        ax.plot(theta, np.full_like(theta, r_limit),
                color="#4A7A4A", linewidth=1, linestyle="-", alpha=0.8)

        # Plot sources
        if self.catalog:
            self._plot_sources_altaz(ax, time)

        # Galactic plane approximation
        self._draw_galactic_plane_altaz(ax, time)

    def _plot_sources_altaz(self, ax: Axes, time: Time) -> None:
        frame = self.site.altaz_frame(time)
        plotted_types = set()

        for src in self.catalog:
            altaz = src.coord.transform_to(frame)
            el = float(altaz.alt.deg)
            az = float(altaz.az.deg)

            style = _source_style(src.source_type)
            r = 90 - el
            theta = np.radians(az)
            alpha = 1.0 if el >= self.site.min_elevation else 0.3

            ax.scatter(theta, r,
                       c=style["color"],
                       marker=style["marker"],
                       s=style["size"] ** 2,
                       alpha=alpha,
                       zorder=3,
                       edgecolors="none")

            if el >= self.site.min_elevation:
                ax.annotate(
                    src.name,
                    (theta, r),
                    fontsize=7,
                    color=style["color"],
                    xytext=(6, 3),
                    textcoords="offset points",
                    alpha=0.9,
                )
            plotted_types.add(src.source_type)

    def _draw_galactic_plane_altaz(self, ax: Axes, time: Time) -> None:
        """Approximate galactic plane trace in AltAz."""
        gal_l = np.linspace(0, 360, 720) * u.deg
        gal_b = np.zeros(720) * u.deg
        from astropy.coordinates import Galactic
        gal_coords = SkyCoord(l=gal_l, b=gal_b, frame=Galactic)
        icrs_coords = gal_coords.icrs
        frame = self.site.altaz_frame(time)
        altaz = icrs_coords.transform_to(frame)

        above = altaz.alt.deg > 0
        az_rad = np.radians(altaz.az.deg[above])
        r = 90 - altaz.alt.deg[above]

        if len(az_rad) > 10:
            # Sort by az to avoid scrambled line
            order = np.argsort(az_rad)
            ax.plot(az_rad[order], r[order],
                    color="#7B68EE", linewidth=0.8, alpha=0.5,
                    linestyle=":", zorder=2)

    # ------------------------------------------------------------------
    # RA/Dec (equatorial) view
    # ------------------------------------------------------------------

    def _render_radec(self, time: Time) -> None:
        self.ax = self.fig.add_subplot(111, facecolor=self.DARK_BG)
        ax = self.ax

        ax.set_xlim(360, 0)      # RA increases right-to-left
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Right Ascension (°)", color=self.TEXT_COLOR, fontsize=15)
        ax.set_ylabel("Declination (°)", color=self.TEXT_COLOR, fontsize=15)
        ax.tick_params(colors=self.TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)
        ax.grid(color=self.GRID_COLOR, linestyle="--", linewidth=0.4, alpha=0.7)

        # Dec limit lines
        max_dec = 90 - self.site.latitude + self.site.min_elevation
        min_dec = -90 + self.site.latitude + self.site.min_elevation
        ax.axhline(y=min_dec, color="#4A7A4A", linewidth=1, linestyle="--",
                   alpha=0.8, label=f"Min observable dec ({min_dec:.0f}°)")
        ax.axhline(y=max_dec, color="#4A7A4A", linewidth=1, linestyle="--",
                   alpha=0.8)

        # Galactic plane
        self._draw_galactic_plane_radec(ax)

        # Sources
        if self.catalog:
            self._plot_sources_radec(ax, time)

    def _plot_sources_radec(self, ax: Axes, time: Time) -> None:
        frame = self.site.altaz_frame(time)
        for src in self.catalog:
            altaz = src.coord.transform_to(frame)
            el = float(altaz.alt.deg)
            style = _source_style(src.source_type)
            alpha = 1.0 if el >= self.site.min_elevation else 0.25
            fc = style["color"] if el >= self.site.min_elevation else "#555555"

            ax.scatter(src.ra_deg, src.dec_deg,
                       c=fc, marker=style["marker"],
                       s=style["size"] ** 2, alpha=alpha,
                       zorder=3, edgecolors="none")
            if el >= self.site.min_elevation:
                ax.annotate(src.name, (src.ra_deg, src.dec_deg),
                            fontsize=7, color=style["color"],
                            xytext=(5, 3), textcoords="offset points")

    def _draw_galactic_plane_radec(self, ax: Axes) -> None:
        from astropy.coordinates import Galactic
        gal_l = np.linspace(0, 360, 720) * u.deg
        gal_b = np.zeros(720) * u.deg
        coords = SkyCoord(l=gal_l, b=gal_b, frame=Galactic).icrs
        # Unwrap to avoid jumps across RA=0/360
        ra = coords.ra.deg
        dec = coords.dec.deg
        order = np.argsort(ra)
        ax.plot(ra[order], dec[order],
                color="#7B68EE", linewidth=1, alpha=0.5,
                linestyle=":", zorder=2, label="Galactic plane")

    # ------------------------------------------------------------------
    # Decorations
    # ------------------------------------------------------------------

    def _add_title(self, time: Time) -> None:
        title = (f"{self.site.name}  |  {time.iso[:16]} UTC  |  "
                 f"Lat={self.site.latitude:+.2f}°  Lon={self.site.longitude:+.2f}°  |  "
                 f"{self.site.frequency_mhz:.1f} MHz")
        self.fig.suptitle(title, color=self.TEXT_COLOR, fontsize=18,
                          y=0.98, fontfamily="monospace")

    def _add_legend(self) -> None:
        handles = []
        seen = set()
        if self.catalog:
            for src in self.catalog:
                t = src.source_type
                if t not in seen:
                    style = _source_style(t)
                    handles.append(
                        mpatches.Patch(color=style["color"], label=t.title())
                    )
                    seen.add(t)
        if handles:
            leg = self.ax.legend(
                handles=handles,
                loc="lower right" if self.mode == "radec" else "upper right",
                fontsize=7,
                framealpha=0.3,
                labelcolor=self.TEXT_COLOR,
                facecolor=self.DARK_BG,
                edgecolor=self.GRID_COLOR,
            )
