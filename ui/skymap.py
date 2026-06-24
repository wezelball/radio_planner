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
        transits: Optional[list] = None,
        show_background: bool = True,
        figsize: Tuple[float, float] = (14, 7),
    ):
        self.site            = site
        self.catalog         = catalog
        self.mode            = mode
        self.beam_az         = beam_az
        self.beam_el         = beam_el
        self.drift_hours     = drift_hours
        self.transits        = transits or []
        self.show_background = show_background
        self.figsize         = figsize
        self._bg_cache: dict = {}   # keyed by freq_mhz -> np.ndarray
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
        Sample the FWHM beam circle on the sky and return it in RA/Dec.

        Uses SkyCoord.directional_offset_by() to walk great-circle offsets
        of radius = FWHM/2 around the beam centre at evenly-spaced position
        angles.  This is geometrically exact — no flat-sky approximation —
        so the boundary is always a clean closed ellipse on the sky regardless
        of elevation or proximity to the poles.
        """
        fwhm   = self.site.beam_fwhm_deg
        radius = (fwhm / 2.0) * u.deg

        # Convert the fixed Az/El pointing to RA/Dec at this instant
        centre = self._altaz_to_radec(az_deg, el_deg, time)

        # Sample position angles 0→360° and step one FWHM/2 radius along each
        position_angles = np.linspace(0, 360, n, endpoint=False) * u.deg
        boundary = centre.directional_offset_by(position_angles, radius)
        return boundary.ra.deg, boundary.dec.deg

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

        # Sky brightness background (drawn first, behind everything)
        has_background = self._draw_sky_background(ax)

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

        # Galactic plane line: skip when background map is shown (plane visible in map)
        if not has_background:
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
    # Sky brightness background
    # ------------------------------------------------------------------

    def _draw_sky_background(self, ax: Axes) -> bool:
        """
        Render the Global Sky Model as a colour-mapped background image.
        Returns True if the background was drawn, False if unavailable.
        The image is cached after the first call so re-renders are fast.
        """
        if not self.show_background:
            return False

        # Use cached image if available
        freq = self.site.frequency_mhz
        if freq not in self._bg_cache:
            print(f"  [skymap] Generating GSM background at {freq:.0f} MHz "
                  f"(this takes ~10-20s) ... ", end="", flush=True)
            img = build_gsm_background(freq)
            if img is None:
                print("pygdsm/healpy not installed — skipping background.")
                print("  Install with: pip install pygdsm healpy")
                return False
            self._bg_cache[freq] = img
            print(f"done  (peak {img.max():.0f} K, min {img.min():.0f} K)")

        img = self._bg_cache[freq]

        # Log-scale for dynamic range, then percentile-clip for contrast
        img_log = np.log10(np.clip(img, 1.0, None))

        # Clip to 2nd–99.5th percentile so the galactic centre doesn't
        # swamp everything and the cold polar sky isn't pure black
        vmin = float(np.percentile(img_log, 2.0))
        vmax = float(np.percentile(img_log, 99.5))

        ax.imshow(
            img_log,
            origin="lower",
            extent=[0, 360, -90, 90],
            aspect="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            alpha=0.90,
            zorder=0,
            interpolation="bilinear",
        )

        # Colorbar with actual Kelvin tick labels
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm   = cm.ScalarMappable(cmap="inferno", norm=norm)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label("log₁₀ T_sky (K)", color=self.TEXT_COLOR, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=self.TEXT_COLOR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=self.TEXT_COLOR, fontsize=7)
        # Annotate a few reference temperatures in Kelvin
        import matplotlib.ticker as mticker
        tick_k = [10, 30, 100, 300, 1000, 3000, 10000]
        tick_log = [np.log10(k) for k in tick_k if vmin <= np.log10(k) <= vmax]
        tick_labels = [f"{k}K" for k in tick_k if vmin <= np.log10(k) <= vmax]
        cbar.set_ticks(tick_log)
        cbar.set_ticklabels(tick_labels)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=self.TEXT_COLOR, fontsize=7)
        return True

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

        # Transit markers
        self._draw_transit_markers(ax, time)

    # ------------------------------------------------------------------
    # Transit markers
    # ------------------------------------------------------------------

    def _draw_transit_markers(self, ax: Axes, time: Time) -> None:
        """
        Mark each BeamTransit's peak position on the drift trail.
        Converts the peak_time Az/El to RA/Dec and plots a labelled dot.
        """
        if not self.transits or self.beam_az is None or self.beam_el is None:
            return

        first = True
        for t in self.transits:
            if t.peak_time is None:
                continue

            # RA/Dec of beam centre at peak crossing time
            centre = self._altaz_to_radec(self.beam_az, self.beam_el, t.peak_time)
            ra_m  = float(centre.ra.deg)
            dec_m = float(centre.dec.deg)

            # Colour by response: bright green for strong, yellow for weak
            color = "#00FF88" if t.is_detected() else "#FFD700"
            size  = 60 if t.is_detected() else 35

            ax.scatter([ra_m], [dec_m], c=color, s=size,
                       zorder=9, edgecolors="white", linewidths=0.5,
                       label="Transit (≥50% beam)" if (first and t.is_detected())
                             else ("Transit (<50% beam)" if (first and not t.is_detected())
                                   else "_nolegend_"))

            # Label: source name + peak UTC time
            peak_str = t.peak_time.iso[11:16]   # HH:MM
            cos_dec  = max(abs(np.cos(np.radians(dec_m))), 0.01)
            offset_ra = self.site.beam_fwhm_deg * 0.4 / cos_dec
            ax.annotate(
                f"{t.source_name}\n{peak_str} UTC\n"
                f"resp={t.peak_response:.2f} dur={t.transit_duration_min:.1f}m",
                (ra_m, dec_m),
                xytext=(ra_m + offset_ra, dec_m + self.site.beam_fwhm_deg * 0.35),
                fontsize=6.5, color=color, alpha=0.95, zorder=10,
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.5, lw=0.8),
            )
            first = False

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
        if any(t.is_detected() for t in self.transits):
            handles.append(mpatches.Patch(color="#00FF88", alpha=0.9,
                                          label="Transit ≥50% beam"))
        if any(not t.is_detected() for t in self.transits):
            handles.append(mpatches.Patch(color="#FFD700", alpha=0.9,
                                          label="Transit <50% beam"))

        if handles:
            self.ax.legend(
                handles=handles, loc="lower right", fontsize=7,
                framealpha=0.3, labelcolor=self.TEXT_COLOR,
                facecolor=self.DARK_BG, edgecolor=self.GRID_COLOR,
            )


# ---------------------------------------------------------------------------
# Sky background module (appended)
# ---------------------------------------------------------------------------

def build_gsm_background(freq_mhz: float = 1420.0,
                          ra_pixels: int = 1440,
                          dec_pixels: int = 720,
                          nside: int = 512) -> Optional[np.ndarray]:
    """
    Generate a plate-carrée brightness temperature map from the Global Sky Model.

    Parameters
    ----------
    freq_mhz   : observing frequency in MHz
    ra_pixels  : number of pixels along RA axis (0–360°)
    dec_pixels : number of pixels along Dec axis (-90–+90°)
    nside      : HEALPix resolution (512 → ~0.11° per pixel, smooth result)

    Returns
    -------
    2D numpy array of shape (dec_pixels, ra_pixels) in Kelvin,
    or None if pygdsm / healpy are not installed.
    """
    try:
        import healpy as hp
        from pygdsm import GlobalSkyModel
    except ImportError:
        return None

    # Generate the GSM map at the requested frequency.
    # IMPORTANT: pygdsm returns the map in galactic coordinates (l, b),
    # stored as a HEALPix RING-ordered array where phi=0 is the galactic
    # centre, NOT RA=0.  We must convert each output pixel's RA/Dec to
    # galactic (l, b) before indexing the map.
    gsm = GlobalSkyModel()
    gsm_map = gsm.generate(freq_mhz)   # HEALPix RING, galactic coords, Kelvin

    # Upgrade/degrade to the requested nside
    gsm_map = hp.ud_grade(gsm_map, nside, order_in='RING', order_out='RING')

    # Build a regular RA/Dec pixel grid (plate carrée)
    ra_grid  = np.linspace(0, 360, ra_pixels,  endpoint=False)   # deg
    dec_grid = np.linspace(-90, 90, dec_pixels, endpoint=True)    # deg
    ra_2d, dec_2d = np.meshgrid(ra_grid, dec_grid)

    # Convert RA/Dec → galactic (l, b) via astropy — this is the key fix.
    # Doing it in chunks avoids building a single enormous SkyCoord array.
    from astropy.coordinates import SkyCoord, Galactic
    import astropy.units as u_ap

    chunk = 180   # process this many Dec rows at a time
    l_2d  = np.empty_like(ra_2d)
    b_2d  = np.empty_like(ra_2d)

    for row_start in range(0, dec_pixels, chunk):
        row_end = min(row_start + chunk, dec_pixels)
        coords = SkyCoord(
            ra=ra_2d[row_start:row_end].ravel() * u_ap.deg,
            dec=dec_2d[row_start:row_end].ravel() * u_ap.deg,
            frame='icrs',
        ).galactic
        l_2d[row_start:row_end] = coords.l.deg.reshape(row_end - row_start, ra_pixels)
        b_2d[row_start:row_end] = coords.b.deg.reshape(row_end - row_start, ra_pixels)

    # healpy convention: theta = colatitude (0=N pole), phi = longitude 0→2π
    # galactic b runs -90→+90, galactic l runs 0→360
    theta_gal = np.radians(90.0 - b_2d)   # colatitude from galactic b
    phi_gal   = np.radians(l_2d)           # galactic longitude

    pix   = hp.ang2pix(nside, theta_gal, phi_gal, nest=False)
    image = gsm_map[pix]                   # shape: (dec_pixels, ra_pixels)

    return image.reshape(dec_pixels, ra_pixels)