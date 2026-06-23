"""
planner.py
Observation planner: elevation vs time plots, schedule table,
and sensitivity calculator display.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from astropy.time import Time
import astropy.units as u

from core.observer import ObserverSite
from core.catalog import RadioSource, RadioCatalog
from core.ephemeris import Ephemeris, VisibilityWindow


# ---------------------------------------------------------------------------
# Elevation vs Time plot
# ---------------------------------------------------------------------------

DARK_BG = "#0D1117"
TEXT_COLOR = "#E2E8F0"
GRID_COLOR = "#2D3748"

# Distinct colors for up to 12 sources
TRACK_COLORS = [
    "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF", "#C3A6FF",
    "#FF8B94", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD",
    "#98FB98", "#87CEEB",
]


class ElevationPlot:
    """
    Multi-source elevation vs. time chart.
    Shows which sources are above the horizon limit over a 24-hour window.
    """

    def __init__(self, site: ObserverSite, figsize=(12, 6)):
        self.site = site
        self.figsize = figsize
        self.eph = Ephemeris(site)

    def render(
        self,
        sources: List[RadioSource],
        start_time: Time,
        duration_hours: float = 24.0,
        max_sources: int = 12,
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize, facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)

        sources = sources[:max_sources]

        for i, src in enumerate(sources):
            color = TRACK_COLORS[i % len(TRACK_COLORS)]
            hours, elevs = self.eph.elevation_track(src.coord, start_time,
                                                     duration_hours)
            ax.plot(hours, elevs, color=color, linewidth=1.8,
                    label=src.name, alpha=0.9)

        # Horizon limit shading
        min_el = self.site.min_elevation
        ax.axhline(y=min_el, color="#4A7A4A", linewidth=1.5,
                   linestyle="--", alpha=0.8, zorder=5)
        ax.fill_between([0, duration_hours], [min_el, min_el], [0, 0],
                        color="#1A2A1A", alpha=0.6, zorder=0)
        ax.axhline(y=0, color="#333333", linewidth=0.5)

        # Zenith line
        ax.axhline(y=90, color="#555555", linewidth=0.5, linestyle=":")

        # X-axis: UTC hours
        ax.set_xlim(0, duration_hours)
        ax.set_ylim(-5, 95)
        ax.set_xlabel("UTC Hours from Start", color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("Elevation (°)", color=TEXT_COLOR, fontsize=10)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(15))
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.4, alpha=0.6)

        # Annotate start time
        start_label = f"Start: {start_time.iso[:16]} UTC"
        ax.text(0.01, 0.97, start_label, transform=ax.transAxes,
                color=TEXT_COLOR, fontsize=8, va="top", fontfamily="monospace")
        ax.text(0.01, 0.92, f"Site: {self.site.name}", transform=ax.transAxes,
                color=TEXT_COLOR, fontsize=8, va="top")
        ax.text(duration_hours * 0.55, min_el + 1.5,
                f"Horizon limit {min_el:.0f}°",
                color="#4A7A4A", fontsize=8, alpha=0.9)

        ax.legend(
            loc="upper right", fontsize=7, framealpha=0.3,
            labelcolor=TEXT_COLOR, facecolor=DARK_BG,
            edgecolor=GRID_COLOR, ncol=max(1, len(sources) // 8),
        )
        fig.suptitle("Elevation vs. Time", color=TEXT_COLOR, fontsize=11, y=1.01)
        plt.tight_layout()
        return fig

    def show(self, sources, start_time, duration_hours=24.0) -> None:
        self.render(sources, start_time, duration_hours)
        plt.show()

    def save(self, path: str, sources, start_time,
             duration_hours=24.0, dpi=150) -> None:
        fig = self.render(sources, start_time, duration_hours)
        fig.savefig(path, dpi=dpi, facecolor=DARK_BG, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Text schedule table
# ---------------------------------------------------------------------------

class ObservationSchedule:
    """
    Generates a plain-text or rich observation schedule table.
    """

    def __init__(self, site: ObserverSite):
        self.site = site
        self.eph = Ephemeris(site)

    def build(
        self,
        sources: List[RadioSource],
        start_time: Time,
        duration_hours: float = 24.0,
    ) -> List[VisibilityWindow]:
        return self.eph.multi_source_schedule(sources, start_time, duration_hours)

    def print_table(
        self,
        sources: List[RadioSource],
        start_time: Time,
        duration_hours: float = 24.0,
    ) -> None:
        windows = self.build(sources, start_time, duration_hours)
        header = (
            f"\n{'Source':<22} {'Rise (UTC)':<18} {'Transit (UTC)':<18} "
            f"{'Set (UTC)':<18} {'Peak El':>8} {'Up (h)':>7}"
        )
        sep = "-" * len(header)
        print(f"\nObservation Schedule — {self.site.name}")
        print(f"Start: {start_time.iso[:16]} UTC  |  Window: {duration_hours:.0f}h")
        print(sep)
        print(header)
        print(sep)
        for w in windows:
            if w.always_up:
                print(f"{w.source_name:<22} {'(always visible)':<18} "
                      f"{w.transit_time.iso[:16] if w.transit_time else '---':<18} "
                      f"{'---':<18} {w.peak_elevation_deg:>7.1f}° {'24.0':>6}h")
            elif w.never_rises:
                print(f"{w.source_name:<22} {'(never rises)':<18} "
                      f"{'---':<18} {'---':<18} {w.peak_elevation_deg:>7.1f}° {'0.0':>6}h")
            else:
                r = w.rise_time.iso[:16] if w.rise_time else "---"
                tr = w.transit_time.iso[:16] if w.transit_time else "---"
                s = w.set_time.iso[:16] if w.set_time else "---"
                print(f"{w.source_name:<22} {r:<18} {tr:<18} "
                      f"{s:<18} {w.peak_elevation_deg:>7.1f}° {w.duration_hours():>6.1f}h")
        print(sep)


# ---------------------------------------------------------------------------
# Sensitivity calculator
# ---------------------------------------------------------------------------

class SensitivityCalculator:
    """Quick sensitivity / integration-time calculator display."""

    def __init__(self, site: ObserverSite):
        self.site = site

    def report(
        self,
        bandwidth_mhz: float = 10.0,
        integration_s: float = 300.0,
        target_snr: float = 5.0,
        target_flux_mjy: Optional[float] = None,
    ) -> str:
        rms = self.site.sensitivity_mjy(bandwidth_mhz, integration_s)
        sefd = self.site.sefd_jy
        beam = self.site.beam_fwhm_deg

        lines = [
            "\n=== Sensitivity Report ===",
            f"  Site         : {self.site.name}",
            f"  Dish diam    : {self.site.dish_diameter:.1f} m",
            f"  Frequency    : {self.site.frequency_mhz:.1f} MHz",
            f"  System Temp  : {self.site.system_temp_k:.0f} K",
            f"  SEFD         : {sefd:.0f} Jy",
            f"  Beam FWHP    : {beam:.2f}°  ({beam * 60:.1f}')",
            "",
            f"  Bandwidth    : {bandwidth_mhz:.1f} MHz",
            f"  Integration  : {integration_s:.0f} s",
            f"  1σ RMS       : {rms:.3f} mJy",
            f"  {target_snr:.0f}σ limit     : {target_snr * rms:.3f} mJy",
        ]

        if target_flux_mjy is not None:
            req_t = (self.site.sefd_jy * 1e3 / target_flux_mjy) ** 2 \
                    / (2 * bandwidth_mhz * 1e6)
            lines.append(f"\n  To detect {target_flux_mjy:.1f} mJy at {target_snr:.0f}σ:")
            lines.append(f"  Required tint: {req_t:.1f} s  ({req_t/60:.1f} min)")

        lines.append("=" * 28)
        return "\n".join(lines)

    def print_report(self, **kwargs) -> None:
        print(self.report(**kwargs))