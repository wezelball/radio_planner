"""
main.py
Radio Astronomy Observation Planner — main entry point.

Usage
-----
    python main.py                        # demo run with default settings
    python main.py --site green_bank      # use a known site
    python main.py --lat 38.4 --lon -79.8 --elev 880
    python main.py --mode radec           # equatorial sky map
    python main.py --freq 408             # change observing frequency
    python main.py --save-map map.png --save-elev elev.png
"""

from __future__ import annotations

import argparse
import sys
import os
from datetime import datetime, timezone

# Ensure the project root is on sys.path so `core` and `ui` are importable
# when running main.py directly (e.g. `python main.py` or `python3 main.py`)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astropy.time import Time

from core.observer import ObserverSite, KNOWN_SITES, now_utc
from core.catalog import default_catalog, RadioCatalog
from core.ephemeris import Ephemeris, DriftScanPredictor, BeamTransit
from ui.skymap import SkyMap
from ui.planner import ElevationPlot, ObservationSchedule, SensitivityCalculator


# ---------------------------------------------------------------------------
# Tee: write to stdout and a file simultaneously
# ---------------------------------------------------------------------------

class _Tee:
    """Mirrors stdout to a file, excluding render/plot status lines."""

    EXCLUDE = (
        "Rendering sky map",
        "Rendering elevation plot",
        "[skymap] Generating GSM",
    )

    def __init__(self, filepath: str):
        self._stdout = sys.stdout
        self._file   = open(filepath, "w", encoding="utf-8")
        self._buf    = ""
        sys.stdout   = self

    def write(self, text: str) -> None:
        self._stdout.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(ex in line for ex in self.EXCLUDE):
                self._file.write(line + "\n")

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        if self._buf and not any(ex in self._buf for ex in self.EXCLUDE):
            self._file.write(self._buf + "\n")
        self._file.close()
        sys.stdout = self._stdout


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Radio Astronomy Observation Planner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Site
    site_grp = p.add_mutually_exclusive_group()
    site_grp.add_argument("--site", choices=list(KNOWN_SITES),
                          help="Use a known observatory site")
    p.add_argument("--lat", type=float, default=38.43,
                   help="Observer latitude (deg, +N)")
    p.add_argument("--lon", type=float, default=-79.84,
                   help="Observer longitude (deg, +E)")
    p.add_argument("--elev", type=float, default=880.0,
                   help="Observer elevation (m)")
    p.add_argument("--name", type=str, default="My Site",
                   help="Site name")
    p.add_argument("--min-el", type=float, default=10.0,
                   help="Minimum observable elevation (deg)")

    # Antenna
    p.add_argument("--dish", type=float, default=3.0,
                   help="Dish diameter (m)")
    p.add_argument("--freq", type=float, default=1420.0,
                   help="Observing frequency (MHz)")
    p.add_argument("--tsys", type=float, default=100.0,
                   help="System temperature (K)")

    # Receiver / sensitivity
    p.add_argument("--bandwidth", type=float, default=1.0,
                   help="Receiver noise bandwidth (MHz). "
                        "Use ~1-2 MHz for HI spectral line, 10+ MHz for continuum")
    p.add_argument("--integration", type=float, default=60.0,
                   help="Software integration time per sample (seconds). "
                        "Longer = lower noise floor")
    p.add_argument("--target-snr", type=float, default=5.0,
                   help="Target signal-to-noise ratio for sensitivity calculation")
    p.add_argument("--target-flux", type=float, default=None,
                   help="Check detectability of a specific source flux (mJy). "
                        "Optional — omit to skip")

    # Time
    p.add_argument("--time", type=str, default=None,
                   help="Start time ISO UTC (e.g. '2024-06-01 22:00:00'). Default: now")
    p.add_argument("--duration", type=float, default=24.0,
                   help="Planning window in hours")

    # Display
    p.add_argument("--mode", choices=["altaz", "radec"], default="altaz",
                   help="Sky map projection")
    p.add_argument("--no-map", action="store_true",
                   help="Skip sky map display")
    p.add_argument("--no-elev", action="store_true",
                   help="Skip elevation plot display")
    p.add_argument("--no-schedule", action="store_true",
                   help="Skip text schedule table")
    p.add_argument("--save-map", type=str, default=None,
                   help="Save sky map to file")
    p.add_argument("--save-elev", type=str, default=None,
                   help="Save elevation plot to file")

    # Catalog
    p.add_argument("--catalog", type=str, default=None,
                   help="Path to custom CSV catalog file")
    p.add_argument("--min-flux", type=float, default=0.0,
                   help="Minimum source flux (Jy) to show")

    # Beam pointing (drift scan — fixed Az/El)
    p.add_argument("--beam-az", type=float, default=180.0,
                   help="Dish azimuth (deg, 0=N 90=E). Default: 180 (South)")
    p.add_argument("--beam-el", type=float, default=45.0,
                   help="Dish elevation (deg above horizon). Default: 45")
    p.add_argument("--drift-hours", type=float, default=24.0,
                   help="Hours of drift trail to draw on the sky map. Default: 24")
    p.add_argument("--no-beam", action="store_true",
                   help="Do not draw the beam footprint on the sky map")
    p.add_argument("--no-background", action="store_true",
                   help="Skip the GSM radio sky brightness background")
    p.add_argument("--no-transits", action="store_true",
                   help="Skip drift-scan beam transit predictions")
    p.add_argument("--min-response", type=float, default=0.0,
                   help="Only show transits with beam response >= this value (0-1)")
    p.add_argument("--report", type=str, default=None, metavar="FILE",
                   help="Save text report to FILE (e.g. --report session.txt)")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- Report file ---
    tee = None
    if args.report:
        tee = _Tee(args.report)
        print(f"# Observation Report")
        print(f"# Generated: {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"# Command:   {' '.join(__import__('sys').argv)}\n")

    # --- Build site ---
    if args.site:
        import copy
        site = copy.copy(KNOWN_SITES[args.site])
        # Only apply CLI flags when the user explicitly passed them
        explicitly_set = {a.dest for a in parser._actions if a.default is not None
                          and getattr(args, a.dest, None) != a.default}
        if "freq"   in explicitly_set: site.frequency_mhz = args.freq
        if "dish"   in explicitly_set: site.dish_diameter = args.dish
        if "tsys"   in explicitly_set: site.system_temp_k = args.tsys
        if "min_el" in explicitly_set: site.min_elevation = args.min_el
    else:
        site = ObserverSite(
            name=args.name,
            latitude=args.lat,
            longitude=args.lon,
            elevation=args.elev,
            min_elevation=args.min_el,
            dish_diameter=args.dish,
            frequency_mhz=args.freq,
            system_temp_k=args.tsys,
        )

    print(f"\n{'='*60}")
    print(f"  Radio Observation Planner")
    print(f"{'='*60}")
    print(f"  Site      : {site.name}")
    print(f"  Location  : {site.latitude:+.4f}°N  {site.longitude:+.4f}°E  "
          f"{site.elevation:.0f}m")
    print(f"  Frequency : {site.frequency_mhz:.1f} MHz")
    print(f"  Dish      : {site.dish_diameter:.1f} m  "
          f"(beam FWHM = {site.beam_fwhm_deg:.2f}°)")
    print(f"  SEFD      : {site.sefd_jy:.0f} Jy")
    print(f"{'='*60}\n")

    # --- Parse time ---
    if args.time:
        start_time = Time(args.time, scale="utc")
    else:
        start_time = now_utc()
    print(f"Planning window: {start_time.iso[:16]} UTC  +{args.duration:.0f}h\n")

    # --- Build catalog ---
    if args.catalog:
        catalog = RadioCatalog.from_csv(args.catalog)
        print(f"Loaded catalog '{catalog.name}': {len(catalog)} sources")
    else:
        catalog = default_catalog()
        print(f"Using built-in bright source catalog: {len(catalog)} sources")

    if args.min_flux > 0:
        catalog = catalog.by_flux_min(args.min_flux, site.frequency_mhz)
        print(f"After flux filter (>{args.min_flux} Jy): {len(catalog)} sources")

    sources = list(catalog)

    # --- Sensitivity report ---
    calc = SensitivityCalculator(site)
    calc.print_report(bandwidth_mhz=args.bandwidth,
                      integration_s=args.integration,
                      target_snr=args.target_snr,
                      target_flux_mjy=args.target_flux)

    # --- Schedule table ---
    if not args.no_schedule:
        sched = ObservationSchedule(site)
        sched.print_table(sources, start_time, args.duration)

    # --- Drift-scan transit predictions ---
    transits = []
    if not args.no_transits and not args.no_beam:
        predictor = DriftScanPredictor(site, beam_az=args.beam_az,
                                       beam_el=args.beam_el)
        transits = predictor.print_transits(
            sources, start_time,
            duration_hours=args.drift_hours,
            min_response=args.min_response,
        )

    # --- Beam (drift scan: fixed Az/El) ---
    beam_az = beam_el = None
    if not getattr(args, "no_beam", False):
        beam_az = args.beam_az
        beam_el = args.beam_el
        print(f"Dish pointing: Az={beam_az:.1f}°  El={beam_el:.1f}°  "
              f"(drift scan, FWHM={site.beam_fwhm_deg:.2f}°)")

    # --- Sky map ---
    if not args.no_map:
        smap = SkyMap(site, catalog, mode=args.mode,
                      beam_az=beam_az, beam_el=beam_el,
                      drift_hours=args.drift_hours,
                      transits=transits,
                      show_background=not args.no_background)
        if args.save_map:
            smap.save(args.save_map, start_time)
            print(f"\nSky map saved to: {args.save_map}")
        else:
            print("\nRendering sky map... (close window to continue)")
            smap.show(start_time)

    # --- Elevation plot ---
    if not args.no_elev:
        elev_plot = ElevationPlot(site)
        if args.save_elev:
            elev_plot.save(args.save_elev, sources, start_time, args.duration)
            print(f"Elevation plot saved to: {args.save_elev}")
        else:
            print("Rendering elevation plot... (close window to continue)")
            elev_plot.show(sources, start_time, args.duration)

    print("\nDone.")

    if tee is not None:
        tee.close()
        # final message goes direct to real stdout
        tee._stdout.write(f"Report saved to: {args.report}\n")


if __name__ == "__main__":
    main()