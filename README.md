# Radio Observation Planner

A Python application for planning radio astronomy observations, inspired by Radio Eyes.

## Features

- **Sky Map** — AltAz (horizon polar) and RA/Dec (equatorial) projections with source overlays
- **Elevation vs. Time** — Multi-source elevation tracks over a configurable time window
- **Observation Schedule** — Rise/transit/set times and visibility duration for each source
- **Sensitivity Calculator** — RMS noise, SEFD, beam width, and minimum detectable flux
- **Catalog Management** — Built-in bright source catalog + CSV import + live NVSS queries
- **Multiple Sites** — Built-in presets (Green Bank, VLA, Parkes, Effelsberg, Arecibo) or custom coordinates

---

## Installation

```bash
pip install -r requirements.txt
```

### Optional (for live catalog queries)
```bash
pip install astroquery
```

---

## Quick Start

# Run virtual environment
source .venv/bin/activate

```bash
# Run with defaults (Green Bank site, current time, South-facing dish at 45° elevation)
python main.py --site green_bank

# Custom site with equatorial sky map
python main.py --lat 38.43 --lon -79.84 --name "My Dish" --mode radec

# Change frequency and dish size
python main.py --site vla --freq 408 --dish 25

# Save plots instead of displaying them
python main.py --save-map skymap.png --save-elev elevations.png

# Use a custom catalog, show only sources > 10 Jy
python main.py --catalog my_sources.csv --min-flux 10

# Schedule only, no plots
python main.py --site parkes --no-map --no-elev

# Point east at 30° elevation
python3 main.py --site green_bank --beam-az 90 --beam-el 30

# Show only the next 6 hours of drift
python3 main.py --site green_bank --beam-az 180 --beam-el 60 --drift-hours 6

# No beam overlay
python3 main.py --site green_bank --no-beam
```

---

## Project Structure

```
radio_planner/
├── core/
│   ├── observer.py      # ObserverSite: location, beam, sensitivity
│   ├── catalog.py       # RadioSource, RadioCatalog, NVSS query
│   └── ephemeris.py     # Rise/set/transit, elevation tracks
├── ui/
│   ├── skymap.py        # SkyMap: AltAz & RA/Dec matplotlib rendering
│   └── planner.py       # ElevationPlot, ObservationSchedule, SensitivityCalculator
├── data/
│   ├── catalogs/        # Place custom CSV catalogs here
│   └── rfi/             # RFI frequency lists (future)
├── tests/
│   └── test_core.py     # pytest unit tests
├── main.py              # CLI entry point
├── requirements.txt
└── README.md
```

---

## CSV Catalog Format

```csv
name,ra_deg,dec_deg,flux_jy,ref_freq_mhz,spectral_index,source_type,notes
Cas A,350.867,58.812,2720,1000,-0.77,supernova remnant,Brightest radio source
My Source,123.45,-30.0,5.2,1400,-0.7,continuum,
```

---

## Using the API in Your Own Scripts

```python
from astropy.time import Time
from core.observer import ObserverSite
from core.catalog import default_catalog
from core.ephemeris import Ephemeris
from ui.skymap import SkyMap
from ui.planner import ElevationPlot, ObservationSchedule

# Create your site
site = ObserverSite("My Backyard", latitude=38.43, longitude=-79.84,
                    elevation=880, dish_diameter=3.0, frequency_mhz=1420.0)

# Load catalog
catalog = default_catalog()

# Build schedule for tonight
start = Time("2024-06-01 22:00:00", scale="utc")
sched = ObservationSchedule(site)
sched.print_table(list(catalog), start, duration_hours=8.0)

# Show elevation plot
ep = ElevationPlot(site)
ep.show(list(catalog)[:8], start, duration_hours=8.0)

# Show sky map
sm = SkyMap(site, catalog, mode="altaz")
sm.show(start)
```

---

## Roadmap

- [ ] PyQt6 GUI with live updating sky map
- [ ] HEALPix all-sky background maps (GSM, NVSS density)
- [ ] Antenna beam footprint overlay
- [ ] RFI frequency flagging
- [ ] Pulsar timing / ephemeris support
- [ ] Export to iCal / CSV schedule
- [ ] VLBI baseline planning
- [ ] Spectral line Doppler shift calculator
