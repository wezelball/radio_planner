# Radio Planner

A Python application for planning radio astronomy observations with stationary (drift-scan) dish antennas, inspired by Radio Eyes.

Radio Planner generates an interactive sky map with a Global Sky Model background, predicts which sources will drift through your beam, computes transit times and elevations, and produces observation reports — all from the command line.

---

## Features

- **Interactive sky map** — rectangular RA/Dec projection with real radio sky brightness background from the Global Sky Model (GSM) at any frequency
- **Drift-scan beam overlay** — geometrically correct beam footprint and 24-hour drift trail showing which sources sweep through your dish
- **Source transit predictions** — enter/peak/exit times, beam response, and transit duration for every catalog source
- **Live cursor readout** — hover over the map to see RA/Dec, Az/El, time to transit, transit UTC, elevation at transit, and nearest source
- **Observation schedule** — rise/set/transit table for all sources over a configurable window
- **Elevation vs. time plots** — multi-source elevation tracks for the planning window
- **Sensitivity calculator** — SEFD, beam FWHM, RMS noise floor, and minimum detectable flux for your receiver setup
- **Observation reports** — save all text output to a file for your observation log
- **Custom catalogs** — load your own sources from CSV, or query NVSS live via astroquery

---

## Requirements

- Python 3.10 or later
- A Unix-like or Windows system with a display (for interactive sky map)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/wezelball/radio-planner.git
cd radio-planner
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv

# Activate — Linux / macOS:
source .venv/bin/activate

# Activate — Windows:
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For live catalog queries (optional):
```bash
pip install astroquery
```

---

## Quick Start

```bash
# Run with a built-in site (Green Bank)
python main.py --site green_bank

# Point the dish south at 45° elevation
python main.py --site green_bank --beam-az 180 --beam-el 45

# Use your own site coordinates
python main.py --lat 37.79 --lon -77.92 --elev 116 --dish 0.7 --tsys 150

# Save the sky map and elevation plot instead of displaying them
python main.py --site green_bank --save-map skymap.png --save-elev elev.png

# Save an observation report
python main.py --site green_bank --report session.txt

# Skip the GSM background for a faster plain map
python main.py --site green_bank --no-background
```

---

## Adding Your Own Site

Open `core/observer.py` and add an entry to the `KNOWN_SITES` dictionary:

```python
"my_site": ObserverSite(
    name="My Backyard Dish",
    latitude=37.7906,       # degrees, positive = North
    longitude=-77.9242,     # degrees, positive = East
    elevation=116,          # meters above sea level
    min_elevation=30.0,     # degrees — your antenna's horizon limit
    dish_diameter=0.7,      # meters
    frequency_mhz=1420.0,   # MHz (1420 = HI line)
    system_temp_k=150.0,    # Kelvin
),
```

Then use it with:

```bash
python main.py --site my_site
```

---

## CLI Reference

### Site
| Flag | Default | Description |
|------|---------|-------------|
| `--site NAME` | — | Use a named site from `observer.py` |
| `--lat DEG` | 38.43 | Observer latitude (°, +N) |
| `--lon DEG` | -79.84 | Observer longitude (°, +E) |
| `--elev M` | 880 | Site elevation (metres) |
| `--name STR` | My Site | Site name label |
| `--min-el DEG` | 10.0 | Minimum observable elevation (°) |

### Antenna
| Flag | Default | Description |
|------|---------|-------------|
| `--dish M` | 3.0 | Dish diameter (metres) |
| `--freq MHZ` | 1420.0 | Observing frequency (MHz) |
| `--tsys K` | 100.0 | System temperature (K) |

### Receiver / Sensitivity
| Flag | Default | Description |
|------|---------|-------------|
| `--bandwidth MHZ` | 1.0 | Receiver noise bandwidth (MHz) |
| `--integration S` | 60.0 | Integration time per sample (seconds) |
| `--target-snr N` | 5.0 | Target signal-to-noise ratio |
| `--target-flux MJY` | — | Check detectability of a specific flux (mJy) |

### Beam / Drift Scan
| Flag | Default | Description |
|------|---------|-------------|
| `--beam-az DEG` | 180.0 | Dish azimuth (°, 0=N 90=E) |
| `--beam-el DEG` | 45.0 | Dish elevation (°) |
| `--drift-hours H` | 24.0 | Length of drift trail on map (hours) |
| `--no-beam` | — | Suppress beam overlay |

### Time & Planning Window
| Flag | Default | Description |
|------|---------|-------------|
| `--time ISO` | now | Start time UTC, e.g. `"2026-01-15 22:00:00"` |
| `--duration H` | 24.0 | Planning window (hours) |

### Output
| Flag | Default | Description |
|------|---------|-------------|
| `--save-map FILE` | — | Save sky map to PNG |
| `--save-elev FILE` | — | Save elevation plot to PNG |
| `--report FILE` | — | Save text report to file |
| `--no-map` | — | Skip sky map |
| `--no-elev` | — | Skip elevation plot |
| `--no-schedule` | — | Skip schedule table |
| `--no-transits` | — | Skip transit predictions |
| `--no-background` | — | Skip GSM sky background |
| `--min-response N` | 0.0 | Only show transits with beam response ≥ N (0–1) |

### Catalog
| Flag | Default | Description |
|------|---------|-------------|
| `--catalog FILE` | — | Load custom CSV catalog |
| `--min-flux JY` | 0.0 | Minimum source flux to display (Jy) |

---

## Custom CSV Catalog Format

```csv
name,ra_deg,dec_deg,flux_jy,ref_freq_mhz,spectral_index,source_type,notes
My Source,123.45,-30.0,5.2,1400,-0.7,continuum,
HII Region X,200.1,+45.3,12.0,1000,-0.1,hii region,
```

Source types: `continuum`, `radio galaxy`, `supernova remnant`, `hii region`, `pulsar`, `galactic center`, `solar`

---

## Built-in Source Catalog

| Source | Common Name | Type |
|--------|------------|------|
| 3C 461 | Cassiopeia A | Supernova remnant |
| 3C 405 | Cygnus A | Radio galaxy |
| 3C 144 | Tau A / Crab Nebula | Supernova remnant |
| 3C 274 | Virgo A / M87 | Radio galaxy |
| Sgr A* | Galactic centre | Galactic center |
| 3C 123 | — | Radio galaxy |
| 3C 218 | Hydra A | Radio galaxy |
| Orion A | — | HII region |
| W3 | — | HII region |
| Puppis A | — | Supernova remnant |
| Sun | — | Solar (position computed at runtime) |

---

## Project Structure

```
radio-planner/
├── core/
│   ├── observer.py      # ObserverSite: location, beam, sensitivity
│   ├── catalog.py       # RadioSource, RadioCatalog, built-in sources
│   └── ephemeris.py     # Rise/set/transit, drift-scan predictor
├── ui/
│   ├── skymap.py        # Sky map with GSM background, beam, transit highlights
│   └── planner.py       # Elevation plot, schedule table, sensitivity report
├── data/
│   ├── catalogs/        # Place custom CSV catalogs here
│   └── rfi/             # RFI frequency lists (future)
├── tests/
│   └── test_core.py     # pytest unit tests
├── main.py              # CLI entry point
├── requirements.txt     # Dependencies
├── pyproject.toml       # Package metadata
├── LICENSE              # MIT License
└── README.md
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `astropy` | Coordinate transforms, ephemeris, time handling |
| `numpy` | Numerical computation |
| `matplotlib` | Sky map and elevation plot rendering |
| `pygdsm` | Global Sky Model — radio background map |
| `healpy` | HEALPix projection for GSM map |
| `astroquery` | *(optional)* Live NVSS/SIMBAD catalog queries |

---

## Known Limitations

- The GSM background takes 10–20 seconds to generate on first run (cached for subsequent runs in the same session)
- Source magnitudes in the built-in catalog are flux densities at a reference frequency, not optical magnitudes
- The Sun's position is computed at the observation start time and does not update during an interactive session
- Flux density extrapolation uses a simple power-law spectral model; real source spectra may deviate, especially near spectral lines

---

## License

MIT — see [LICENSE](LICENSE) for details.