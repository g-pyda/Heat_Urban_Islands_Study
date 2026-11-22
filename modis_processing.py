"""Download and preparation of daily MODIS Land Surface Temperature products.

Handles the daily 1 km gridded LST products:

* MOD11A1 - Terra (local overpass ~10:30 / ~22:30)
* MYD11A1 - Aqua  (local overpass ~13:30 / ~01:30)

The goal is cleaned day and night LST maps in degrees Celsius, exported as
``xarray`` objects for the analysis of day/night temperature differences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Layer (subdataset) names in MOD11A1 / MYD11A1 HDF4 files
# ---------------------------------------------------------------------------

LST_DAY = "LST_Day_1km"
LST_NIGHT = "LST_Night_1km"
QC_DAY = "QC_Day"
QC_NIGHT = "QC_Night"
VIEW_TIME_DAY = "Day_view_time"
VIEW_TIME_NIGHT = "Night_view_time"

DAY_LAYERS = (LST_DAY, QC_DAY, VIEW_TIME_DAY)
NIGHT_LAYERS = (LST_NIGHT, QC_NIGHT, VIEW_TIME_NIGHT)
REQUIRED_LAYERS = DAY_LAYERS + NIGHT_LAYERS

# ---------------------------------------------------------------------------
# Scaling and fill values (MOD11 Collection 6.1 user guide)
# ---------------------------------------------------------------------------

LST_SCALE = 0.02          # DN -> Kelvin
KELVIN_OFFSET = 273.15    # Kelvin -> Celsius
LST_FILL = 0              # no-data value for LST layers
LST_VALID_RANGE = (7500, 65535)

VIEW_TIME_SCALE = 0.1     # DN -> local solar time [h]
VIEW_TIME_FILL = 255
VIEW_TIME_VALID_RANGE = (0, 240)

# ---------------------------------------------------------------------------
# Satellite / product definitions
# ---------------------------------------------------------------------------
# Terra and Aqua observe at different local solar times, so their products
# are kept as separate objects and are never mixed in a single time series.


@dataclass(frozen=True)
class ModisProduct:
    """Static description of one daily MODIS LST product.

    Attributes
    ----------
    short_name:
        NASA CMR short name of the product (e.g. ``"MOD11A1"``).
    platform:
        Satellite carrying the MODIS instrument (``"Terra"`` or ``"Aqua"``).
    day_overpass_lst:
        Nominal daytime overpass, local solar time in hours.
    night_overpass_lst:
        Nominal nighttime overpass, local solar time in hours.
    version:
        Collection version used for searching/downloading.
    """

    short_name: str
    platform: str
    day_overpass_lst: float
    night_overpass_lst: float
    version: str = "061"

    @property
    def raw_dir(self) -> Path:
        """Directory holding raw granules of this product."""
        return RAW_DIR / self.short_name

    @property
    def processed_path(self) -> Path:
        """Default output file for the cleaned LST dataset."""
        return PROCESSED_DIR / f"{self.short_name}_lst.nc"

    def __str__(self) -> str:
        return f"{self.short_name} ({self.platform})"


TERRA = ModisProduct(
    short_name="MOD11A1",
    platform="Terra",
    day_overpass_lst=10.5,
    night_overpass_lst=22.5,
)

AQUA = ModisProduct(
    short_name="MYD11A1",
    platform="Aqua",
    day_overpass_lst=13.5,
    night_overpass_lst=1.5,
)

PRODUCTS = {p.platform.lower(): p for p in (TERRA, AQUA)}


def get_product(name: str) -> ModisProduct:
    """Return a product by platform (``"terra"``/``"aqua"``) or short name."""
    key = name.lower()
    if key in PRODUCTS:
        return PRODUCTS[key]
    for product in PRODUCTS.values():
        if product.short_name.lower() == key:
            return product
    raise ValueError(f"Unknown MODIS product: {name!r}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

BoundingBox = tuple[float, float, float, float]  # (west, south, east, north)


def download_granules(
    product: ModisProduct,
    bbox: BoundingBox,
    start: date | str,
    end: date | str,
    out_dir: Path | str | None = None,
    login_strategy: str = "netrc",
) -> list[Path]:
    """Search and download MODIS LST granules from NASA Earthdata.

    Parameters
    ----------
    product:
        ``TERRA`` or ``AQUA`` (each is downloaded into its own directory).
    bbox:
        Area of interest as ``(west, south, east, north)`` in degrees.
    start, end:
        Inclusive date range, as ``datetime.date`` or ``"YYYY-MM-DD"``.
    out_dir:
        Target directory; defaults to ``product.raw_dir``.
    login_strategy:
        ``earthaccess`` login strategy: ``"netrc"`` (``~/.netrc`` or
        ``~/_netrc``), ``"environment"`` (``EARTHDATA_USERNAME`` /
        ``EARTHDATA_PASSWORD``) or ``"interactive"``.

    Returns
    -------
    list[Path]
        Sorted paths of the local HDF files (already present files are reused).
    """
    import earthaccess  # imported lazily: only needed for downloading

    out_dir = Path(out_dir) if out_dir is not None else product.raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    auth = earthaccess.login(strategy=login_strategy)
    if not auth or not auth.authenticated:
        raise RuntimeError("Earthdata login failed - check your credentials.")

    results = earthaccess.search_data(
        short_name=product.short_name,
        version=product.version,
        bounding_box=tuple(bbox),
        temporal=(str(start), str(end)),
    )
    logger.info("%s: found %d granules for %s..%s", product, len(results), start, end)
    if not results:
        return []

    files = earthaccess.download(results, local_path=str(out_dir))
    return sorted(Path(f) for f in files)
