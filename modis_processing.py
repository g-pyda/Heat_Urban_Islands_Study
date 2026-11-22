"""Download and preparation of daily MODIS Land Surface Temperature products.

Handles the daily 1 km gridded LST products:

* MOD11A1 - Terra (local overpass ~10:30 / ~22:30)
* MYD11A1 - Aqua  (local overpass ~13:30 / ~01:30)

The goal is cleaned day and night LST maps in degrees Celsius, exported as
``xarray`` objects for the analysis of day/night temperature differences.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
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


# ---------------------------------------------------------------------------
# Loading raw layers
# ---------------------------------------------------------------------------
# MOD11A1/MYD11A1 granules are HDF4-EOS files. GDAL builds shipped with pip
# wheels usually lack the HDF4 driver, so the files are read with ``pyhdf``
# and the sinusoidal grid coordinates are rebuilt from the EOS metadata.

# MODIS sinusoidal projection (sphere of radius 6371007.181 m)
MODIS_SINUSOIDAL_PROJ4 = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
)

# SDS attributes worth keeping on the raw arrays
_KEPT_SDS_ATTRS = (
    "long_name", "units", "valid_range", "_FillValue", "scale_factor", "add_offset",
)

_GRANULE_RE = re.compile(
    r"^(?P<product>M[OY]D11A1)\.A(?P<year>\d{4})(?P<doy>\d{3})"
    r"\.(?P<tile>h\d{2}v\d{2})\.(?P<version>\d{3})"
)


def parse_granule_name(path: Path | str) -> dict:
    """Parse product, acquisition date, tile and version from a file name.

    Example: ``MOD11A1.A2025182.h19v03.061.2025184093512.hdf``.
    """
    name = Path(path).name
    match = _GRANULE_RE.match(name)
    if match is None:
        raise ValueError(f"Not a MOD11A1/MYD11A1 granule name: {name}")
    acq = datetime.strptime(match["year"] + match["doy"], "%Y%j")
    return {
        "product": match["product"],
        "date": acq,
        "tile": match["tile"],
        "version": match["version"],
    }


def _parse_grid_metadata(struct_metadata: str) -> dict:
    """Extract grid size and corner coordinates from ``StructMetadata.0``."""

    def _find(pattern: str) -> str:
        match = re.search(pattern, struct_metadata)
        if match is None:
            raise ValueError(f"StructMetadata entry not found: {pattern}")
        return match.group(1)

    ul = _find(r"UpperLeftPointMtrs=\(([^)]*)\)").split(",")
    lr = _find(r"LowerRightMtrs=\(([^)]*)\)").split(",")
    return {
        "nx": int(_find(r"XDim=(\d+)")),
        "ny": int(_find(r"YDim=(\d+)")),
        "ulx": float(ul[0]), "uly": float(ul[1]),
        "lrx": float(lr[0]), "lry": float(lr[1]),
    }


def _sinusoidal_coords(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre x/y coordinates (metres) of the sinusoidal grid."""
    dx = (grid["lrx"] - grid["ulx"]) / grid["nx"]
    dy = (grid["lry"] - grid["uly"]) / grid["ny"]  # negative (north-up)
    x = grid["ulx"] + dx * (np.arange(grid["nx"]) + 0.5)
    y = grid["uly"] + dy * (np.arange(grid["ny"]) + 0.5)
    return x, y


def open_granule(
    path: Path | str,
    layers: tuple[str, ...] = REQUIRED_LAYERS,
) -> xr.Dataset:
    """Read raw (unscaled) layers of one MOD11A1/MYD11A1 granule.

    Returns a Dataset with dims ``(y, x)`` in MODIS sinusoidal metres, a
    scalar ``time`` coordinate (acquisition day) and the original digital
    numbers; scaling and masking happen in later steps.
    """
    from pyhdf.SD import SD, SDC  # HDF4 reader

    path = Path(path)
    info = parse_granule_name(path)

    hdf = SD(str(path), SDC.READ)
    try:
        grid = _parse_grid_metadata(hdf.attributes()["StructMetadata.0"])
        available = hdf.datasets()
        data_vars = {}
        for name in layers:
            if name not in available:
                raise KeyError(f"Layer {name!r} missing in {path.name}")
            sds = hdf.select(name)
            try:
                values = sds.get()
                attrs = {k: v for k, v in sds.attributes().items()
                         if k in _KEPT_SDS_ATTRS}
            finally:
                sds.endaccess()
            data_vars[name] = (("y", "x"), values, attrs)
    finally:
        hdf.end()

    x, y = _sinusoidal_coords(grid)
    return xr.Dataset(
        data_vars,
        coords={"x": x, "y": y, "time": np.datetime64(info["date"], "ns")},
        attrs={
            "product": info["product"],
            "tile": info["tile"],
            "collection": info["version"],
            "source_file": path.name,
            "crs": MODIS_SINUSOIDAL_PROJ4,
        },
    )


def load_lst_layers(path: Path | str, product: ModisProduct) -> xr.Dataset:
    """Load LST, QC and view-time layers (day and night) for one granule.

    Checks that the file belongs to ``product`` so Terra and Aqua granules
    cannot be mixed up, and tags the result with the platform and its
    nominal overpass times.
    """
    info = parse_granule_name(path)
    if info["product"] != product.short_name:
        raise ValueError(
            f"{Path(path).name} is {info['product']}, expected {product.short_name}"
        )
    ds = open_granule(path, REQUIRED_LAYERS)
    ds.attrs.update(
        platform=product.platform,
        day_overpass_lst=product.day_overpass_lst,
        night_overpass_lst=product.night_overpass_lst,
    )
    return ds
