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


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
# QC_Day / QC_Night bit layout (MOD11 C6.1 user guide, Table 13):
#
#   bits 1-0  Mandatory QA   00 LST produced, good quality
#                            01 LST produced, other quality (check other bits)
#                            10 LST not produced due to cloud effects
#                            11 LST not produced, other reasons
#   bits 3-2  Data quality   00 good, 01 other quality, 1x TBD
#   bits 5-4  Emissivity err 00 <=0.01, 01 <=0.02, 10 <=0.04, 11 >0.04
#   bits 7-6  LST error      00 <=1 K,  01 <=2 K,  10 <=3 K,  11 >3 K

QC_GOOD = 0b00
QC_OTHER_QUALITY = 0b01
QC_CLOUD = 0b10
QC_NOT_PRODUCED = 0b11

# Upper bound [K] of each LST error class (bits 7-6)
_LST_ERROR_CLASS_MAX_K = (1.0, 2.0, 3.0, np.inf)

LST_QC_PAIRS = {LST_DAY: QC_DAY, LST_NIGHT: QC_NIGHT}
VIEW_TIME_OF = {LST_DAY: VIEW_TIME_DAY, LST_NIGHT: VIEW_TIME_NIGHT}


def decode_qc(qc: xr.DataArray) -> xr.Dataset:
    """Split a QC_Day/QC_Night byte into its four 2-bit fields."""
    qc = qc.astype(np.uint8)
    return xr.Dataset({
        "mandatory_qa": qc & 0b11,
        "data_quality": (qc >> 2) & 0b11,
        "emissivity_error": (qc >> 4) & 0b11,
        "lst_error": (qc >> 6) & 0b11,
    })


def quality_mask(
    qc: xr.DataArray,
    max_lst_error: float | None = 2.0,
    allow_other_quality: bool = True,
) -> xr.DataArray:
    """Boolean mask of pixels to keep (``True`` = usable retrieval).

    Parameters
    ----------
    qc:
        Raw ``QC_Day`` or ``QC_Night`` layer.
    max_lst_error:
        Largest accepted LST error class in kelvin (1, 2 or 3);
        ``None`` disables the check.
    allow_other_quality:
        If ``False`` only "good quality" (mandatory QA = 00) pixels are kept
        (strict mode). If ``True`` "other quality" (01) pixels are also kept,
        provided they pass the LST error threshold.

    Cloud-affected (10) and not-produced (11) pixels are always rejected.
    """
    fields = decode_qc(qc)
    accepted = [QC_GOOD, QC_OTHER_QUALITY] if allow_other_quality else [QC_GOOD]
    keep = fields["mandatory_qa"].isin(accepted)

    if max_lst_error is not None:
        if max_lst_error not in _LST_ERROR_CLASS_MAX_K[:3]:
            raise ValueError("max_lst_error must be 1, 2, 3 or None")
        max_class = _LST_ERROR_CLASS_MAX_K.index(max_lst_error)
        keep &= fields["lst_error"] <= max_class

    return keep.rename("quality_mask")


def apply_qc(
    lst: xr.DataArray,
    qc: xr.DataArray,
    max_lst_error: float | None = 2.0,
    allow_other_quality: bool = True,
) -> xr.DataArray:
    """Set rejected, fill-valued and out-of-range LST pixels to NaN.

    Values stay as digital numbers (float32); conversion to Celsius is done
    separately.
    """
    lo, hi = LST_VALID_RANGE
    valid = (lst != LST_FILL) & (lst >= lo) & (lst <= hi)
    keep = valid & quality_mask(qc, max_lst_error, allow_other_quality)

    attrs = {k: v for k, v in lst.attrs.items() if k != "_FillValue"}
    out = lst.astype(np.float32).where(keep)
    out.attrs = attrs
    return out


def mask_granule(
    ds: xr.Dataset,
    max_lst_error: float | None = 2.0,
    allow_other_quality: bool = True,
) -> xr.Dataset:
    """Apply QC masking to the day and night layers of a loaded granule.

    LST pixels failing QC become NaN, and the matching view-time pixels are
    masked as well so that each valid temperature has a valid timestamp.
    Masking statistics are stored as ``valid_fraction_*`` attributes.
    """
    out = ds.copy()
    for lst_name, qc_name in LST_QC_PAIRS.items():
        if lst_name not in ds or qc_name not in ds:
            continue
        masked = apply_qc(ds[lst_name], ds[qc_name], max_lst_error, allow_other_quality)
        out[lst_name] = masked

        vt_name = VIEW_TIME_OF[lst_name]
        if vt_name in ds:
            vt_attrs = {k: v for k, v in ds[vt_name].attrs.items() if k != "_FillValue"}
            out[vt_name] = ds[vt_name].astype(np.float32).where(masked.notnull())
            out[vt_name].attrs = vt_attrs

        out.attrs[f"valid_fraction_{lst_name}"] = float(masked.notnull().mean())

    out.attrs.update(
        qc_max_lst_error=str(max_lst_error),
        qc_allow_other_quality=int(allow_other_quality),
    )
    return out


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def dn_to_celsius(dn: xr.DataArray) -> xr.DataArray:
    """Convert LST digital numbers to degrees Celsius.

    ``Ts [degC] = 0.02 * DN - 273.15``

    Fill values (``DN == 0``) and values outside the valid range are returned
    as NaN, so the conversion is safe on raw as well as QC-masked input
    (a fill of 0 would otherwise become -273.15 degC).
    """
    lo, hi = LST_VALID_RANGE
    valid = (dn >= lo) & (dn <= hi)
    celsius = (dn.astype(np.float32) * np.float32(LST_SCALE)
               - np.float32(KELVIN_OFFSET)).where(valid)
    celsius.attrs = {
        "long_name": dn.attrs.get("long_name", dn.name or "land surface temperature"),
        "standard_name": "surface_temperature",
        "units": "degC",
    }
    return celsius


def view_time_to_hours(vt: xr.DataArray) -> xr.DataArray:
    """Convert view-time digital numbers to local solar time in hours.

    ``t [h] = 0.1 * DN``; the fill value 255 and values outside 0-240 become
    NaN.
    """
    lo, hi = VIEW_TIME_VALID_RANGE
    valid = (vt >= lo) & (vt <= hi) & (vt != VIEW_TIME_FILL)
    hours = (vt.astype(np.float32) * np.float32(VIEW_TIME_SCALE)).where(valid)
    hours.attrs = {
        "long_name": vt.attrs.get("long_name", "local solar time of observation"),
        "units": "hours",
        "comment": "local solar time (0-24 h)",
    }
    return hours


def convert_granule(ds: xr.Dataset) -> xr.Dataset:
    """Convert LST layers to degC and view-time layers to hours.

    Works on the output of :func:`load_lst_layers` or :func:`mask_granule`;
    QC layers (if present) are left untouched.
    """
    out = ds.copy()
    for name in (LST_DAY, LST_NIGHT):
        if name in out:
            out[name] = dn_to_celsius(ds[name])
    for name in (VIEW_TIME_DAY, VIEW_TIME_NIGHT):
        if name in out:
            out[name] = view_time_to_hours(ds[name])
    return out


# ---------------------------------------------------------------------------
# Pipeline and export
# ---------------------------------------------------------------------------

OUTPUT_NAMES = {
    LST_DAY: "lst_day",
    LST_NIGHT: "lst_night",
    VIEW_TIME_DAY: "view_time_day",
    VIEW_TIME_NIGHT: "view_time_night",
}

# Per-granule attributes that would conflict when stacking/mosaicking
_GRANULE_ATTRS = ("tile", "source_file")


def process_granule(
    path: Path | str,
    product: ModisProduct,
    max_lst_error: float | None = 2.0,
    allow_other_quality: bool = True,
) -> xr.Dataset:
    """Load, QC-mask and convert one granule.

    Returns a Dataset with ``lst_day``/``lst_night`` [degC] and
    ``view_time_day``/``view_time_night`` [h local solar time] on the
    sinusoidal ``(y, x)`` grid; QC layers are dropped.
    """
    ds = load_lst_layers(path, product)
    ds = mask_granule(ds, max_lst_error, allow_other_quality)
    ds = convert_granule(ds)
    ds = ds.drop_vars([QC_DAY, QC_NIGHT]).rename(OUTPUT_NAMES)
    for key in list(ds.attrs):
        if key.startswith("valid_fraction_") or key in _GRANULE_ATTRS:
            del ds.attrs[key]
    return ds


def process_product(
    product: ModisProduct,
    paths: list[Path | str],
    max_lst_error: float | None = 2.0,
    allow_other_quality: bool = True,
) -> xr.Dataset:
    """Process all granules of one product into a ``(time, y, x)`` Dataset.

    Granules from the same day (neighbouring tiles) are mosaicked, then days
    are stacked along ``time``. Only granules of ``product`` are accepted,
    so Terra and Aqua always end up in separate datasets.
    """
    if not paths:
        raise ValueError(f"No granules given for {product}")

    by_day: dict[datetime, list[xr.Dataset]] = {}
    for path in sorted(paths):
        day = parse_granule_name(path)["date"]
        logger.info("%s: processing %s", product, Path(path).name)
        by_day.setdefault(day, []).append(
            process_granule(path, product, max_lst_error, allow_other_quality)
        )

    daily = [
        tiles[0] if len(tiles) == 1
        else xr.combine_by_coords(tiles, combine_attrs="drop_conflicts")
        for _, tiles in sorted(by_day.items())
    ]
    ds = xr.concat(daily, dim="time", combine_attrs="drop_conflicts")

    for part in ("day", "night"):
        ds[f"valid_fraction_{part}"] = ds[f"lst_{part}"].notnull().mean(("y", "x"))
        ds[f"valid_fraction_{part}"].attrs = {
            "long_name": f"fraction of {part} pixels passing QC",
            "units": "1",
        }

    ds.attrs.update(
        title=f"Cleaned daily {product.short_name} LST ({product.platform})",
        product=product.short_name,
        platform=product.platform,
        history=f"{datetime.now():%Y-%m-%d %H:%M} processed with modis_processing.py",
    )
    return ds


def export(ds: xr.Dataset, path: Path | str | None = None) -> Path:
    """Write a processed dataset to compressed NetCDF.

    Defaults to ``data/processed/<short_name>_lst.nc`` based on the
    ``product`` attribute. Reload with ``xr.open_dataset(path)``.
    """
    if path is None:
        path = get_product(ds.attrs["product"]).processed_path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in ds.data_vars
    }
    ds.to_netcdf(path, encoding=encoding)
    logger.info("Saved %s", path)
    return path


def run(
    product: ModisProduct,
    bbox: BoundingBox,
    start: date | str,
    end: date | str,
    max_lst_error: float | None = 2.0,
    allow_other_quality: bool = True,
) -> xr.Dataset:
    """Download, process and export one product end to end."""
    paths = download_granules(product, bbox, start, end)
    ds = process_product(product, paths, max_lst_error, allow_other_quality)
    export(ds)
    return ds


def _parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and clean daily MODIS LST (MOD11A1/MYD11A1)."
    )
    parser.add_argument("--satellite", choices=["terra", "aqua", "both"], default="both")
    parser.add_argument("--bbox", type=float, nargs=4, required=True,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--max-lst-error", type=float, choices=[1, 2, 3], default=2)
    parser.add_argument("--strict", action="store_true",
                        help="keep only mandatory QA = 00 (good quality) pixels")
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    products = [TERRA, AQUA] if args.satellite == "both" else [get_product(args.satellite)]
    for prod in products:
        run(prod, tuple(args.bbox), args.start, args.end,
            max_lst_error=args.max_lst_error, allow_other_quality=not args.strict)
