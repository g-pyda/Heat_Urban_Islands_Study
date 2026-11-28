"""Discover and validate Landsat 8/9 Collection 2 Level-2 scenes."""

from pathlib import Path

import rasterio
import numpy as np
import xarray as xr
from rasterio.windows import Window


REQUIRED_BANDS = ("ST_B10", "SR_B4", "SR_B5", "SR_B6", "QA_PIXEL")
# Collection 2 Landsat 8/9 QA_PIXEL bits, in order from bit 0 to bit 5:
# fill, dilated cloud, cirrus, cloud, cloud shadow, and snow.
QA_PIXEL_REJECT_BITS = 0b00111111

ST_B10_SCALE_K = 0.00341802
ST_B10_OFFSET_K = 149.0
KELVIN_TO_CELSIUS_OFFSET = 273.15

SR_SCALE = 0.0000275
SR_OFFSET = -0.2

MIN_INDEX_DENOMINATOR = 1e-6

def discover_scenes(raw_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Return required GeoTIFF paths grouped by full Landsat scene ID."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Landsat directory does not exist: {raw_dir}")

    scenes: dict[str, dict[str, Path]] = {}
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".tif":
            continue

        for band in REQUIRED_BANDS:
            suffix = f"_{band}"
            if not path.stem.endswith(suffix):
                continue

            scene_id = path.stem.removesuffix(suffix)
            bands = scenes.setdefault(scene_id, {})
            if band in bands:
                raise ValueError(f"{scene_id}: found multiple files for {band}")
            bands[band] = path
            break

    if not scenes:
        raise FileNotFoundError(f"No Landsat science or QA bands found in {raw_dir}")

    for scene_id, bands in scenes.items():
        missing = [band for band in REQUIRED_BANDS if band not in bands]
        if missing:
            raise FileNotFoundError(
                f"{scene_id}: missing required bands {', '.join(missing)}"
            )

    return scenes


def validate_scene_grid(bands: dict[str, Path]) -> None:
    """Check that all required bands share the ST_B10 pixel grid."""
    with rasterio.open(bands["ST_B10"], mode="r") as thermal:
        if thermal.crs is None:
            raise ValueError(f"{bands['ST_B10'].name}: missing CRS")
        reference_grid = (thermal.crs, thermal.transform, thermal.shape)

    for band in REQUIRED_BANDS:
        if band == "ST_B10":
            continue

        with rasterio.open(bands[band], mode="r") as src:
            grid = (src.crs, src.transform, src.shape)

        if grid != reference_grid:
            raise ValueError(
                f"{bands[band].name}: CRS, transform, or shape differs from ST_B10"
            )

def qa_pixel_valid(qa_pixel: np.ndarray) -> np.ndarray:
    """Return True where none of the rejected Landsat QA flags is set."""
    return (qa_pixel & QA_PIXEL_REJECT_BITS) == 0

def read_masked_window(bands: dict[str, Path], window: Window) -> xr.Dataset:
    """Read QA-masked Landsat digital numbers on a native scene window."""
    validate_scene_grid(bands)

    with rasterio.open(bands["ST_B10"], mode="r") as thermal:
        if not (
            window.width > 0
            and window.height > 0
            and 0 <= window.col_off <= thermal.width - window.width
            and 0 <= window.row_off <= thermal.height - window.height
        ):
            raise ValueError("Window must be nonempty and inside the scene")
        if any(
            value != int(value)
            for value in (window.col_off, window.row_off, window.width, window.height)
        ):
            raise ValueError("Window offsets and dimensions must be whole pixels")

        transform = thermal.window_transform(window)
        crs = thermal.crs.to_string()

    if transform.b != 0 or transform.d != 0:
        raise ValueError("Expected a grid without rotation or shear")

    with rasterio.open(bands["QA_PIXEL"], mode="r") as src:
        qa_dn = src.read(1, window=window)
        qa_available = src.read_masks(1, window=window) != 0

    qa_valid = qa_available & qa_pixel_valid(qa_dn)
    ds = xr.Dataset(
        data_vars={"qa_valid": (("y", "x"), qa_valid)},
        coords={
            "x": transform.c + transform.a * (np.arange(qa_valid.shape[1]) + 0.5),
            "y": transform.f + transform.e * (np.arange(qa_valid.shape[0]) + 0.5),
        },
        attrs={"crs": crs, "transform": tuple(transform)[:6]},
    )

    for band in REQUIRED_BANDS:
        if band == "QA_PIXEL":
            continue

        with rasterio.open(bands[band], mode="r") as src:
            dn = src.read(1, window=window)
            band_valid = src.read_masks(1, window=window) != 0

        values = np.where(qa_valid & band_valid, dn.astype(np.float32), np.nan)
        ds[band] = (("y", "x"), values, {"units": "DN"})

    return ds

def st_b10_to_celsius(st_b10_dn: xr.DataArray) -> xr.DataArray:
    """Convert QA-masked Landsat Collection 2 ST_B10 DN to degrees Celsius."""
    temperature_k = st_b10_dn.astype(np.float64) * ST_B10_SCALE_K + ST_B10_OFFSET_K
    lst = (temperature_k - KELVIN_TO_CELSIUS_OFFSET).rename("lst")
    lst.attrs = {"units": "degC", "long_name": "land surface temperature"}
    return lst

def sr_dn_to_reflectance(sr_dn: xr.DataArray) -> xr.DataArray:
    """Scale QA-masked Landsat Collection 2 surface-reflectance DN."""
    reflectance = sr_dn.astype(np.float64) * SR_SCALE + SR_OFFSET
    reflectance.attrs = {"units": "1", "long_name": "surface reflectance"}
    return reflectance

def _normalized_difference(
    first: xr.DataArray, second: xr.DataArray
) -> xr.DataArray:
    """Return a normalized difference for eligible reflectance pairs."""
    denominator = first + second
    valid = (
        np.isfinite(first)
        & np.isfinite(second)
        & (first >= 0.0)
        & (second >= 0.0)
        & (denominator > MIN_INDEX_DENOMINATOR)
    )
    return (first - second).where(valid) / denominator.where(valid)


def landsat_indices(dn: xr.Dataset) -> xr.Dataset:
    """Compute NDVI and NDBI from QA-masked Landsat reflectance DN."""
    red = sr_dn_to_reflectance(dn["SR_B4"])
    nir = sr_dn_to_reflectance(dn["SR_B5"])
    swir1 = sr_dn_to_reflectance(dn["SR_B6"])

    ndvi = _normalized_difference(nir, red)
    ndvi.attrs = {
        "units": "1",
        "long_name": "normalized difference vegetation index",
    }
    ndbi = _normalized_difference(swir1, nir)
    ndbi.attrs = {
        "units": "1",
        "long_name": "normalized difference built-up index",
    }

    return xr.Dataset({"ndvi": ndvi, "ndbi": ndbi}, attrs=dict(dn.attrs))

def process_landsat_window(bands: dict[str, Path], window: Window) -> xr.Dataset:
    """Return calibrated LST, NDVI, and NDBI on a native Landsat window."""
    dn = read_masked_window(bands, window)
    processed = landsat_indices(dn)
    processed["lst"] = st_b10_to_celsius(dn["ST_B10"])
    return processed