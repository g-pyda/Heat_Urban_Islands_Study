"""Align CORINE reference zones to the native Landsat analysis grid."""

import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import Affine
from math import ceil, floor
from pathlib import Path

import rasterio
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from corine_utils import CLC_NODATA, clc_bounds, clc_name
from landsat_processing import process_landsat_window

ZONE_CODES = {"urban": 1, "rural": 2}
ZONE_OUTSIDE = 255


def align_reference_zones(
    zones: xr.Dataset, landsat: xr.Dataset
) -> xr.DataArray:
    """Project CORINE urban and rural zones onto a processed Landsat window."""
    urban = zones["urban"].transpose("y", "x").values
    rural = zones["rural"].transpose("y", "x").values
    if urban.dtype != np.bool_ or rural.dtype != np.bool_:
        raise ValueError("Expected Boolean CORINE zone masks")
    if np.any(urban & rural):
        raise ValueError("CORINE urban and rural masks overlap")

    source = np.zeros(urban.shape, dtype=np.uint8)
    source[urban] = ZONE_CODES["urban"]
    source[rural] = ZONE_CODES["rural"]

    x, y = zones.x.values, zones.y.values
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    src_transform = Affine(
        dx, 0, float(x[0]) - dx / 2,
        0, dy, float(y[0]) - dy / 2,
    )
    dst_transform = Affine(*landsat.attrs["transform"])
    labels = np.full(
        (landsat.sizes["y"], landsat.sizes["x"]),
        ZONE_OUTSIDE,
        dtype=np.uint8,
    )

    reproject(
        source=source,
        destination=labels,
        src_transform=src_transform,
        src_crs=zones.attrs["crs"],
        src_nodata=None,
        dst_transform=dst_transform,
        dst_crs=landsat.attrs["crs"],
        dst_nodata=ZONE_OUTSIDE,
        resampling=Resampling.nearest,
    )

    return xr.DataArray(
        labels,
        dims=("y", "x"),
        coords={"y": landsat.y, "x": landsat.x},
        name="reference_zone",
        attrs={
            "crs": landsat.attrs["crs"],
            "transform": landsat.attrs["transform"],
            "flag_values": [0, 1, 2, ZONE_OUTSIDE],
            "flag_meanings": "neither urban rural outside_source",
        },
    )

def landsat_suhi(landsat: xr.Dataset, labels: xr.DataArray) -> xr.Dataset:
    """Compute urban and rural LST means, SUHI, and rural-relative anomalies."""
    if any(labels.attrs[key] != landsat.attrs[key] for key in ("crs", "transform")):
        raise ValueError("Reference zones and Landsat must share a CRS and transform")

    lst = landsat["lst"].transpose("y", "x")
    labels = labels.transpose("y", "x")
    lst, labels = xr.align(lst, labels, join="exact")
    lst = lst.where(np.isfinite(lst))

    urban = lst.where(labels == ZONE_CODES["urban"])
    rural = lst.where(labels == ZONE_CODES["rural"])
    urban_mean = urban.mean(skipna=True)
    rural_mean = rural.mean(skipna=True)

    result = xr.Dataset(
        {
            "urban_mean": urban_mean,
            "rural_mean": rural_mean,
            "urban_n_valid": urban.count(),
            "rural_n_valid": rural.count(),
            "suhi": urban_mean - rural_mean,
            "anomaly": lst - rural_mean,
        },
        attrs=dict(landsat.attrs),
    )
    result["urban_mean"].attrs = {"units": "degC", "long_name": "urban mean LST"}
    result["rural_mean"].attrs = {"units": "degC", "long_name": "rural mean LST"}
    result["suhi"].attrs = {"units": "degC", "long_name": "urban minus rural mean LST"}
    result["anomaly"].attrs = {"units": "degC", "long_name": "LST minus rural mean LST"}
    return result

def align_land_cover(clc: xr.DataArray, landsat: xr.Dataset) -> xr.DataArray:
    """Transfer CORINE class codes to the native Landsat window."""
    clc = clc.transpose("y", "x")
    x, y = clc.x.values, clc.y.values
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    src_transform = Affine(
        dx, 0, float(x[0]) - dx / 2,
        0, dy, float(y[0]) - dy / 2,
    )
    classes = np.full(
        (landsat.sizes["y"], landsat.sizes["x"]), CLC_NODATA, dtype=np.uint16
    )
    reproject(
        source=clc.values,
        destination=classes,
        src_transform=src_transform,
        src_crs=clc.attrs["crs"],
        src_nodata=CLC_NODATA,
        dst_transform=Affine(*landsat.attrs["transform"]),
        dst_crs=landsat.attrs["crs"],
        dst_nodata=CLC_NODATA,
        resampling=Resampling.nearest,
    )
    return xr.DataArray(
        classes,
        dims=("y", "x"),
        coords={"y": landsat.y, "x": landsat.x},
        name="clc_class",
        attrs={
            "crs": landsat.attrs["crs"],
            "transform": landsat.attrs["transform"],
            "nodata": CLC_NODATA,
        },
    )


def landsat_class_statistics(
    landsat: xr.Dataset, classes: xr.DataArray
) -> xr.Dataset:
    """Return valid counts, means, and sample SDs by aligned CORINE class."""
    if any(classes.attrs[key] != landsat.attrs[key] for key in ("crs", "transform")):
        raise ValueError("Land cover and Landsat must share a CRS and transform")

    lst, classes = xr.align(
        landsat["lst"].transpose("y", "x"),
        classes.transpose("y", "x"),
        join="exact",
    )
    temperatures = lst.values
    codes = classes.values
    class_codes = np.unique(codes[codes != CLC_NODATA])
    valid = np.isfinite(temperatures)

    counts = np.zeros(class_codes.size, dtype=np.int64)
    means = np.full(class_codes.size, np.nan)
    sample_stds = np.full(class_codes.size, np.nan)

    for index, code in enumerate(class_codes):
        values = temperatures[valid & (codes == code)]
        counts[index] = values.size
        if values.size:
            means[index] = values.mean()
        if values.size > 1:
            sample_stds[index] = values.std(ddof=1)

    return xr.Dataset(
        {
            "n_valid": ("clc_class", counts),
            "mean_lst": ("clc_class", means, {"units": "degC"}),
            "sample_std_lst": ("clc_class", sample_stds, {"units": "degC"}),
        },
        coords={
            "clc_class": class_codes,
            "clc_name": ("clc_class", [clc_name(int(code)) for code in class_codes]),
        },
    )

def landsat_index_regressions(
    landsat: xr.Dataset, study_mask: xr.DataArray
) -> xr.Dataset:
    """Fit per-index LST regressions on valid pairs inside the study mask."""
    lst, study_mask = xr.align(
        landsat["lst"].transpose("y", "x"),
        study_mask.transpose("y", "x"),
        join="exact",
    )
    if study_mask.dtype != np.bool_:
        raise ValueError("Expected a Boolean study mask")

    temperatures = lst.values
    valid_lst = study_mask.values & np.isfinite(temperatures)
    index_names = ("ndvi", "ndbi")
    counts = np.zeros(len(index_names), dtype=np.int64)
    slopes = np.full(len(index_names), np.nan)
    intercepts = np.full(len(index_names), np.nan)
    correlations = np.full(len(index_names), np.nan)

    for position, name in enumerate(index_names):
        index_values = landsat[name].transpose("y", "x").values
        valid = valid_lst & np.isfinite(index_values)
        x = index_values[valid]
        y = temperatures[valid]
        counts[position] = x.size
        if x.size < 2 or x.min() == x.max():
            continue

        x_mean = x.mean()
        y_mean = y.mean()
        x_centered = x - x_mean
        y_centered = y - y_mean
        sxx = np.dot(x_centered, x_centered)
        sxy = np.dot(x_centered, y_centered)

        slopes[position] = sxy / sxx
        intercepts[position] = y_mean - slopes[position] * x_mean

        if y.min() != y.max():
            syy = np.dot(y_centered, y_centered)
            correlations[position] = np.clip(
                sxy / np.sqrt(sxx * syy), -1.0, 1.0
            )

    return xr.Dataset(
        {
            "n_valid": ("index", counts),
            "slope": ("index", slopes, {"units": "degC per unit index"}),
            "intercept": ("index", intercepts, {"units": "degC"}),
            "pearson_r": ("index", correlations, {"units": "1"}),
        },
        coords={"index": list(index_names)},
    )

def analyze_landsat_scene(
    bands: dict[str, Path], clc: xr.DataArray, zones: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Return Landsat maps and SUHI, CORINE class statistics, and index fits."""
    with rasterio.open(bands["ST_B10"], mode="r") as src:
        bounds = transform_bounds(clc.attrs["crs"], src.crs, *clc_bounds(clc))
        extent = from_bounds(*bounds, transform=src.transform)
        left = floor(extent.col_off)
        top = floor(extent.row_off)
        right = ceil(extent.col_off + extent.width)
        bottom = ceil(extent.row_off + extent.height)
        window = Window(left, top, right - left, bottom - top)

    landsat = process_landsat_window(bands, window)
    labels = align_reference_zones(zones, landsat)
    classes = align_land_cover(clc, landsat)

    suhi = landsat_suhi(landsat, labels)
    class_stats = landsat_class_statistics(landsat, classes)
    fits = landsat_index_regressions(landsat, classes != CLC_NODATA)

    maps = xr.merge(
        [landsat, suhi, labels, classes],
        join="exact",
        combine_attrs="override",
    )
    return maps, class_stats, fits

def export_landsat_analysis(
    path: str | Path,
    scene_id: str,
    maps: xr.Dataset,
    class_stats: xr.Dataset,
    fits: xr.Dataset,
) -> Path:
    """Save one scene's georeferenced maps and statistics as NetCDF."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")

    # Distinguish the 2-D land-cover raster from the class table coordinate.
    maps = maps.rename({"clc_class": "land_cover_code"})
    class_stats = class_stats.rename({"n_valid": "class_n_valid"})
    fits = fits.rename({"n_valid": "index_n_valid"})

    output = xr.merge(
        [maps, class_stats, fits], join="exact", combine_attrs="override"
    )
    output.attrs.update(
        scene_id=scene_id,
        land_cover_source="CORINE Land Cover 2018",
        regression_footprint="Classified CORINE cells with valid LST and index",
    )
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name, variable in output.data_vars.items()
        if variable.dims == ("y", "x")
    }
    output.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path