"""Surface Urban Heat Island (SUHI) analysis of daily MODIS LST.

Combines the cleaned LST from :mod:`modis_processing` (``lst_day`` /
``lst_night`` [degC] on the sinusoidal ``(time, y, x)`` grid) with the
land-cover masks aggregated to the same 1 km grid in :mod:`corine_utils`
(:func:`corine_utils.modis_urban_rural_masks`,
:func:`corine_utils.class_fractions`).

For every observation date the module computes

* the mean surface temperature of the urban (``T_u``) and rural reference
  (``T_r``) pixels, separately for day and night,
* the SUHI intensity ``dT_SUHI = T_u - T_r`` for day and night.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

PERIODS = ("day", "night")

# A zone mean is only reported if at least this share of the zone's pixels
# has a valid (cloud-free, QC-passed) LST value on that date; partially
# clouded zones would otherwise give means of an unrepresentative subset.
DEFAULT_MIN_VALID_FRACTION = 0.5

Zones = Mapping[object, xr.DataArray] | xr.Dataset


# ---------------------------------------------------------------------------
# Grid alignment
# ---------------------------------------------------------------------------


def align_to_grid(lst: xr.Dataset, grid: xr.DataArray | xr.Dataset) -> xr.Dataset:
    """Select the LST pixels lying on the (cropped) mask grid.

    The CORINE masks keep the MODIS pixel-centre coordinates, so pixels are
    matched by coordinate (with a tolerance of 1 % of the pixel size to
    absorb float round-off from NetCDF round trips).
    """
    x, y = grid["x"].values, grid["y"].values
    sizes = [abs(np.diff(c)).min() for c in (x, y) if len(c) > 1]
    tol = 0.01 * min(sizes) if sizes else 1e-6
    try:
        out = lst.sel(x=x, y=y, method="nearest", tolerance=tol)
    except KeyError as err:
        raise ValueError("LST and mask grids are not aligned") from err
    return out.assign_coords(x=x, y=y)


def _ensure_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.dims:
        ds = ds.expand_dims("time") if "time" in ds.coords else ds.expand_dims(time=[0])
    return ds


def _period_stack(lst: xr.Dataset) -> xr.DataArray:
    """``(period, time, y, x)`` array of day and night LST."""
    missing = [f"lst_{p}" for p in PERIODS if f"lst_{p}" not in lst]
    if missing:
        raise ValueError(f"LST dataset lacks variables: {missing}")
    stack = xr.concat([lst[f"lst_{p}"] for p in PERIODS], dim="period")
    return stack.assign_coords(period=np.array(PERIODS, dtype=object))


def _zone_stack(zones: Zones, like: xr.DataArray) -> xr.DataArray:
    """``(zone, y, x)`` boolean array of the zone masks on ``like``'s grid."""
    if isinstance(zones, xr.Dataset):
        zones = {k: zones[k] for k in zones.data_vars}
    if not zones:
        raise ValueError("No zones given")
    shape = (like.sizes["y"], like.sizes["x"])
    values = []
    for name, mask in zones.items():
        mask = np.asarray(mask.transpose("y", "x").values, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"Zone {name!r} has shape {mask.shape}, expected {shape}")
        values.append(mask)
    return xr.DataArray(np.stack(values), dims=("zone", "y", "x"),
                        coords={"zone": list(zones), "y": like["y"], "x": like["x"]})


# ---------------------------------------------------------------------------
# Zone statistics
# ---------------------------------------------------------------------------


def zone_statistics(
    lst: xr.Dataset,
    zones: Zones,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    paired: bool = False,
) -> xr.Dataset:
    """Day and night LST statistics of pixel sets (zones) for every date.

    Parameters
    ----------
    lst:
        Processed LST with ``lst_day`` / ``lst_night`` [degC] on ``(time, y,
        x)`` (a single date without ``time`` is accepted as well).
    zones:
        Boolean ``(y, x)`` masks on the MODIS grid, as a mapping
        ``name -> mask`` or a Dataset (one variable per zone). All masks
        share one grid; ``lst`` is cut to it with :func:`align_to_grid`.
    min_valid_fraction:
        Minimum share of a zone's pixels with valid LST for its mean to be
        reported (NaN otherwise).
    paired:
        Use only pixels valid both day *and* night on a date, so that day
        and night means describe the same surface.

    Returns
    -------
    Dataset on ``(time, zone, period)`` with ``mean`` and ``std`` [degC],
    ``n_valid`` (pixels with valid LST), ``valid_fraction`` and the zone
    size ``n_pixels`` (``zone``).
    """
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be within [0, 1]")
    first = (zones[next(iter(zones.data_vars))] if isinstance(zones, xr.Dataset)
             else next(iter(zones.values()), None))
    if first is None:
        raise ValueError("No zones given")

    temps = _period_stack(_ensure_time(align_to_grid(lst, first)))
    if paired:
        temps = temps.where(temps.notnull().all("period"))
    zone = _zone_stack(zones, temps)

    masked = temps.where(zone)
    n_pixels = zone.sum(("y", "x"))
    n_valid = masked.notnull().sum(("y", "x"))
    valid_fraction = (n_valid / n_pixels.where(n_pixels > 0)).fillna(0.0)
    ok = (n_valid > 0) & (valid_fraction >= min_valid_fraction)

    dims = ("time", "zone", "period")
    stats = xr.Dataset(
        {
            "mean": masked.mean(("y", "x")).where(ok).transpose(*dims),
            "std": masked.std(("y", "x")).where(ok).transpose(*dims),
            "n_valid": n_valid.transpose(*dims),
            "valid_fraction": valid_fraction.transpose(*dims),
            "n_pixels": n_pixels,
        },
        attrs={"min_valid_fraction": min_valid_fraction, "paired": int(paired),
               **{k: v for k, v in lst.attrs.items()
                  if k in ("product", "platform")}},
    )
    stats["mean"].attrs = {"units": "degC", "long_name": "zone mean LST"}
    stats["std"].attrs = {"units": "degC", "long_name": "zone LST standard deviation"}
    stats["valid_fraction"].attrs = {"units": "1",
                                     "long_name": "share of zone pixels with valid LST"}
    return stats


def urban_rural_lst(
    lst: xr.Dataset,
    masks: xr.Dataset,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    paired: bool = False,
) -> xr.Dataset:
    """Mean urban (``T_u``) and rural (``T_r``) LST for day and night.

    ``masks`` is the output of :func:`corine_utils.modis_urban_rural_masks`
    (boolean ``urban`` / ``rural`` on the MODIS grid). Returns
    :func:`zone_statistics` with ``zone = ["urban", "rural"]``.
    """
    return zone_statistics(lst, {"urban": masks["urban"], "rural": masks["rural"]},
                           min_valid_fraction=min_valid_fraction, paired=paired)


# ---------------------------------------------------------------------------
# SUHI intensity
# ---------------------------------------------------------------------------


def suhi_intensity(
    stats: xr.Dataset,
    urban: object = "urban",
    rural: object = "rural",
) -> xr.DataArray:
    """SUHI intensity ``dT_SUHI = T_u - T_r`` [degC] on ``(time, period)``.

    ``stats`` is the output of :func:`urban_rural_lst` (or
    :func:`zone_statistics` with the zones named ``urban`` / ``rural``).
    The intensity is NaN on dates where either zone mean is missing.
    """
    zones = list(stats["zone"].values)
    for name in (urban, rural):
        if name not in zones:
            raise ValueError(f"Zone {name!r} not in statistics (have {zones})")
    mean = stats["mean"]
    suhi = mean.sel(zone=urban, drop=True) - mean.sel(zone=rural, drop=True)
    suhi.name = "suhi"
    suhi.attrs = {"units": "degC",
                  "long_name": "surface urban heat island intensity (T_u - T_r)"}
    return suhi
