"""Unit tests for suhi_analysis using synthetic LST and masks (no downloads)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import suhi_analysis as s

PX = 926.625433  # MODIS 1 km pixel size [m]
NY, NX = 6, 8
DATES = pd.to_datetime(["2025-07-01", "2025-07-02", "2025-07-03"])

# Synthetic temperatures [degC]: urban hotter than rural, more so at night
T = {
    "urban": {"day": 35.0, "night": 22.0},
    "rural": {"day": 31.0, "night": 15.0},
    "other": {"day": 33.0, "night": 18.0},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def coords(pad=0):
    """Pixel centres of the mask grid, optionally with extra LST pixels."""
    x = (np.arange(-pad, NX + pad) + 0.5) * PX + 2.0e6
    y = 5.6e6 - (np.arange(-pad, NY + pad) + 0.5) * PX  # north-up
    return x, y


def grid_da(values, name):
    x, y = coords()
    return xr.DataArray(values, dims=("y", "x"), coords={"x": x, "y": y}, name=name)


def make_masks():
    """Urban block in the west (cols 0-2), rural block in the east (cols 5-7)."""
    urban = np.zeros((NY, NX), bool)
    rural = np.zeros((NY, NX), bool)
    urban[:, :3] = True
    rural[:, 5:] = True
    return xr.Dataset({"urban": grid_da(urban, "urban"),
                       "rural": grid_da(rural, "rural")})


def zone_field(period, pad=0):
    """LST field with the zone temperatures of ``T`` (other pixels = 'other')."""
    field = np.full((NY + 2 * pad, NX + 2 * pad), T["other"][period])
    field[:, : pad + 3] = T["urban"][period]
    field[:, pad + 5: pad + NX] = T["rural"][period]
    return field


def make_lst(pad=2, n_days=3):
    """Processed-LST-like dataset on a grid larger than the masks."""
    x, y = coords(pad)
    data = {
        f"lst_{p}": (("time", "y", "x"),
                     np.stack([zone_field(p, pad)] * n_days))
        for p in s.PERIODS
    }
    return xr.Dataset(data, coords={"time": DATES[:n_days], "x": x, "y": y},
                      attrs={"platform": "Terra", "product": "MOD11A1"})


# ---------------------------------------------------------------------------
# Urban and rural mean LST
# ---------------------------------------------------------------------------


def test_align_to_grid_cuts_lst_to_mask_grid():
    masks = make_masks()
    out = s.align_to_grid(make_lst(pad=2), masks)
    assert out.sizes["x"] == NX and out.sizes["y"] == NY
    np.testing.assert_array_equal(out.x, masks.x)
    np.testing.assert_array_equal(out.y, masks.y)


def test_align_to_grid_tolerates_round_off_and_rejects_shifted_grid():
    masks = make_masks()
    lst = make_lst(pad=1)
    jitter = lst.assign_coords(x=lst.x + 1e-3)
    assert s.align_to_grid(jitter, masks).sizes["x"] == NX
    shifted = lst.assign_coords(x=lst.x + PX / 2)
    with pytest.raises(ValueError):
        s.align_to_grid(shifted, masks)


def test_urban_rural_lst_means():
    stats = s.urban_rural_lst(make_lst(), make_masks())
    assert stats["mean"].dims == ("time", "zone", "period")
    assert list(stats.zone.values) == ["urban", "rural"]
    assert list(stats.period.values) == ["day", "night"]
    for zone in ("urban", "rural"):
        for period in s.PERIODS:
            np.testing.assert_allclose(
                stats["mean"].sel(zone=zone, period=period), T[zone][period])
    assert stats["n_pixels"].sel(zone="urban") == 3 * NY
    assert (stats["valid_fraction"] == 1.0).all()
    assert stats.attrs["platform"] == "Terra"


def test_zone_mean_ignores_cloudy_pixels_and_respects_min_valid_fraction():
    lst = make_lst(pad=0)
    # day 0: one urban pixel at 50 degC, the rest of that column clouded
    lst["lst_day"][0, :, 0] = np.nan
    lst["lst_day"][0, 0, 0] = 50.0
    # day 1: urban zone fully clouded at night
    lst["lst_night"][1, :, :3] = np.nan
    stats = s.urban_rural_lst(lst, make_masks())

    n_urban = 3 * NY
    urban_day0 = stats.sel(zone="urban", period="day").isel(time=0)
    assert int(urban_day0["n_valid"]) == n_urban - NY + 1
    expected = (50.0 + (n_urban - NY) * T["urban"]["day"]) / (n_urban - NY + 1)
    assert float(urban_day0["mean"]) == pytest.approx(expected)
    assert np.isnan(stats["mean"].sel(zone="urban", period="night").isel(time=1))

    strict = s.urban_rural_lst(lst, make_masks(), min_valid_fraction=0.9)
    assert np.isnan(strict["mean"].sel(zone="urban", period="day").isel(time=0))


def test_paired_uses_pixels_valid_day_and_night():
    lst = make_lst(pad=0)
    lst["lst_night"][0, :, 0] = np.nan          # clouds over one urban column
    lst["lst_day"][0, :, 0] = 60.0              # ...which is hot by day
    unpaired = s.urban_rural_lst(lst, make_masks())
    paired = s.urban_rural_lst(lst, make_masks(), paired=True)
    sel = dict(zone="urban", period="day", time=DATES[0])
    assert float(unpaired["mean"].sel(sel)) > T["urban"]["day"]
    assert float(paired["mean"].sel(sel)) == pytest.approx(T["urban"]["day"])
    assert int(paired["n_valid"].sel(sel)) == 2 * NY


def test_zone_statistics_single_date_and_custom_zones():
    lst = make_lst().isel(time=0)
    masks = make_masks()
    stats = s.zone_statistics(lst, {"west": masks.urban, "all": masks.urban | True})
    assert stats.sizes["time"] == 1
    assert stats["n_pixels"].sel(zone="all") == NY * NX


def test_zone_statistics_input_validation():
    masks = make_masks()
    with pytest.raises(ValueError):
        s.zone_statistics(make_lst(), {})
    with pytest.raises(ValueError):
        s.zone_statistics(make_lst().drop_vars("lst_night"), masks)
    with pytest.raises(ValueError):
        s.zone_statistics(make_lst(), masks, min_valid_fraction=1.5)
