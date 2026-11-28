import numpy as np
import xarray as xr

from integration_stats import (
    align_land_cover,
    align_reference_zones,
    landsat_class_statistics,
    landsat_suhi,
    landsat_index_regressions,
)

def test_align_reference_zones_preserves_labels_and_target_coordinates():
    zones = xr.Dataset(
        {
            "urban": (("y", "x"), [[True, False], [False, False]]),
            "rural": (("y", "x"), [[False, True], [False, True]]),
        },
        coords={"x": [500_030, 500_090], "y": [5_549_970, 5_549_910]},
        attrs={"crs": "EPSG:32634"},
    )
    landsat = xr.Dataset(
        coords={
            "x": [500_015, 500_045, 500_075, 500_105, 500_135],
            "y": [5_549_985, 5_549_955, 5_549_925, 5_549_895],
        },
        attrs={
            "crs": "EPSG:32634",
            "transform": (30.0, 0.0, 500_000.0, 0.0, -30.0, 5_550_000.0),
        },
    )

    labels = align_reference_zones(zones, landsat)

    np.testing.assert_array_equal(
        labels.values,
        [
            [1, 1, 2, 2, 255],
            [1, 1, 2, 2, 255],
            [0, 0, 2, 2, 255],
            [0, 0, 2, 2, 255],
        ],
    )
    np.testing.assert_array_equal(labels.x.values, landsat.x.values)
    np.testing.assert_array_equal(labels.y.values, landsat.y.values)

def test_landsat_suhi_uses_valid_rural_mean_for_all_anomalies():
    landsat = xr.Dataset(
        {"lst": (("y", "x"), [[30.0, 34.0, 20.0], [22.0, np.nan, 27.0]])},
        coords={"x": [500_015, 500_045, 500_075], "y": [5_549_985, 5_549_955]},
        attrs={
            "crs": "EPSG:32634",
            "transform": (30.0, 0.0, 500_000.0, 0.0, -30.0, 5_550_000.0),
        },
    )
    labels = xr.DataArray(
        [[1, 1, 2], [2, 2, 0]],
        dims=("y", "x"),
        coords=landsat.coords,
        attrs=dict(landsat.attrs),
    )

    result = landsat_suhi(landsat, labels)

    assert result["urban_mean"].item() == 32.0
    assert result["rural_mean"].item() == 21.0
    assert result["urban_n_valid"].item() == 2
    assert result["rural_n_valid"].item() == 2
    assert result["suhi"].item() == 11.0
    np.testing.assert_allclose(
        result["anomaly"].values,
        [[9.0, 13.0, -1.0], [1.0, np.nan, 6.0]],
    )
    assert result.attrs == landsat.attrs

    no_rural = landsat_suhi(landsat, labels.where(labels != 2, 0))
    assert no_rural["rural_n_valid"].item() == 0
    assert np.isnan(no_rural["suhi"].item())
    assert np.isnan(no_rural["anomaly"].values).all()

def test_landsat_class_statistics_uses_sample_std_and_excludes_unclassified():
    landsat = xr.Dataset(
        {"lst": (("y", "x"), [[30.0, 34.0, 20.0], [np.nan, 99.0, np.nan]])},
        coords={"x": [500_015, 500_045, 500_075], "y": [5_549_985, 5_549_955]},
        attrs={
            "crs": "EPSG:32634",
            "transform": (30.0, 0.0, 500_000.0, 0.0, -30.0, 5_550_000.0),
        },
    )
    clc = xr.DataArray(
        np.array([[111, 111, 211], [111, 0, 311]], dtype=np.uint16),
        dims=("y", "x"),
        coords=landsat.coords,
        attrs={"crs": "EPSG:32634"},
    )

    aligned = align_land_cover(clc, landsat)
    np.testing.assert_array_equal(aligned.values, clc.values)

    stats = landsat_class_statistics(landsat, aligned)

    np.testing.assert_array_equal(stats.clc_class.values, [111, 211, 311])
    np.testing.assert_array_equal(stats["n_valid"].values, [2, 1, 0])
    np.testing.assert_allclose(stats["mean_lst"].values, [32.0, 20.0, np.nan])
    np.testing.assert_allclose(
        stats["sample_std_lst"].values, [np.sqrt(8.0), np.nan, np.nan]
    )

def test_landsat_index_regressions_use_separate_valid_pairs():
    landsat = xr.Dataset(
        {
            "lst": (("y", "x"), [[10.0, 20.0, 30.0, 99.0]]),
            "ndvi": (("y", "x"), [[0.0, 0.5, 1.0, 0.75]]),
            "ndbi": (("y", "x"), [[1.0, np.nan, 0.0, 0.25]]),
        },
        coords={"x": [500_015, 500_045, 500_075, 500_105], "y": [5_549_985]},
    )
    study_mask = xr.DataArray(
        [[True, True, True, False]], dims=("y", "x"), coords=landsat.coords
    )

    fits = landsat_index_regressions(landsat, study_mask)

    np.testing.assert_array_equal(fits["n_valid"].values, [3, 2])
    np.testing.assert_allclose(fits["slope"].values, [20.0, -20.0])
    np.testing.assert_allclose(fits["intercept"].values, [10.0, 30.0])
    np.testing.assert_allclose(fits["pearson_r"].values, [1.0, -1.0])

    landsat["ndbi"] = xr.zeros_like(landsat["ndbi"])
    constant = landsat_index_regressions(landsat, study_mask).sel(index="ndbi")
    assert constant["n_valid"].item() == 3
    assert np.isnan(constant["slope"].item())
    assert np.isnan(constant["pearson_r"].item())