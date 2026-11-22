"""Unit tests for modis_processing using synthetic data (no downloads)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import modis_processing as m

TILE_M = 1111950.519667  # width of a MODIS sinusoidal tile in metres
N = 4                    # pixels per tile side in synthetic granules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def da(values, dtype):
    return xr.DataArray(np.array(values, dtype=dtype), dims="x")


def write_granule(
    directory: Path,
    product: str = "MOD11A1",
    doy: int = 182,
    h: int = 19,
    v: int = 3,
    lst_day: int = 15000,
    lst_night: int = 14000,
    qc_day: int = 0,
    qc_night: int = 0,
    vt_day: int = 105,
    vt_night: int = 225,
) -> Path:
    """Write a small MOD11A1-like HDF4 granule with constant layers."""
    SD = pytest.importorskip("pyhdf.SD")

    path = directory / f"{product}.A2025{doy:03d}.h{h:02d}v{v:02d}.061.2025999000000.hdf"
    ulx, uly = (h - 18) * TILE_M, (9 - v) * TILE_M
    struct = (
        f"XDim={N}\nYDim={N}\n"
        f"UpperLeftPointMtrs=({ulx},{uly})\nLowerRightMtrs=({ulx + TILE_M},{uly - TILE_M})\n"
    )
    hdf = SD.SD(str(path), SD.SDC.WRITE | SD.SDC.CREATE)
    hdf.attr("StructMetadata.0").set(SD.SDC.CHAR, struct)
    layers = {
        m.LST_DAY: (SD.SDC.UINT16, np.uint16, lst_day),
        m.LST_NIGHT: (SD.SDC.UINT16, np.uint16, lst_night),
        m.QC_DAY: (SD.SDC.UINT8, np.uint8, qc_day),
        m.QC_NIGHT: (SD.SDC.UINT8, np.uint8, qc_night),
        m.VIEW_TIME_DAY: (SD.SDC.UINT8, np.uint8, vt_day),
        m.VIEW_TIME_NIGHT: (SD.SDC.UINT8, np.uint8, vt_night),
    }
    for name, (hdf_type, np_type, value) in layers.items():
        sds = hdf.create(name, hdf_type, (N, N))
        sds[:] = np.full((N, N), value, np_type)
        sds.attr("long_name").set(SD.SDC.CHAR, name)
        sds.attr("unrelated_attr").set(SD.SDC.CHAR, "x")
        sds.endaccess()
    hdf.end()
    return path


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


def test_products_are_separate():
    assert m.TERRA.short_name == "MOD11A1" and m.AQUA.short_name == "MYD11A1"
    assert m.TERRA.raw_dir != m.AQUA.raw_dir
    assert m.TERRA.processed_path != m.AQUA.processed_path


@pytest.mark.parametrize("name, expected", [
    ("terra", "Terra"), ("AQUA", "Aqua"), ("mod11a1", "Terra"), ("MYD11A1", "Aqua"),
])
def test_get_product(name, expected):
    assert m.get_product(name).platform == expected


def test_get_product_unknown():
    with pytest.raises(ValueError):
        m.get_product("sentinel")


# ---------------------------------------------------------------------------
# File names and grid metadata
# ---------------------------------------------------------------------------


def test_parse_granule_name():
    info = m.parse_granule_name("MYD11A1.A2024366.h19v03.061.2025001000000.hdf")
    assert info["product"] == "MYD11A1"
    assert info["date"].strftime("%Y-%m-%d") == "2024-12-31"  # leap year, DOY 366
    assert info["tile"] == "h19v03" and info["version"] == "061"


def test_parse_granule_name_rejects_other_products():
    with pytest.raises(ValueError):
        m.parse_granule_name("MOD13A1.A2025001.h19v03.061.2025001000000.hdf")


def test_sinusoidal_coords_are_pixel_centres():
    grid = {"nx": 4, "ny": 4, "ulx": 0.0, "uly": 400.0, "lrx": 400.0, "lry": 0.0}
    x, y = m._sinusoidal_coords(grid)
    np.testing.assert_allclose(x, [50, 150, 250, 350])
    np.testing.assert_allclose(y, [350, 250, 150, 50])


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

#            good,<=1K   other,<=2K  other,<=3K  cloud       not prod.
QC_SAMPLE = [0b00000000, 0b01000001, 0b10000001, 0b00000010, 0b00000011]


def test_decode_qc():
    fields = m.decode_qc(da([0b11100110], np.uint8)).squeeze()
    assert int(fields.mandatory_qa) == 0b10
    assert int(fields.data_quality) == 0b01
    assert int(fields.emissivity_error) == 0b10
    assert int(fields.lst_error) == 0b11


@pytest.mark.parametrize("kwargs, expected", [
    ({}, [True, True, False, False, False]),
    ({"max_lst_error": 1}, [True, False, False, False, False]),
    ({"max_lst_error": 3}, [True, True, True, False, False]),
    ({"max_lst_error": None}, [True, True, True, False, False]),
    ({"allow_other_quality": False}, [True, False, False, False, False]),
])
def test_quality_mask(kwargs, expected):
    assert m.quality_mask(da(QC_SAMPLE, np.uint8), **kwargs).values.tolist() == expected


def test_quality_mask_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        m.quality_mask(da(QC_SAMPLE, np.uint8), max_lst_error=1.5)


def test_apply_qc_masks_fill_range_and_quality():
    lst = da([15000, 0, 7000, 15000], np.uint16)
    lst.attrs = {"_FillValue": 0, "long_name": "LST"}
    qc = da([0, 0, 0, 0b10], np.uint8)
    out = m.apply_qc(lst, qc)
    assert out.dtype == np.float32
    assert np.isnan(out.values).tolist() == [False, True, True, True]
    assert "_FillValue" not in out.attrs and out.attrs["long_name"] == "LST"


def test_mask_granule_masks_view_time_with_lst():
    lst = da([15000, 15000], np.uint16)
    qc = da([0, 0b10], np.uint8)
    vt = da([105, 105], np.uint8)
    ds = xr.Dataset({m.LST_DAY: lst, m.QC_DAY: qc, m.VIEW_TIME_DAY: vt})
    out = m.mask_granule(ds)
    assert np.isnan(out[m.VIEW_TIME_DAY].values).tolist() == [False, True]
    assert out.attrs[f"valid_fraction_{m.LST_DAY}"] == 0.5
    assert ds[m.LST_DAY].dtype == np.uint16  # input not modified


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def test_dn_to_celsius():
    out = m.dn_to_celsius(da([15000, 7500, 0, 7499], np.uint16))
    np.testing.assert_allclose(out.values[:2], [26.85, -123.15], atol=1e-3)
    assert np.isnan(out.values[2:]).all()  # fill must not become -273.15
    assert out.attrs["units"] == "degC"


def test_dn_to_celsius_on_masked_input():
    masked = da([15000, np.nan], np.float32)
    out = m.dn_to_celsius(masked)
    assert abs(float(out[0]) - 26.85) < 1e-3 and np.isnan(out[1])


def test_view_time_to_hours():
    out = m.view_time_to_hours(da([0, 105, 240, 241, 255], np.uint8))
    np.testing.assert_allclose(out.values[:3], [0.0, 10.5, 24.0], atol=1e-5)
    assert np.isnan(out.values[3:]).all()
    assert out.attrs["units"] == "hours"


# ---------------------------------------------------------------------------
# HDF loading and full pipeline (require pyhdf)
# ---------------------------------------------------------------------------


def test_load_lst_layers(tmp_path):
    path = write_granule(tmp_path)
    ds = m.load_lst_layers(path, m.TERRA)
    assert set(m.REQUIRED_LAYERS) <= set(ds.data_vars)
    assert ds[m.LST_DAY].dtype == np.uint16 and int(ds[m.LST_DAY][0, 0]) == 15000
    assert "unrelated_attr" not in ds[m.LST_DAY].attrs
    assert ds.sizes == {"y": N, "x": N}
    assert str(ds.time.values)[:10] == "2025-07-01"
    assert ds.attrs["platform"] == "Terra" and ds.attrs["tile"] == "h19v03"


def test_load_lst_layers_rejects_wrong_satellite(tmp_path):
    path = write_granule(tmp_path, product="MYD11A1")
    with pytest.raises(ValueError, match="expected MOD11A1"):
        m.load_lst_layers(path, m.TERRA)


def test_process_granule(tmp_path):
    ds = m.process_granule(write_granule(tmp_path, qc_night=0b10), m.TERRA)
    assert set(ds.data_vars) == {"lst_day", "lst_night", "view_time_day", "view_time_night"}
    assert abs(float(ds.lst_day[0, 0]) - 26.85) < 1e-3
    assert float(ds.view_time_day[0, 0]) == pytest.approx(10.5)
    assert ds.lst_night.isnull().all() and ds.view_time_night.isnull().all()


def test_process_product_mosaics_tiles_and_stacks_days(tmp_path):
    paths = [
        write_granule(tmp_path, doy=182, h=19),
        write_granule(tmp_path, doy=182, h=20, qc_night=0b10),
        write_granule(tmp_path, doy=183, h=19),
        write_granule(tmp_path, doy=183, h=20),
    ]
    ds = m.process_product(m.TERRA, paths)
    assert dict(ds.sizes) == {"time": 2, "y": N, "x": 2 * N}
    assert bool((ds.x.diff("x") > 0).all())
    assert ds.valid_fraction_night.values.tolist() == [0.5, 1.0]
    assert ds.attrs["platform"] == "Terra" and "tile" not in ds.attrs


def test_process_product_rejects_mixed_satellites(tmp_path):
    paths = [write_granule(tmp_path), write_granule(tmp_path, product="MYD11A1")]
    with pytest.raises(ValueError):
        m.process_product(m.TERRA, paths)


def test_export_roundtrip(tmp_path):
    pytest.importorskip("netCDF4")
    ds = m.process_product(m.AQUA, [write_granule(tmp_path, product="MYD11A1")])
    path = m.export(ds, tmp_path / "out" / "aqua.nc")
    with xr.open_dataset(path) as back:
        xr.testing.assert_allclose(back.lst_day, ds.lst_day)
        assert back.lst_day.dtype == np.float32
        assert back.attrs["platform"] == "Aqua"
