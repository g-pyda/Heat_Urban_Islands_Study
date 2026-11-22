"""Unit tests for corine_utils using a synthetic CLC raster (no downloads)."""

from pathlib import Path

import numpy as np
import xarray as xr
import pytest

import corine_utils as c

rasterio = pytest.importorskip("rasterio")
from pyproj import Transformer  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

RES = 100.0
N = 400  # 40 km x 40 km synthetic raster centred on Krakow

_to_laea = Transformer.from_crs("EPSG:4326", c.CORINE_CRS, always_xy=True)
CX, CY = _to_laea.transform(*c.KRAKOW_CENTRE)
ORIGIN_X = np.floor(CX / RES) * RES - N / 2 * RES
ORIGIN_Y = np.floor(CY / RES) * RES + N / 2 * RES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def grid_code(clc_code: int) -> int:
    return c.CLC_CODES.index(clc_code) + 1


def synthetic_grid() -> np.ndarray:
    """Grid codes: urban core, arable ring, forest in the east, no-data row."""
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(xx - N / 2, yy - N / 2) * RES
    g = np.full((N, N), grid_code(211), dtype=np.uint8)
    g[xx > 0.8 * N] = grid_code(311)
    g[r < 6000] = grid_code(112)
    g[r < 3000] = grid_code(111)
    g[(r < 6000) & (xx > N / 2 + 40)] = grid_code(121)
    g[0, :] = 48  # CLC "NODATA"
    return g


def write_clc(path: Path, data: np.ndarray, nodata=None) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
        count=1, dtype=data.dtype, crs=c.CORINE_CRS, nodata=nodata,
        transform=from_origin(ORIGIN_X, ORIGIN_Y, RES, RES),
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def clc_path(tmp_path):
    return write_clc(tmp_path / "clc.tif", synthetic_grid(), nodata=128)


@pytest.fixture
def clc(clc_path):
    return c.load_corine(clc_path, bbox=None)


# ---------------------------------------------------------------------------
# Code conversion
# ---------------------------------------------------------------------------


def test_grid_code_table_covers_44_classes():
    assert len(c.GRID_CODE_TO_CLC) == 44
    assert c.GRID_CODE_TO_CLC[1] == 111
    assert c.GRID_CODE_TO_CLC[12] == 211
    assert c.GRID_CODE_TO_CLC[23] == 311
    assert c.GRID_CODE_TO_CLC[44] == 523


def test_grid_to_clc_maps_nodata_codes():
    out = c.grid_to_clc(np.array([1, 2, 3, 12, 23, 48, 128, 0]))
    np.testing.assert_array_equal(out, [111, 112, 121, 211, 311, 0, 0, 0])


def test_rasters_with_clc_codes_are_kept(tmp_path):
    data = np.array([[111, 311], [211, 999]], dtype=np.uint16)
    path = write_clc(tmp_path / "codes.tif", data, nodata=999)
    out = c.load_corine(path, bbox=None)
    np.testing.assert_array_equal(out.values, [[111, 311], [211, 0]])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_full_raster(clc):
    assert clc.shape == (N, N)
    assert clc.dtype == np.uint16
    assert clc.attrs["crs"] == c.CORINE_CRS
    assert clc.attrs["res"] == (RES, RES)
    # pixel-centre coordinates, north-up
    assert clc.x[0] == pytest.approx(ORIGIN_X + RES / 2)
    assert clc.y[0] == pytest.approx(ORIGIN_Y - RES / 2)
    assert np.all(np.diff(clc.y) < 0)
    assert set(np.unique(clc.values)) == {0, 111, 112, 121, 211, 311}
    assert np.all(clc.values[0] == c.CLC_NODATA)


def test_load_window_from_bbox(clc_path):
    bbox = (19.90, 50.04, 19.98, 50.08)
    sub = c.load_corine(clc_path, bbox=bbox)
    full = c.load_corine(clc_path, bbox=None)
    assert sub.shape[0] < N and sub.shape[1] < N
    # the window is an exact sub-grid of the full raster
    expected = full.sel(x=sub.x, y=sub.y)
    np.testing.assert_array_equal(sub.values, expected.values)
    # and it encloses the requested box
    x0, y0, x1, y1 = c._bbox_to_crs(bbox, c.CORINE_CRS)
    assert sub.x.min() - RES / 2 <= x0 and sub.x.max() + RES / 2 >= x1
    assert sub.y.min() - RES / 2 <= y0 and sub.y.max() + RES / 2 >= y1


def test_load_bbox_outside_raster_raises(clc_path):
    with pytest.raises(Exception):
        c.load_corine(clc_path, bbox=(0.0, 40.0, 1.0, 41.0))


# ---------------------------------------------------------------------------
# Class lookup at coordinates
# ---------------------------------------------------------------------------


def test_clc_at_city_centre_is_continuous_urban(clc):
    assert c.clc_at(clc, *c.KRAKOW_CENTRE) == 111


def test_clc_at_vectorised_and_outside(clc):
    lon = np.array([c.KRAKOW_CENTRE[0], 20.18, 19.70, 0.0])
    lat = np.array([c.KRAKOW_CENTRE[1], 50.06, 50.06, 45.0])
    np.testing.assert_array_equal(c.clc_at(clc, lon, lat), [111, 311, 211, 0])


def test_clc_at_projected_coordinates(clc):
    x, y = float(clc.x[5]), float(clc.y[200])
    assert c.clc_at(clc, x, y, crs=c.CORINE_CRS) == clc.values[200, 5]


def test_clc_name():
    assert c.clc_name(311) == "Broad-leaved forest"
    assert c.clc_name(0) == "No data"


# ---------------------------------------------------------------------------
# Land-cover class masks
# ---------------------------------------------------------------------------


def test_class_mask_single_and_multiple_codes(clc):
    forest = c.class_mask(clc, 311)
    urban = c.class_mask(clc, c.URBAN_CODES)
    assert forest.dtype == bool and forest.shape == clc.shape
    np.testing.assert_array_equal(forest.values, clc.values == 311)
    np.testing.assert_array_equal(
        urban.values, np.isin(clc.values, [111, 112, 121])
    )
    assert urban.attrs["clc_codes"] == [111, 112, 121]
    assert "nodata" not in urban.attrs


def test_class_mask_excludes_nodata(clc):
    assert not c.class_mask(clc, c.CLC_CODES).values[0].any()


def test_class_mask_rejects_unknown_codes(clc):
    with pytest.raises(ValueError, match="999"):
        c.class_mask(clc, [111, 999])


def test_land_cover_masks_default_groups(clc):
    ds = c.land_cover_masks(clc)
    assert set(ds.data_vars) == {"urban", "forest", "arable", "valid"}
    assert ds.attrs["crs"] == c.CORINE_CRS
    # groups are mutually exclusive and all lie within valid pixels
    stacked = np.stack([ds[k].values for k in ("urban", "forest", "arable")])
    assert stacked.sum(axis=0).max() == 1
    assert not (stacked.any(axis=0) & ~ds["valid"].values).any()
    # the synthetic raster contains only these classes besides no-data
    np.testing.assert_array_equal(stacked.any(axis=0), ds["valid"].values)
    assert ds["forest"].attrs["clc_names"] == ["Broad-leaved forest"]


def test_land_cover_masks_custom_groups(clc):
    ds = c.land_cover_masks(clc, {"fabric": (111, 112)})
    assert set(ds.data_vars) == {"fabric", "valid"}
    np.testing.assert_array_equal(
        ds["fabric"].values, np.isin(clc.values, [111, 112])
    )


def test_masks_agree_with_point_lookup(clc):
    ds = c.land_cover_masks(clc)
    x, y = ds.x.values[[200, 395, 10]], ds.y.values[[200, 200, 200]]
    codes = c.clc_at(clc, x, y, crs=c.CORINE_CRS)
    np.testing.assert_array_equal(codes, [111, 311, 211])
    assert ds["urban"].values[200, 200] and ds["forest"].values[200, 395]


def test_class_shares(clc):
    shares = c.class_shares(clc)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert set(shares) == {111, 112, 121, 211, 311}
    assert shares[211] > shares[311] > 0


# ---------------------------------------------------------------------------
# Urban core and rural reference masks
# ---------------------------------------------------------------------------

VILLAGE = (slice(195, 206), slice(45, 56))     # 121 px, ~15 km west
SPECK = (slice(290, 293), slice(199, 202))     # 9 px, ~9 km south


@pytest.fixture
def clc_settl(tmp_path):
    """Synthetic raster with a village and a tiny built-up speck."""
    g = synthetic_grid()
    g[VILLAGE] = grid_code(112)
    g[SPECK] = grid_code(121)
    return c.load_corine(write_clc(tmp_path / "s.tif", g, nodata=128), bbox=None)


def dist_from_centre(clc):
    return c.distance_from_point(clc).values


def pixel_lon(clc):
    """Longitude of every pixel centre (meridians are tilted in LAEA)."""
    xx, yy = np.meshgrid(clc.x.values, clc.y.values)
    lon, _ = Transformer.from_crs(c.CORINE_CRS, "EPSG:4326",
                                  always_xy=True).transform(xx, yy)
    return lon


def test_distance_to_single_pixel(clc):
    mask = clc.copy(data=np.zeros(clc.shape, bool))
    mask.values[100, 100] = True
    d = c.distance_to(mask)
    assert d.attrs["units"] == "m"
    assert d.values[100, 100] == 0
    assert d.values[100, 103] == pytest.approx(300)
    assert d.values[104, 103] == pytest.approx(500)


def test_distance_to_empty_mask_is_inf(clc):
    mask = clc.copy(data=np.zeros(clc.shape, bool))
    assert np.isinf(c.distance_to(mask).values).all()


def test_distance_from_point_centre(clc):
    d = dist_from_centre(clc)
    assert d.min() < RES
    assert d[0, 0] > 25_000


def test_polygon_mask_projected_square(clc):
    x0, y0 = float(clc.x[10]) - RES / 2, float(clc.y[29]) - RES / 2
    square = {"type": "Polygon", "coordinates": [[
        (x0, y0), (x0 + 2000, y0), (x0 + 2000, y0 + 2000),
        (x0, y0 + 2000), (x0, y0)]]}
    m = c.polygon_mask(clc, square, crs=c.CORINE_CRS)
    assert m.dtype == bool and m.values.sum() == 400
    assert m.values[10:30, 10:30].all()


def test_urban_core_keeps_city_drops_village_and_speck(clc_settl):
    core = c.urban_core_mask(clc_settl).values
    urban_all = c.class_mask(clc_settl, c.URBAN_CODES).values
    city = urban_all & (dist_from_centre(clc_settl) < 6_100)
    np.testing.assert_array_equal(core, city)
    assert not core[VILLAGE].any() and not core[SPECK].any()


def test_urban_core_min_cluster_and_radius(clc_settl):
    core = c.urban_core_mask(clc_settl, radius_m=16_000, min_cluster_px=1)
    assert core.values[VILLAGE].all() and core.values[SPECK].all()


def test_urban_core_clipped_to_boundary(clc):
    west_half = {"type": "Polygon", "coordinates": [[
        (19.0, 49.0), (c.KRAKOW_CENTRE[0], 49.0),
        (c.KRAKOW_CENTRE[0], 51.0), (19.0, 51.0), (19.0, 49.0)]]}
    core = c.urban_core_mask(clc, boundary=west_half).values
    full = c.urban_core_mask(clc).values
    assert 0 < core.sum() < full.sum()
    assert pixel_lon(clc)[core].max() <= c.KRAKOW_CENTRE[0]
    np.testing.assert_array_equal(core, full & (pixel_lon(clc) <= c.KRAKOW_CENTRE[0]))


def test_rural_reference_default_criteria(clc_settl):
    urban = c.urban_core_mask(clc_settl)
    rural = c.rural_reference_mask(clc_settl, urban=urban)
    r = rural.values
    assert r.any()
    assert set(np.unique(clc_settl.values[r])) == {211, 311}
    d_core = c.distance_to(urban).values[r]
    assert d_core.min() >= 5_000 and d_core.max() <= 20_000
    # villages are buffered out, the city never overlaps
    d_any = c.distance_to(c.class_mask(clc_settl, c.URBAN_CODES)).values[r]
    assert d_any.min() >= 500
    assert not (r & urban.values).any()
    assert rural.attrs["min_distance_m"] == 5_000


def test_rural_reference_settlement_buffer_disabled(clc_settl):
    with_buf = c.rural_reference_mask(clc_settl).values
    no_buf = c.rural_reference_mask(clc_settl, settlement_buffer_m=0).values
    assert no_buf.sum() > with_buf.sum()
    assert not no_buf[VILLAGE].any()  # built-up pixels are never rural


def test_rural_reference_class_selection(clc_settl):
    forest = c.rural_reference_mask(clc_settl, classes=c.FOREST_CODES).values
    assert forest.any()
    assert np.all(clc_settl.values[forest] == 311)


def test_rural_reference_invalid_ring(clc):
    with pytest.raises(ValueError):
        c.rural_reference_mask(clc, min_distance_m=10_000, max_distance_m=5_000)


def make_dem(clc, step=1):
    """Plane rising eastwards by 10 m/km, 200 m at the city centre."""
    x = clc.x.values[::step]
    y = clc.y.values[::step]
    z = 200 + 0.01 * (np.tile(x, (len(y), 1)) - CX)
    return xr.DataArray(z, dims=("y", "x"), coords={"x": x, "y": y})


@pytest.mark.parametrize("step", [1, 2])
def test_rural_reference_elevation_filter(clc, step):
    dem = make_dem(clc, step)
    rural = c.rural_reference_mask(clc, dem=dem, max_elev_diff_m=50)
    r = rural.values
    assert r.any()
    assert rural.attrs["reference_elevation_m"] == pytest.approx(200, abs=10)
    xx = np.tile(clc.x.values, (clc.shape[0], 1))
    # only pixels within ~5 km east/west of the centre survive
    assert np.abs(xx[r] - CX).max() <= 5_000 + 10 * RES
    assert r.sum() < c.rural_reference_mask(clc).values.sum()


def test_rural_reference_manual_region(clc):
    west = {"type": "Polygon", "coordinates": [[
        (19.0, 49.0), (19.85, 49.0), (19.85, 51.0), (19.0, 51.0), (19.0, 49.0)]]}
    r = c.rural_reference_mask(clc, region=west).values
    assert r.any()
    assert pixel_lon(clc)[r].max() <= 19.85
    full = c.rural_reference_mask(clc).values
    assert full[pixel_lon(clc) > 19.85].any()  # the region really restricts


def test_urban_rural_masks_dataset(clc_settl):
    ds = c.urban_rural_masks(clc_settl, rural_kwargs={"max_distance_m": 15_000})
    assert set(ds.data_vars) == {"urban", "rural", "distance_to_urban"}
    assert ds["urban"].dtype == bool and ds["rural"].dtype == bool
    assert not (ds["urban"] & ds["rural"]).any()
    assert ds["urban"].values[N // 2, N // 2]
    assert float(ds["distance_to_urban"].where(ds["rural"]).max()) <= 15_000
    assert ds.attrs["crs"] == c.CORINE_CRS
