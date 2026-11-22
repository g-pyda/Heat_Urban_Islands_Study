"""Unit tests for corine_utils using a synthetic CLC raster (no downloads)."""

from pathlib import Path

import numpy as np
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
