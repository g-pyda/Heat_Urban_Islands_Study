import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from landsat_processing import discover_scenes, read_masked_window


def test_read_masked_window_combines_qa_and_band_nodata(tmp_path):
    scene_id = "LC08_L2SP_188025_20250605_20250607_02_T1"
    transform = from_origin(500_000, 5_550_000, 30, 30)
    layers = {
        "ST_B10": [[45_000, 45_100], [0, 45_300]],
        "SR_B4": [[10_000, 10_100], [10_200, 0]],
        "SR_B5": [[12_000, 12_100], [12_200, 12_300]],
        "SR_B6": [[11_000, 11_100], [11_200, 11_300]],
        "QA_PIXEL": [[0, 1 << 3], [0, 1 << 7]],
    }

    for band, values in layers.items():
        path = tmp_path / f"{scene_id}_{band}.TIF"
        with rasterio.open(
            path,
            mode="w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="uint16",
            crs="EPSG:32634",
            transform=transform,
            nodata=None if band == "QA_PIXEL" else 0,
        ) as dst:
            dst.write(np.asarray(values, dtype=np.uint16), 1)

    bands = discover_scenes(tmp_path)[scene_id]
    ds = read_masked_window(bands, Window(0, 0, 2, 2))

    np.testing.assert_array_equal(
        ds["qa_valid"].values, [[True, False], [True, True]]
    )
    np.testing.assert_allclose(
        ds["ST_B10"].values, [[45_000, np.nan], [np.nan, 45_300]]
    )
    np.testing.assert_allclose(
        ds["SR_B4"].values, [[10_000, np.nan], [10_200, np.nan]]
    )
    np.testing.assert_allclose(
        ds["SR_B5"].values, [[12_000, np.nan], [12_200, 12_300]]
    )
    np.testing.assert_array_equal(ds.x.values, [500_015, 500_045])
    np.testing.assert_array_equal(ds.y.values, [5_549_985, 5_549_955])