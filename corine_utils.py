"""Preparation of the CORINE Land Cover 2018 (CLC2018) raster.

CLC2018 is distributed by Copernicus as a 100 m GeoTIFF in the ETRS89-LAEA
projection (EPSG:3035). The module

* loads a window of the raster around the study area (Krakow),
* converts the stored grid codes into the 3-digit CLC class codes,
* assigns CLC classes to arbitrary geographic coordinates.

The result is an ``xarray.DataArray`` of CLC codes, used later to build
urban / rural masks and land-cover fractions on the 1 km MODIS grid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and study area
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
CORINE_DIR = DATA_DIR / "raw" / "corine"
CORINE_PATH = CORINE_DIR / "U2018_CLC2018_V2020_20u1.tif"

BoundingBox = tuple[float, float, float, float]  # (west, south, east, north)

# Krakow with a rural ring of ~20-25 km around it (degrees, WGS84)
KRAKOW_BBOX: BoundingBox = (19.55, 49.85, 20.35, 50.25)
# Main Market Square, used as the city centre
KRAKOW_CENTRE = (19.9372, 50.0617)  # (lon, lat)

CORINE_CRS = "EPSG:3035"
CLC_NODATA = 0  # value used for no-data after code conversion

# ---------------------------------------------------------------------------
# CLC nomenclature (level 3)
# ---------------------------------------------------------------------------

CLC_CLASSES: dict[int, str] = {
    111: "Continuous urban fabric",
    112: "Discontinuous urban fabric",
    121: "Industrial or commercial units",
    122: "Road and rail networks and associated land",
    123: "Port areas",
    124: "Airports",
    131: "Mineral extraction sites",
    132: "Dump sites",
    133: "Construction sites",
    141: "Green urban areas",
    142: "Sport and leisure facilities",
    211: "Non-irrigated arable land",
    212: "Permanently irrigated land",
    213: "Rice fields",
    221: "Vineyards",
    222: "Fruit trees and berry plantations",
    223: "Olive groves",
    231: "Pastures",
    241: "Annual crops associated with permanent crops",
    242: "Complex cultivation patterns",
    243: "Land principally occupied by agriculture with natural vegetation",
    244: "Agro-forestry areas",
    311: "Broad-leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",
    321: "Natural grasslands",
    322: "Moors and heathland",
    323: "Sclerophyllous vegetation",
    324: "Transitional woodland-shrub",
    331: "Beaches, dunes, sands",
    332: "Bare rocks",
    333: "Sparsely vegetated areas",
    334: "Burnt areas",
    335: "Glaciers and perpetual snow",
    411: "Inland marshes",
    412: "Peat bogs",
    421: "Salt marshes",
    422: "Salines",
    423: "Intertidal flats",
    511: "Water courses",
    512: "Water bodies",
    521: "Coastal lagoons",
    522: "Estuaries",
    523: "Sea and ocean",
}

CLC_CODES = tuple(CLC_CLASSES)  # ordered: grid code i <-> CLC_CODES[i - 1]

# The official GeoTIFF stores GRID_CODE 1..44 (plus 48 = "NODATA" and
# 128 = outside coverage) instead of the 3-digit codes.
GRID_CODE_TO_CLC: dict[int, int] = {i + 1: c for i, c in enumerate(CLC_CODES)}


def grid_to_clc(values: np.ndarray) -> np.ndarray:
    """Translate CLC raster grid codes (1..44) into 3-digit CLC codes.

    Values that are not valid grid codes (48, 128, fill values) become
    ``CLC_NODATA``.
    """
    lut = np.full(256, CLC_NODATA, dtype=np.uint16)
    for grid, clc in GRID_CODE_TO_CLC.items():
        lut[grid] = clc
    values = np.asarray(values)
    out = np.full(values.shape, CLC_NODATA, dtype=np.uint16)
    in_range = (values >= 0) & (values < 256)
    out[in_range] = lut[values[in_range].astype(np.intp)]
    return out


def _to_clc_codes(values: np.ndarray, nodata) -> np.ndarray:
    """Return 3-digit CLC codes whether the raster holds grid or CLC codes."""
    values = np.asarray(values)
    valid = values != nodata if nodata is not None else np.ones(values.shape, bool)
    if valid.any() and values[valid].max() > 100:
        # already CLC codes (e.g. a re-coded product); keep known codes only
        out = np.where(np.isin(values, CLC_CODES), values, CLC_NODATA)
        return out.astype(np.uint16)
    return grid_to_clc(np.where(valid, values, 0))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _bbox_to_crs(bbox: BoundingBox, crs, src_crs="EPSG:4326") -> BoundingBox:
    """Transform a bounding box, densifying edges to keep it enclosing."""
    from rasterio.warp import transform_bounds

    return transform_bounds(src_crs, crs, *bbox, densify_pts=21)


def load_corine(
    path: Path | str = CORINE_PATH,
    bbox: BoundingBox | None = KRAKOW_BBOX,
    bbox_crs: str = "EPSG:4326",
) -> xr.DataArray:
    """Load (a window of) the CLC raster as a DataArray of CLC class codes.

    Parameters
    ----------
    path:
        CLC GeoTIFF (Europe-wide file or a pre-clipped subset).
    bbox:
        Area to read as ``(west, south, east, north)`` in ``bbox_crs``;
        ``None`` reads the whole file (avoid for the Europe-wide raster).
    bbox_crs:
        CRS of ``bbox``, geographic WGS84 by default.

    Returns
    -------
    DataArray with dims ``(y, x)`` holding pixel-centre coordinates in the
    raster CRS, values as 3-digit CLC codes (``uint16``, ``CLC_NODATA`` = 0)
    and ``crs`` / ``transform`` / ``res`` in ``attrs``.
    """
    import rasterio
    from rasterio.windows import Window, from_bounds

    path = Path(path)
    with rasterio.open(path) as src:
        if bbox is not None:
            bounds = _bbox_to_crs(bbox, src.crs, bbox_crs)
            w = from_bounds(*bounds, transform=src.transform)
            # round outwards so the window fully encloses the box
            col0, row0 = np.floor(w.col_off), np.floor(w.row_off)
            col1 = np.ceil(w.col_off + w.width)
            row1 = np.ceil(w.row_off + w.height)
            window = Window(int(col0), int(row0), int(col1 - col0), int(row1 - row0))
            full = Window(0, 0, src.width, src.height)
            window = window.intersection(full)  # raises if outside raster
        else:
            window = Window(0, 0, src.width, src.height)

        raw = src.read(1, window=window)
        transform = src.window_transform(window)
        crs = src.crs
        nodata = src.nodata

    codes = _to_clc_codes(raw, nodata)
    ny, nx = codes.shape
    x = transform.c + transform.a * (np.arange(nx) + 0.5)
    y = transform.f + transform.e * (np.arange(ny) + 0.5)

    logger.info("Loaded CLC window %dx%d from %s", nx, ny, path.name)
    return xr.DataArray(
        codes,
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name="clc_code",
        attrs={
            "crs": crs.to_string(),
            "transform": tuple(transform)[:6],
            "res": (abs(transform.a), abs(transform.e)),
            "nodata": CLC_NODATA,
            "source_file": path.name,
        },
    )


# ---------------------------------------------------------------------------
# Class lookup at coordinates
# ---------------------------------------------------------------------------


def _transform_from_coords(da: xr.DataArray):
    """Affine transform of a regular grid given by pixel-centre coords."""
    from rasterio.transform import Affine

    x, y = da["x"].values, da["y"].values
    dx = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else da.attrs["res"][0]
    dy = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else -da.attrs["res"][1]
    return Affine(dx, 0.0, x[0] - dx / 2, 0.0, dy, y[0] - dy / 2)


def clc_at(
    clc: xr.DataArray,
    lon,
    lat,
    crs: str = "EPSG:4326",
) -> np.ndarray:
    """Return the CLC class code at the given coordinates.

    ``lon``/``lat`` may be scalars or arrays (in ``crs``, WGS84 by default).
    Points outside the loaded window get ``CLC_NODATA``.
    """
    from pyproj import Transformer

    lon_a, lat_a = np.broadcast_arrays(np.asarray(lon, float), np.asarray(lat, float))
    tr = Transformer.from_crs(crs, clc.attrs["crs"], always_xy=True)
    xs, ys = tr.transform(lon_a, lat_a)

    inv = ~_transform_from_coords(clc)
    cols, rows = inv @ (np.asarray(xs), np.asarray(ys))
    cols = np.floor(cols).astype(np.intp)
    rows = np.floor(rows).astype(np.intp)
    ny, nx = clc.shape
    inside = (cols >= 0) & (cols < nx) & (rows >= 0) & (rows < ny)

    out = np.full(lon_a.shape, CLC_NODATA, dtype=np.uint16)
    out[inside] = clc.values[rows[inside], cols[inside]]
    return out


def clc_name(code: int) -> str:
    """Human-readable name of a CLC class code."""
    return CLC_CLASSES.get(int(code), "No data")
