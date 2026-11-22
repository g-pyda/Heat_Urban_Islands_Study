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


# ---------------------------------------------------------------------------
# Land-cover class masks
# ---------------------------------------------------------------------------
# Classes relevant for the urban heat island analysis:
# built-up surfaces (urban fabric + industrial) versus the two dominant
# rural surface types around Krakow (arable land and broad-leaved forest).

URBAN_CODES = (111, 112, 121)   # continuous/discontinuous fabric, industrial
FOREST_CODES = (311,)           # broad-leaved forest
ARABLE_CODES = (211,)           # non-irrigated arable land

CLASS_GROUPS: dict[str, tuple[int, ...]] = {
    "urban": URBAN_CODES,
    "forest": FOREST_CODES,
    "arable": ARABLE_CODES,
}


def class_mask(clc: xr.DataArray, codes) -> xr.DataArray:
    """Boolean mask of pixels whose CLC code is in ``codes``.

    ``codes`` is a single code or an iterable of codes; unknown codes raise
    ``ValueError``. No-data pixels are always ``False``.
    """
    codes = tuple(np.atleast_1d(codes).astype(int).tolist())
    unknown = [code for code in codes if code not in CLC_CLASSES]
    if unknown:
        raise ValueError(f"Unknown CLC codes: {unknown}")

    mask = xr.DataArray(
        np.isin(clc.values, codes),
        dims=clc.dims,
        coords=clc.coords,
        attrs={**clc.attrs, "clc_codes": list(codes)},
    )
    mask.attrs.pop("nodata", None)
    return mask


def valid_mask(clc: xr.DataArray) -> xr.DataArray:
    """Boolean mask of pixels holding a valid CLC class (not no-data)."""
    return class_mask(clc, CLC_CODES).rename("valid")


def land_cover_masks(
    clc: xr.DataArray,
    groups: dict[str, tuple[int, ...]] | None = None,
) -> xr.Dataset:
    """Boolean masks for each land-cover group plus the ``valid`` mask.

    Parameters
    ----------
    clc:
        CLC code raster from :func:`load_corine`.
    groups:
        Mapping ``name -> CLC codes``; defaults to :data:`CLASS_GROUPS`
        (``urban``, ``forest``, ``arable``).
    """
    groups = CLASS_GROUPS if groups is None else groups
    masks = {name: class_mask(clc, codes) for name, codes in groups.items()}
    masks["valid"] = valid_mask(clc)
    ds = xr.Dataset(masks)
    ds.attrs = {k: v for k, v in clc.attrs.items() if k != "nodata"}
    for name, codes in groups.items():
        ds[name].attrs["clc_names"] = [clc_name(code) for code in codes]
    return ds


def class_shares(clc: xr.DataArray) -> dict[int, float]:
    """Share of each CLC class among the valid pixels of the raster."""
    values = clc.values[clc.values != CLC_NODATA]
    if values.size == 0:
        return {}
    codes, counts = np.unique(values, return_counts=True)
    return {int(k): n / values.size for k, n in zip(codes, counts)}


# ---------------------------------------------------------------------------
# Urban core and rural reference masks
# ---------------------------------------------------------------------------
# The urban mask is the contiguous built-up area of Krakow (connected urban
# clusters near the city centre), not every built-up pixel in the window.
# The rural reference consists of forest / arable pixels that are
#   * at least ``min_distance_m`` from the urban core (outside its thermal
#     footprint, also wider than a 1 km MODIS pixel),
#   * at most ``max_distance_m`` from it (similar regional climate),
#   * away from villages and small towns (any urban pixel),
#   * optionally at a similar elevation as the city (DEM required) and/or
#     inside a manually drawn polygon.

RURAL_CODES = FOREST_CODES + ARABLE_CODES

DEFAULT_CORE_RADIUS_M = 12_000      # clusters touching this disk form Krakow
DEFAULT_MIN_CLUSTER_PX = 50         # 0.5 km2 at 100 m resolution
DEFAULT_MIN_DISTANCE_M = 5_000      # rural ring inner edge (from urban core)
DEFAULT_MAX_DISTANCE_M = 20_000     # rural ring outer edge
DEFAULT_SETTLEMENT_BUFFER_M = 500   # distance from any other built-up pixel
DEFAULT_MAX_ELEV_DIFF_M = 50.0      # allowed |z - median urban z|


def _pixel_size(da: xr.DataArray) -> tuple[float, float]:
    """(dy, dx) pixel size in metres (positive) from the coordinates."""
    if "res" in da.attrs:
        rx, ry = da.attrs["res"]
        return float(ry), float(rx)
    return (float(abs(da.y[1] - da.y[0])), float(abs(da.x[1] - da.x[0])))


def _like(template: xr.DataArray, values: np.ndarray, name: str, **attrs):
    """DataArray on the grid of ``template`` (keeps CRS information)."""
    base = {k: template.attrs[k] for k in ("crs", "transform", "res")
            if k in template.attrs}
    return xr.DataArray(values, dims=template.dims, coords=template.coords,
                        name=name, attrs={**base, **attrs})


def distance_to(mask: xr.DataArray) -> xr.DataArray:
    """Euclidean distance (metres) from each pixel to the nearest True pixel.

    Pixels inside the mask get 0; if the mask is empty all distances are
    ``inf``.
    """
    from scipy.ndimage import distance_transform_edt

    values = np.asarray(mask.values, bool)
    if not values.any():
        dist = np.full(values.shape, np.inf)
    else:
        dist = distance_transform_edt(~values, sampling=_pixel_size(mask))
    return _like(mask, dist, "distance", units="m")


def distance_from_point(
    clc: xr.DataArray,
    point: tuple[float, float] = KRAKOW_CENTRE,
    crs: str = "EPSG:4326",
) -> xr.DataArray:
    """Distance (metres) of every pixel centre from a point."""
    from pyproj import Transformer

    tr = Transformer.from_crs(crs, clc.attrs["crs"], always_xy=True)
    px, py = tr.transform(*point)
    xx, yy = np.meshgrid(clc.x.values, clc.y.values)
    return _like(clc, np.hypot(xx - px, yy - py), "distance", units="m")


def _densify_ring(ring, n: int) -> list:
    """Insert ``n - 1`` points into every segment of a coordinate ring."""
    pts = np.asarray(ring, float)
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    segs = [a + t * (b - a) for a, b in zip(pts[:-1], pts[1:])]
    return np.vstack(segs + [pts[-1:]]).tolist()


def _densify(geometry: dict, n: int = 100) -> dict:
    """Densify polygon edges so they stay straight lines in the source CRS
    after reprojection (only vertices are transformed otherwise)."""
    kind = geometry["type"]
    if kind == "Polygon":
        coords = [_densify_ring(r, n) for r in geometry["coordinates"]]
    elif kind == "MultiPolygon":
        coords = [[_densify_ring(r, n) for r in poly]
                  for poly in geometry["coordinates"]]
    else:
        raise ValueError(f"Expected Polygon or MultiPolygon, got {kind}")
    return {"type": kind, "coordinates": coords}


def polygon_mask(
    clc: xr.DataArray,
    geometry: dict,
    crs: str = "EPSG:4326",
) -> xr.DataArray:
    """Rasterise a (manually drawn) polygon onto the CLC grid.

    ``geometry`` is a GeoJSON-like mapping (``Polygon``/``MultiPolygon``),
    e.g. a reference area or the city boundary, given in ``crs``.
    Pixels whose centre lies inside the polygon are ``True``.
    """
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom

    geometry = getattr(geometry, "__geo_interface__", geometry)
    geom = transform_geom(crs, clc.attrs["crs"], _densify(geometry))
    inside = geometry_mask(
        [geom], out_shape=clc.shape, transform=_transform_from_coords(clc),
        invert=True,
    )
    return _like(clc, inside, "polygon")


def urban_core_mask(
    clc: xr.DataArray,
    centre: tuple[float, float] = KRAKOW_CENTRE,
    radius_m: float = DEFAULT_CORE_RADIUS_M,
    min_cluster_px: int = DEFAULT_MIN_CLUSTER_PX,
    codes: tuple[int, ...] = URBAN_CODES,
    boundary: dict | None = None,
    boundary_crs: str = "EPSG:4326",
) -> xr.DataArray:
    """Densely built-up area of the city.

    Urban pixels (``codes``) are grouped into 8-connected clusters; a cluster
    belongs to the city if it has at least ``min_cluster_px`` pixels and at
    least one pixel within ``radius_m`` of ``centre``. Optionally the result
    is clipped to an administrative ``boundary`` polygon.
    """
    from scipy.ndimage import label

    urban = class_mask(clc, codes).values
    labels, n = label(urban, structure=np.ones((3, 3), bool))
    if n == 0:
        core = np.zeros(urban.shape, bool)
    else:
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        near = distance_from_point(clc, centre).values <= radius_m
        keep = np.zeros(n + 1, bool)
        keep[np.unique(labels[near & urban])] = True
        keep &= sizes >= min_cluster_px
        keep[0] = False
        core = keep[labels]

    if boundary is not None:
        core &= polygon_mask(clc, boundary, boundary_crs).values

    return _like(clc, core, "urban", clc_codes=list(codes),
                 description="contiguous urban core of the city")


def _align_dem(dem: xr.DataArray, clc: xr.DataArray) -> np.ndarray:
    """DEM values on the CLC grid (DEM must already be in the CLC CRS)."""
    if dem.shape == clc.shape and np.allclose(dem.x, clc.x) and np.allclose(dem.y, clc.y):
        return np.asarray(dem.values, float)
    return np.asarray(dem.interp(x=clc.x, y=clc.y, method="linear").values, float)


def elevation_mask(
    dem: xr.DataArray,
    clc: xr.DataArray,
    urban: xr.DataArray,
    max_diff_m: float = DEFAULT_MAX_ELEV_DIFF_M,
) -> xr.DataArray:
    """Pixels whose elevation is within ``max_diff_m`` of the urban median.

    ``dem`` must be in the CLC CRS (e.g. reprojected with
    ``dem.rio.reproject_match``); it is interpolated onto the CLC grid if
    the grids differ. NaN elevations are excluded.
    """
    z = _align_dem(dem, clc)
    z_urban = z[np.asarray(urban.values, bool)]
    z_urban = z_urban[np.isfinite(z_urban)]
    if z_urban.size == 0:
        raise ValueError("No valid DEM values inside the urban mask")
    z_ref = float(np.median(z_urban))
    with np.errstate(invalid="ignore"):
        ok = np.abs(z - z_ref) <= max_diff_m
    return _like(clc, ok, "elevation_ok", reference_elevation_m=z_ref,
                 max_elevation_diff_m=max_diff_m)


def rural_reference_mask(
    clc: xr.DataArray,
    urban: xr.DataArray | None = None,
    classes: tuple[int, ...] = RURAL_CODES,
    min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
    max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
    settlement_buffer_m: float = DEFAULT_SETTLEMENT_BUFFER_M,
    dem: xr.DataArray | None = None,
    max_elev_diff_m: float = DEFAULT_MAX_ELEV_DIFF_M,
    region: dict | None = None,
    region_crs: str = "EPSG:4326",
) -> xr.DataArray:
    """Rural reference pixels for the urban heat island baseline.

    Parameters
    ----------
    clc:
        CLC code raster.
    urban:
        Urban core mask; computed with :func:`urban_core_mask` if ``None``.
    classes:
        CLC codes allowed as reference (forest 311 and arable 211).
    min_distance_m, max_distance_m:
        Ring around the urban core in which reference pixels are searched.
    settlement_buffer_m:
        Minimum distance to *any* built-up pixel (villages, small towns);
        ``0`` disables the criterion.
    dem, max_elev_diff_m:
        Optional DEM (in the CLC CRS); pixels deviating more than
        ``max_elev_diff_m`` from the median urban elevation are dropped.
    region:
        Optional manually drawn polygon (GeoJSON-like, in ``region_crs``)
        restricting the reference area.
    """
    if min_distance_m > max_distance_m:
        raise ValueError("min_distance_m must not exceed max_distance_m")
    if urban is None:
        urban = urban_core_mask(clc)

    d_core = distance_to(urban).values
    rural = class_mask(clc, classes).values
    rural &= (d_core >= min_distance_m) & (d_core <= max_distance_m)

    if settlement_buffer_m > 0:
        d_any = distance_to(class_mask(clc, URBAN_CODES)).values
        rural &= d_any >= settlement_buffer_m

    attrs = {
        "clc_codes": list(classes),
        "min_distance_m": min_distance_m,
        "max_distance_m": max_distance_m,
        "settlement_buffer_m": settlement_buffer_m,
        "description": "rural reference pixels for the UHI baseline",
    }
    if dem is not None:
        elev = elevation_mask(dem, clc, urban, max_elev_diff_m)
        rural &= elev.values
        attrs.update({k: elev.attrs[k] for k in
                      ("reference_elevation_m", "max_elevation_diff_m")})
    if region is not None:
        rural &= polygon_mask(clc, region, region_crs).values

    if not rural.any():
        logger.warning("Rural reference mask is empty - relax the criteria")
    return _like(clc, rural, "rural", **attrs)


def urban_rural_masks(
    clc: xr.DataArray,
    urban_kwargs: dict | None = None,
    rural_kwargs: dict | None = None,
) -> xr.Dataset:
    """Binary ``urban`` and ``rural`` masks plus distance to the urban core.

    Keyword dictionaries are passed to :func:`urban_core_mask` and
    :func:`rural_reference_mask` respectively.
    """
    urban = urban_core_mask(clc, **(urban_kwargs or {}))
    rural = rural_reference_mask(clc, urban=urban, **(rural_kwargs or {}))
    ds = xr.Dataset({
        "urban": urban,
        "rural": rural,
        "distance_to_urban": distance_to(urban),
    })
    ds.attrs = {k: v for k, v in clc.attrs.items() if k != "nodata"}
    return ds
