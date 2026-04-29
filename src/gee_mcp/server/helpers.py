"""Shared helper functions for the GEE MCP server."""

import functools
import inspect
import json
import logging
import os
from typing import Any, Dict, List, Optional

import ee
import requests
import re
from loguru import logger

from .constants import (
    _REDUCER_MAP,
    MAX_PIXELS_PER_DOWNLOAD,
    SPECTRAL_INDICES,
    STAC_BASE_URL,
)
from .models import RegionParams

import os
from datetime import datetime, timedelta

# removes leading spaces of each line in a string
remove_leading_spaces = lambda t: '\n'.join([l.strip() for l in t.splitlines()])

class NoTagFoundError(Exception):
    pass

def extract_xml_tag(text, tag):
    """Extract content between ``<tag>`` and ``</tag>`` in text."""
    p1 = text.find(f"<{tag}>")
    p2 = text.find(f"</{tag}>")

    if p1 < 0 or p2 <= p1:
        raise NoTagFoundError(f"no {tag} found in genai response")

    return text[p1 + len(tag) + 2 : p2]


def extract_tag(text, tag):
    """Extract the first ```{tag} ... ``` markdown fence from text.

    Uses a non-greedy match so that subsequent fences in the same text
    are not consumed.
    """
    pattern = rf"```{tag}(.*?)```"
    if match := re.search(pattern, text, flags=re.DOTALL):
        return match.group(1)
    raise NoTagFoundError(f"no {tag} found in genai response")

def is_file_older_than_one_hour(filepath):
    """
    Checks if the file at the given filepath was modified more than one hour ago.

    Args:
        filepath (str or os.PathLike): The path to the file.

    Returns:
        bool: True if the file is older than one hour, False otherwise.
    """
    # Get the file's last modification timestamp in seconds since the epoch
    modification_timestamp = os.path.getmtime(filepath)
    
    # Convert the timestamp to a datetime object
    modification_time = datetime.fromtimestamp(modification_timestamp)
    
    # Calculate the cutoff time (one hour ago from now)
    cutoff_time = datetime.now() - timedelta(hours=1)
    
    # Compare the file's modification time with the cutoff time
    return modification_time < cutoff_time


# ------------------------------------------------------------------
# Dataset cache (for list_datasets)
# ------------------------------------------------------------------
_datasets_cache: Optional[str] = None
_datasets_last_update: float = 0

# ------------------------------------------------------------------
# STAC metadata cache
# ------------------------------------------------------------------
_stac_cache: dict[str, Optional[dict]] = {}


def _fetch_stac_json(catalog_id: str) -> Optional[dict]:
    """Fetch the full STAC catalogue JSON for a dataset.

    The STAC files live at ``{STAC_BASE_URL}/{PROVIDER}/{id}.json``
    where PROVIDER is the first underscore-delimited segment of the
    *catalog_id*.  Results are cached in-memory.
    """
    if catalog_id in _stac_cache:
        return _stac_cache[catalog_id]

    provider = catalog_id.split("_")[0]
    stac_url = f"{STAC_BASE_URL}/{provider}/{catalog_id}.json"
    try:
        resp = requests.get(stac_url, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except Exception:
        result = None

    _stac_cache[catalog_id] = result
    return result


# ------------------------------------------------------------------
# Region builder
# ------------------------------------------------------------------


def _build_region(params: RegionParams) -> ee.Geometry:
    """Convert RegionParams into an ee.Geometry."""
    if params.region_geojson:
        try:
            geojson_dict = json.loads(params.region_geojson)
            logging.info("GeoJSON: %s", geojson_dict)
            return ee.Geometry(geojson_dict)
        except Exception as e:
            raise ValueError(f"Invalid GeoJSON provided: {e}")
    elif params.bounding_box:
        try:
            min_lon, min_lat, max_lon, max_lat = params.bounding_box
            logging.info(
                "Bounding box: %s, %s, %s, %s",
                min_lon,
                min_lat,
                max_lon,
                max_lat,
            )
            coords = [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
            return ee.Geometry.Polygon(coords)
        except Exception as e:
            raise ValueError(
                "Invalid bounding_box provided. "
                "Ensure it is a list of four numbers. "
                f"Error: {e}"
            )
    else:
        point = ee.Geometry.Point([params.longitude, params.latitude])
        logging.info("Point: %s", point)
        return point.buffer(params.buffer_m)


# ------------------------------------------------------------------
# Download helper functions
# ------------------------------------------------------------------


def _check_and_split_region(region: ee.Geometry, scale: int) -> List[ee.Geometry]:
    """Check pixel count and recursively split if too large."""
    try:
        proj = ee.Projection("EPSG:4326").atScale(scale)
        pixel_area_image = ee.Image.pixelArea().reproject(proj)

        area = region.area(maxError=1).getInfo()

        pixel_area_at_centroid = (
            pixel_area_image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=region.centroid(maxError=1),
                scale=scale,
            )
            .get("area")
            .getInfo()
        )

        if pixel_area_at_centroid is None or pixel_area_at_centroid == 0:
            raise ValueError("Could not determine pixel area.")

        total_pixels = area / pixel_area_at_centroid

    except Exception as e:
        logging.error(
            "Could not compute region size: %s. "
            "Assuming it's too large and splitting.",
            e,
        )
        total_pixels = MAX_PIXELS_PER_DOWNLOAD + 1

    if total_pixels <= MAX_PIXELS_PER_DOWNLOAD:
        logging.info(
            "Region size is acceptable (%.0f pixels).",
            total_pixels,
        )
        return [region]
    else:
        logging.info(
            "Region too large (%.0f pixels). " "Splitting into 4 quadrants.",
            total_pixels,
        )
        bounds = region.bounds().getInfo()["coordinates"][0]
        center = region.centroid(maxError=1).getInfo()["coordinates"]
        lon_c, lat_c = center[0], center[1]

        min_lon, min_lat = bounds[0][0], bounds[0][1]
        max_lon, max_lat = bounds[2][0], bounds[2][1]

        quadrants = [
            ee.Geometry.Rectangle([min_lon, min_lat, lon_c, lat_c]),
            ee.Geometry.Rectangle([lon_c, min_lat, max_lon, lat_c]),
            ee.Geometry.Rectangle([min_lon, lat_c, lon_c, max_lat]),
            ee.Geometry.Rectangle([lon_c, lat_c, max_lon, max_lat]),
        ]

        tiles: List[ee.Geometry] = []
        for q in quadrants:
            tiles.extend(_check_and_split_region(q, scale))
        return tiles


def _split_region_quadrants(
    region: ee.Geometry,
) -> List[ee.Geometry]:
    """Split a region into four rectangular quadrants."""
    bounds = region.bounds().getInfo()["coordinates"][0]
    center = region.centroid(maxError=1).getInfo()["coordinates"]
    lon_c, lat_c = center[0], center[1]
    min_lon, min_lat = bounds[0][0], bounds[0][1]
    max_lon, max_lat = bounds[2][0], bounds[2][1]
    return [
        ee.Geometry.Rectangle([min_lon, min_lat, lon_c, lat_c]),
        ee.Geometry.Rectangle([lon_c, min_lat, max_lon, lat_c]),
        ee.Geometry.Rectangle([min_lon, lat_c, lon_c, max_lat]),
        ee.Geometry.Rectangle([lon_c, lat_c, max_lon, max_lat]),
    ]


def _download_with_fallback(
    selected_bands: ee.Image,
    region: ee.Geometry,
    scale: int,
    prefix: str,
    depth: int = 0,
) -> List[str]:
    """Download with recursive split / scale-backoff."""
    download_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "download")
    os.makedirs(download_dir, exist_ok=True)

    download_region = region.bounds().getInfo()["coordinates"]
    download_params = {
        "scale": scale,
        "region": json.dumps(download_region),
        "format": "GEO_TIFF",
    }
    try:
        url = selected_bands.getDownloadURL(download_params)
        file_path = os.path.join(download_dir, f"{prefix}.tif")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return [os.path.abspath(file_path)]
    except Exception as e:
        err_msg = str(e)
        if (
            "Total request size" in err_msg
            and "must be less than or equal to" in err_msg
        ):
            logging.info(
                "Tile too large at scale=%d. depth=%d. " "Attempting fallback...",
                scale,
                depth,
            )
            if depth < 2:
                results: List[str] = []
                for idx, sub in enumerate(_split_region_quadrants(region), start=1):
                    results.extend(
                        _download_with_fallback(
                            selected_bands,
                            sub,
                            scale,
                            f"{prefix}_q{idx}",
                            depth + 1,
                        )
                    )
                return results
            if scale < 160 and depth < 5:
                new_scale = min(scale * 2, 160)
                logging.info(
                    "Increasing scale to %d and retrying " "for prefix=%s",
                    new_scale,
                    prefix,
                )
                return _download_with_fallback(
                    selected_bands,
                    region,
                    new_scale,
                    f"{prefix}_s{new_scale}",
                    depth + 1,
                )
            logging.warning(
                "Giving up on region after fallbacks " "for prefix=%s: %s",
                prefix,
                err_msg,
            )
            return []
        raise


# ------------------------------------------------------------------
# Shared: build filtered collection + median composite
# ------------------------------------------------------------------


def _build_collection(
    dataset: str,
    start_date: str,
    end_date: str,
    region: ee.Geometry,
    max_cloud_cover: Optional[float] = 20.0,
    cloud_filter_property: Optional[str] = None,
) -> ee.ImageCollection:
    """Build a filtered ee.ImageCollection."""
    collection = (
        ee.ImageCollection(dataset)
        .filterDate(start_date, end_date)
        .filterBounds(region)
    )
    if cloud_filter_property and max_cloud_cover is not None:
        collection = collection.filter(
            ee.Filter.lt(cloud_filter_property, max_cloud_cover)
        )
    elif dataset.startswith("COPERNICUS/S2") and max_cloud_cover is not None:
        collection = collection.filter(
            ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover)
        )
    elif dataset.startswith("LANDSAT") and max_cloud_cover is not None:
        collection = collection.filter(ee.Filter.lt("CLOUD_COVER", max_cloud_cover))
    return collection


# ------------------------------------------------------------------
# Shared helpers: resolve target image and apply masks
# ------------------------------------------------------------------


def _resolve_target_image(
    composite: ee.Image,
    index_name: Optional[str] = None,
    expression: Optional[str] = None,
    bands: Optional[str] = None,
) -> ee.Image:
    """Pick band/index/expression from a composite.

    Priority: index_name > expression > bands > full composite.
    """
    if index_name:
        name = index_name.upper()
        if name not in SPECTRAL_INDICES:
            raise ValueError(
                f"Unknown index_name '{name}'. "
                "Supported: "
                f"{', '.join(SPECTRAL_INDICES.keys())}."
            )
        spec = SPECTRAL_INDICES[name]
        band_names = spec["bands"]
        if len(band_names) == 2:
            band_vals = list(band_names.values())
            return composite.normalizedDifference(band_vals).rename(name)
        band_map = {k: composite.select(v) for k, v in band_names.items()}
        return composite.expression(spec["formula"], band_map).rename(name)

    if expression:
        band_info = composite.bandNames().getInfo()
        band_map = {b: composite.select(b) for b in band_info}
        return composite.expression(expression, band_map).rename("custom_index")

    if bands:
        band_list = [b.strip() for b in bands.split(",") if b.strip()]
        return composite.select(band_list)

    return composite


def _apply_ancillary_mask(
    image: ee.Image,
    mask_dataset: str,
    mask_band: str,
    mask_min: Optional[float] = None,
    mask_max: Optional[float] = None,
) -> ee.Image:
    """Load an ancillary raster, build a range mask, apply it."""
    ancillary = ee.Image(mask_dataset).select(mask_band)
    mask = ee.Image.constant(1)
    if mask_min is not None:
        mask = mask.And(ancillary.gte(mask_min))
    if mask_max is not None:
        mask = mask.And(ancillary.lte(mask_max))
    return image.updateMask(mask)


def _apply_pixel_mask(
    image: ee.Image,
    pixel_mask_band: str,
    pixel_mask_min: Optional[float] = None,
    pixel_mask_max: Optional[float] = None,
) -> ee.Image:
    """Mask pixels by value range on a band within the image."""
    qa = image.select(pixel_mask_band)
    mask = ee.Image.constant(1)
    if pixel_mask_min is not None:
        mask = mask.And(qa.gte(pixel_mask_min))
    if pixel_mask_max is not None:
        mask = mask.And(qa.lte(pixel_mask_max))
    return image.updateMask(mask)


# ------------------------------------------------------------------
# Shared: build combined reducer from a comma-separated list
# ------------------------------------------------------------------


def _build_reducer(reducers: str):
    """Return a combined ee.Reducer from a comma-separated string."""
    requested = [r.strip() for r in reducers.split(",") if r.strip()]
    if not requested:
        raise ValueError("No reducers specified.")
    combined = _REDUCER_MAP[requested[0]]()
    for r_name in requested[1:]:
        if r_name in _REDUCER_MAP:
            combined = combined.combine(_REDUCER_MAP[r_name](), sharedInputs=True)
    return combined


# ------------------------------------------------------------------
# Flatten-model decorator for FastMCP tool registration
# ------------------------------------------------------------------


def _flatten_model(model_cls):
    """Flatten a Pydantic model into individual keyword args.

    When a tool is defined as ``def tool(args: MyModel)``, FastMCP
    exposes a nested ``{"args": {...}}`` schema.  The Google ADK
    flattens that schema for Gemini, so Gemini sends flat kwargs that
    FastMCP then rejects.

    Apply this decorator **between** ``@mcp.tool`` and the function
    so that FastMCP sees individual keyword parameters matching the
    model's fields::

        @mcp.tool
        @_flatten_model(MyModel)
        def my_tool(args: MyModel) -> dict:
            ...
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if args and isinstance(args[0], model_cls):
                return fn(args[0])
            return fn(model_cls(**kwargs))

        params = []
        annotations: Dict[str, Any] = {}
        for name, field_info in model_cls.model_fields.items():
            if field_info.is_required():
                default = inspect.Parameter.empty
            else:
                default = field_info.default
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=field_info.annotation,
                )
            )
            annotations[name] = field_info.annotation
        ret = inspect.signature(fn).return_annotation
        annotations["return"] = ret
        wrapper.__signature__ = inspect.Signature(
            params,
            return_annotation=ret,
        )
        wrapper.__annotations__ = annotations
        return wrapper

    return decorator
