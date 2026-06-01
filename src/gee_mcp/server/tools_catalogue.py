"""Catalogue / metadata tools and prompts for the GEE MCP server.

Tools
-----
- ``list_datasets``
- ``get_dataset_info``
- ``extract_metadata``
- ``analyze_metadata``
- ``get_dataset_metadata``
- ``check_imagery_availability``

Prompts
-------
- ``get_llm_prompt``
- ``get_metadata_prompt``
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import ee
import requests
from bs4 import BeautifulSoup
from markitdown import MarkItDown

from . import helpers
from .app import mcp
from .constants import BASE_URL


# -------------------------------------------------------------------
# Tool: list_datasets
# -------------------------------------------------------------------
@mcp.tool(description="List all available Google Earth Engine datasets")
def list_datasets() -> str:
    """List all available datasets in the GEE catalogue.

    Returns a JSON string containing dataset IDs, URLs, and short
    descriptions.  Results are cached for one hour first in mem, then in file.
    """
    if helpers._datasets_cache and (
        time.time() - helpers._datasets_last_update < 3600
    ):
        return helpers._datasets_cache

    import tempfile

    temp_dir = tempfile.gettempdir()
    datasets_metadata_file = os.path.join(
        temp_dir, "gee-datasets-metadata.json"
    )
    if os.path.exists(
        datasets_metadata_file
    ) and not helpers.is_file_older_than_one_hour(datasets_metadata_file):
        with open(datasets_metadata_file, "r") as f:
            r = json.load(f)
            r = [
                {k: v if v is not None else "" for k, v in ri.items()}
                for ri in r
            ]
            return json.dumps(r, indent=2)

    try:
        response = requests.get(BASE_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        datasets = []
        seen_ids: set[str] = set()

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/earth-engine/datasets/catalog/" in href:
                id_part = href.split("/earth-engine/datasets/catalog/")[-1]
                if not id_part:
                    continue
                if id_part in seen_ids:
                    continue

                seen_ids.add(id_part)
                title = link.get_text(strip=True)
                full_url = (
                    f"https://developers.google.com{href}"
                    if href.startswith("/")
                    else href
                )

                # Resolve the GEE asset ID from STAC metadata
                stac = helpers._fetch_stac_json(id_part)
                dataset_id = stac.get("id") if stac else None

                datasets.append(
                    {
                        "catalog_id": id_part,
                        "title": title,
                        "description": title,
                        "url": full_url,
                        "dataset_id": dataset_id,
                    }
                )

        with open(datasets_metadata_file, "w") as f:
            json.dump(datasets, f, indent=2)

        helpers._datasets_cache = json.dumps(datasets, indent=2)
        helpers._datasets_last_update = time.time()
        return helpers._datasets_cache

    except requests.exceptions.Timeout:
        return "Error: Request timed out while scraping datasets."
    except Exception as e:
        return f"Error scraping datasets: {e!s}"


# -------------------------------------------------------------------
# Tool: get_dataset_info
# -------------------------------------------------------------------
def _get_dataset_info(dataset_id: str) -> str:
    """Retrieve detailed information about a specific dataset.

    Returns a markdown string of the dataset page stripped of header
    and footer content.  Used internally and also exposed via the
    MCP tool ``get_dataset_info``.
    """
    url_id = dataset_id.replace("/", "_")
    url = f"{BASE_URL}/{url_id}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            return json.dumps(
                {
                    "error": "Dataset not found",
                    "id": dataset_id,
                    "url": url,
                }
            )
        response.raise_for_status()

        md_engine = MarkItDown()
        result = md_engine.convert_response(response)

        t = result.text_content
        t = t[t.find("\n#") :]

        if "### Connect" in t[len(t) // 2 :]:
            t = t[: t.find("### Connect")]

        if "Need to tell us more?" in t[len(t) // 2 :]:
            t = t[: t.find("Need to tell us more?")]

        return t

    except requests.exceptions.Timeout:
        return json.dumps({"error": "Request timed out", "id": dataset_id})
    except Exception as e:
        return json.dumps({"error": str(e), "id": dataset_id})


@mcp.tool(
    description=("Get detailed information about a specific GEE dataset")
)
def get_dataset_info(dataset_id: str) -> str:
    """Get detailed information about a specific GEE dataset."""
    return _get_dataset_info(dataset_id)


# -------------------------------------------------------------------
# Tool: extract_metadata
# -------------------------------------------------------------------
@mcp.tool(
    description=(
        "Extract structured metadata (bands, pixel size, "
        "availability, cadence) from a GEE dataset page"
    )
)
def extract_metadata(dataset_id: str) -> str:
    """Fetch a dataset page and extract structured metadata.

    Wraps ``utils.extract_dataset_metadata``, converting the
    resulting pandas DataFrame to JSON records.
    """
    from .utils import extract_dataset_metadata

    page_content = _get_dataset_info(dataset_id)
    if page_content.startswith("{"):
        try:
            err = json.loads(page_content)
            if "error" in err:
                return page_content
        except json.JSONDecodeError:
            pass

    result = extract_dataset_metadata(page_content)

    # Convert DataFrame to serialisable form
    data_records = result["data"].to_dict(orient="records")
    return json.dumps(
        {
            "data": data_records,
            "pixel_size": result["pixel_size"],
            "availability_start_date": result["availability_start_date"],
            "availability_end_date": result["availability_end_date"],
            "cadence": result["cadence"],
        },
        indent=2,
        default=str,
    )


# -------------------------------------------------------------------
# Tool: analyze_metadata
# -------------------------------------------------------------------
@mcp.tool(
    description=(
        "Use Gemini AI to analyse a GEE dataset description and "
        "extract structured metadata including bands, applications, "
        "and cadence"
    )
)
def analyze_metadata(dataset_id: str) -> str:
    """Fetch a dataset page and use Gemini to analyse metadata.

    The Gemini client is created lazily so that the server can start
    without a Gemini API key if this tool is not invoked.
    """
    from .llm import GoogleLLM
    from .utils import analyze_dataset_metadata

    page_content = _get_dataset_info(dataset_id)
    if page_content.startswith("{"):
        try:
            err = json.loads(page_content)
            if "error" in err:
                return page_content
        except json.JSONDecodeError:
            pass

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return json.dumps(
            {
                "error": (
                    "GEMINI_API_KEY or GOOGLE_API_KEY environment "
                    "variable is required for analyze_metadata"
                )
            }
        )

    genai = GoogleLLM(api_key=api_key)
    response = analyze_dataset_metadata(genai, page_content)

    if isinstance(response, dict) and "answer" in response:
        return json.dumps(response["answer"], indent=2, default=str)
    return json.dumps(response, indent=2, default=str)


# -------------------------------------------------------------------
# Tool: get_dataset_metadata  (STAC-based structured metadata)
# -------------------------------------------------------------------
@mcp.tool(
    description=(
        "Get structured STAC metadata for a GEE dataset including "
        "bands, temporal interval, spatial bbox, and revisit "
        "interval"
    )
)
def get_dataset_metadata(catalog_id: str) -> str:
    """Fetch STAC metadata for a single GEE dataset.

    Args:
        catalog_id: Underscore-based catalogue ID from
            ``list_datasets``
            (e.g. ``COPERNICUS_S2_SR_HARMONIZED``).
    """
    stac = helpers._fetch_stac_json(catalog_id)
    if not stac:
        return json.dumps(
            {
                "error": "STAC metadata not found",
                "catalog_id": catalog_id,
            }
        )

    bands = []
    for band in stac.get("summaries", {}).get("eo:bands", []):
        name = band.get("name")
        if name:
            bands.append(name)

    temporal_interval = None
    try:
        temporal_interval = stac["extent"]["temporal"]["interval"][0]
    except (KeyError, IndexError, TypeError):
        pass

    spatial_bbox = None
    try:
        spatial_bbox = stac["extent"]["spatial"]["bbox"][0]
    except (KeyError, IndexError, TypeError):
        pass

    description = stac.get("description", "")
    if len(description) > 500:
        description = description[:500] + "..."

    result = {
        "catalog_id": catalog_id,
        "dataset_id": stac.get("id"),
        "title": stac.get("title"),
        "description": description,
        "gee_type": stac.get("gee:type"),
        "temporal_interval": temporal_interval,
        "spatial_bbox": spatial_bbox,
        "revisit_interval": stac.get("gee:interval"),
        "bands": bands,
        "keywords": stac.get("keywords", []),
    }
    return json.dumps(result, indent=2)


# -------------------------------------------------------------------
# Tool: check_imagery_availability
# -------------------------------------------------------------------
@mcp.tool(
    description=(
        "Check imagery availability for a GEE dataset within a "
        "date range and optional bounding box"
    )
)
def check_imagery_availability(
    dataset_id: str,
    start_date: str,
    end_date: str,
    north: Optional[float] = None,
    south: Optional[float] = None,
    east: Optional[float] = None,
    west: Optional[float] = None,
    max_cloud_cover: Optional[float] = None,
) -> str:
    """Check how many images exist for a dataset with filters.

    Args:
        dataset_id: GEE ImageCollection asset ID,
            e.g. ``COPERNICUS/S2_SR_HARMONIZED``.
        start_date: Start of date range (YYYY-MM-DD, inclusive).
        end_date: End of date range (YYYY-MM-DD, exclusive).
        north: Northern latitude of bounding box (-90 to 90).
        south: Southern latitude of bounding box (-90 to 90).
        east: Eastern longitude of bounding box (-180 to 180).
        west: Western longitude of bounding box (-180 to 180).
            All four bbox parameters must be provided together
            or omitted entirely.
        max_cloud_cover: Maximum cloud cover percentage (0-100).
            Only applied if the dataset has a cloud cover property.
    """
    # Validate bounding box: all-or-nothing
    bbox_params = [north, south, east, west]
    bbox_provided = [p is not None for p in bbox_params]
    if any(bbox_provided) and not all(bbox_provided):
        return json.dumps(
            {
                "error": (
                    "Incomplete bounding box. All four parameters "
                    "(north, south, east, west) must be provided "
                    "together."
                ),
                "dataset_id": dataset_id,
            }
        )
    has_bbox = all(bbox_provided)

    try:
        collection = ee.ImageCollection(dataset_id).filterDate(
            start_date, end_date
        )

        if has_bbox:
            bbox = ee.Geometry.Rectangle([west, south, east, north])
            collection = collection.filterBounds(bbox)

        count = collection.size().getInfo()

        if count == 0:
            total_size = ee.ImageCollection(dataset_id).size().getInfo()
            stac_interval = None
            if total_size > 0:
                cat_id = dataset_id.replace("/", "_")
                stac = helpers._fetch_stac_json(cat_id)
                if stac:
                    try:
                        stac_interval = stac["extent"]["temporal"]["interval"][
                            0
                        ]
                    except (KeyError, IndexError, TypeError):
                        pass

            message = "No images found matching the specified criteria."
            if total_size > 0:
                message = (
                    f"No images match the date filter, but the "
                    f"collection contains {total_size} images "
                    "total. This dataset may use nominal "
                    "timestamps that do not match the documented "
                    "availability range."
                )

            return json.dumps(
                {
                    "dataset_id": dataset_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "image_count": 0,
                    "total_collection_size": total_size,
                    "stac_temporal_interval": stac_interval,
                    "message": message,
                    "bounding_box": (
                        {
                            "north": north,
                            "south": south,
                            "east": east,
                            "west": west,
                        }
                        if has_bbox
                        else None
                    ),
                    "max_cloud_cover_filter": max_cloud_cover,
                },
                indent=2,
            )

        # Detect cloud cover property
        cloud_property = None
        first_props = collection.first().propertyNames().getInfo()
        for prop in [
            "CLOUDY_PIXEL_PERCENTAGE",
            "CLOUD_COVER",
            "CLOUD_COVER_LAND",
        ]:
            if prop in first_props:
                cloud_property = prop
                break

        # Apply cloud filter if requested and property exists
        if max_cloud_cover is not None and cloud_property:
            collection = collection.filter(
                ee.Filter.lte(cloud_property, max_cloud_cover)
            )
            count = collection.size().getInfo()

            if count == 0:
                return json.dumps(
                    {
                        "dataset_id": dataset_id,
                        "start_date": start_date,
                        "end_date": end_date,
                        "image_count": 0,
                        "message": (
                            f"No images found with "
                            f"{cloud_property} "
                            f"<= {max_cloud_cover}."
                        ),
                        "cloud_cover_property": cloud_property,
                        "bounding_box": (
                            {
                                "north": north,
                                "south": south,
                                "east": east,
                                "west": west,
                            }
                            if has_bbox
                            else None
                        ),
                        "max_cloud_cover_filter": max_cloud_cover,
                    },
                    indent=2,
                )

        # Date range of matched images
        date_range = collection.reduceColumns(
            ee.Reducer.minMax(), ["system:time_start"]
        ).getInfo()
        earliest_ms = date_range.get("min")
        latest_ms = date_range.get("max")
        earliest_date = (
            datetime.fromtimestamp(
                earliest_ms / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d")
            if earliest_ms
            else None
        )
        latest_date = (
            datetime.fromtimestamp(latest_ms / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
            if latest_ms
            else None
        )

        # Cloud cover statistics
        cloud_stats = None
        if cloud_property:
            stats = collection.aggregate_stats(cloud_property).getInfo()
            cloud_stats = {
                "property_name": cloud_property,
                "min": round(stats.get("min", 0), 2),
                "max": round(stats.get("max", 0), 2),
                "mean": round(stats.get("mean", 0), 2),
            }
        elif max_cloud_cover is not None:
            cloud_stats = {
                "property_name": None,
                "message": (
                    "No cloud cover property found in this "
                    "dataset. Cloud filter was not applied."
                ),
            }

        # Sample of first 10 image dates
        sample_limit = min(count, 10)
        sample_list = (
            collection.sort("system:time_start")
            .limit(sample_limit)
            .aggregate_array("system:time_start")
            .getInfo()
        )
        sample_dates = (
            [
                datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%d"
                )
                for ts in sample_list
            ]
            if sample_list
            else []
        )

        result = {
            "dataset_id": dataset_id,
            "start_date": start_date,
            "end_date": end_date,
            "image_count": count,
            "date_range": {
                "earliest": earliest_date,
                "latest": latest_date,
            },
            "cloud_cover_stats": cloud_stats,
            "sample_image_dates": sample_dates,
            "bounding_box": (
                {
                    "north": north,
                    "south": south,
                    "east": east,
                    "west": west,
                }
                if has_bbox
                else None
            ),
            "max_cloud_cover_filter": max_cloud_cover,
        }
        return json.dumps(result, indent=2)

    except ee.EEException as e:
        return json.dumps(
            {
                "error": f"Earth Engine error: {e!s}",
                "dataset_id": dataset_id,
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "error": f"Unexpected error: {e!s}",
                "dataset_id": dataset_id,
            }
        )


# -------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------
@mcp.prompt
def get_llm_prompt(query: str) -> str:
    """Generate a download-oriented system prompt for the LLM."""
    return f"""
You are an expert GEE assistant responsible for translating user requests
into precise tool calls.  Your task is to answer the user's query about
satellite image downloads by using the `download_satellite_image` tool.

**VERY IMPORTANT:** You must determine the user's region of interest and
format it correctly.
- If the user provides a latitude and longitude, use the `latitude` and
  `longitude` parameters.
- If the user provides a bounding box (e.g., [min_lon, min_lat, max_lon,
  max_lat]), you have two options:
  1. **(Preferred)** Pass the coordinates as a simple list to the
     `bounding_box` parameter.
  2. Convert it into a valid GeoJSON string and pass it to the
     `region_geojson` parameter.
- Use the `bounding_box` parameter for simple rectangular areas.
  Use `region_geojson` for more complex shapes like polygons.
- Coordinates MUST be in [longitude, latitude] order. Do NOT swap to
  [latitude, longitude].
- For bounding boxes, preserve the exact order
  [min_lon, min_lat, max_lon, max_lat].
- For GeoJSON, every coordinate pair must be [lon, lat].
- Provide only one of: coordinates, a bounding box, or GeoJSON.

Extract all other parameters like dates, cloud cover, and image count
from the query.

Query: {query}
"""


@mcp.prompt
def get_metadata_prompt(query: str) -> str:
    """Generate a dataset-discovery system prompt for the LLM."""
    return f"""
You are an expert Google Earth Engine dataset specialist.  Your task is
to help the user find, explore, and understand GEE datasets.

You have access to the following tools:
- `list_datasets` -- list all available GEE datasets
- `get_dataset_info` -- get detailed markdown information for a dataset
- `get_dataset_metadata` -- get structured STAC metadata (bands,
  temporal interval, spatial bbox, revisit interval) for a dataset
- `check_imagery_availability` -- check image counts, date range,
  and cloud cover statistics for a dataset
- `extract_metadata` -- extract structured metadata (bands, pixel size,
  availability, cadence)
- `analyze_metadata` -- use Gemini AI to produce a rich structured
  analysis of a dataset
- `compute_index` -- compute spectral indices (NDVI, NDWI, EVI, NBR,
  NDBI, SAVI) or custom band math over a region
- `zonal_statistics` -- compute summary statistics for bands or indices
  within a region
- `temporal_composite` -- create cloud-free composites (median, mosaic,
  greenest pixel, most recent)
- `mask_by_raster` -- apply DEM/land-cover masks to imagery and compute
  statistics on the masked result
- `threshold_area` -- compute the area of pixels meeting a threshold
  condition on a band, index, or expression
- `multi_period_analysis` -- compare the same metric across multiple
  date ranges for temporal change detection

Use these tools to answer the user's query as comprehensively as
possible.

Query: {query}
"""
