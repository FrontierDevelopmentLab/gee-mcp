"""Analysis tools registered on the GEE MCP server.

Tools
-----
- ``download_satellite_image``
- ``compute_index``
- ``zonal_statistics``
- ``temporal_composite``
- ``mask_by_raster``
- ``threshold_area``
- ``multi_period_analysis``
- ``sensitivy_analysus``
"""

import json
import logging
from typing import Any, Dict, List, Optional

import ee

from .app import mcp
from .constants import SPECTRAL_INDICES
from .helpers import (
    _apply_ancillary_mask,
    _apply_pixel_mask,
    _build_collection,
    _build_reducer,
    _build_region,
    _check_and_split_region,
    _download_with_fallback,
    _flatten_model,
    _resolve_target_image,
)
from .models import (
    ComputeIndexParams,
    DownloadParams,
    MaskByRasterParams,
    MultiPeriodParams,
    TemporalCompositeParams,
    ThresholdAreaParams,
    ZonalStatsParams,
)


# -------------------------------------------------------------------
# Tool: download_satellite_image
# -------------------------------------------------------------------
@mcp.tool
@_flatten_model(DownloadParams)
def download_satellite_image(
    args: DownloadParams,
) -> Dict[str, Any]:
    """Download satellite images from GEE.

    Returns a dictionary with file paths and a status message.
    Arguments include coordinates, date range, dataset, bands, scale,
    buffer, cloud cover, and image count.
    """
    # Determine region of interest
    region = _build_region(args)

    # Pre-flight check and AOI tiling
    logging.info("Performing pre-flight checks for region size...")
    tiles = _check_and_split_region(region, args.scale)
    if len(tiles) > 1:
        logging.info(
            "Region is too large for a single download. " "Splitting into %d tiles.",
            len(tiles),
        )

    downloaded_files: List[str] = []
    for i, tile_region in enumerate(tiles):
        if len(tiles) > 1:
            logging.info("Processing tile %d of %d...", i + 1, len(tiles))

        collection = (
            ee.ImageCollection(args.dataset)
            .filterDate(args.start_date, args.end_date)
            .filterBounds(tile_region)
        )

        if (
            args.dataset.startswith("COPERNICUS/S2")
            and args.max_cloud_cover is not None
        ):
            collection = collection.filter(
                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    args.max_cloud_cover,
                )
            )

        collection_size = collection.size().getInfo()
        if collection_size == 0:
            logging.warning(
                "No images found for tile %d that meet the "
                "specified cloud cover rate. Skipping.",
                i + 1,
            )
            continue

        collection = collection.sort("CLOUDY_PIXEL_PERCENTAGE").limit(args.image_count)

        image_list = collection.toList(args.image_count)
        num_images_found = image_list.size().getInfo()

        for j in range(num_images_found):
            image = ee.Image(image_list.get(j))

            date_string = (
                ee.Date(image.get("system:time_start")).format("YYYYMMdd").getInfo()
            )

            if args.bands:
                band_list = [b.strip() for b in args.bands.split(",") if b.strip()]
                selected_bands = image.select(band_list)
            else:
                selected_bands = image

            prefix = f"gee_image_{date_string}_t{i+1}_n{j+1}"

            file_paths = _download_with_fallback(
                selected_bands, tile_region, args.scale, prefix
            )
            if not file_paths:
                logging.warning(
                    "No files produced for tile %d, image %d " "after fallbacks.",
                    i + 1,
                    j + 1,
                )
            else:
                downloaded_files.extend(file_paths)

    message = (
        f"Successfully downloaded {len(downloaded_files)} image(s) "
        f"across {len(tiles)} tile(s)."
    )
    if len(tiles) > 1:
        message += (
            " (Note: Some images were skipped due to insufficient "
            "cloud cover or no images found in some tiles.)"
        )

    return {"files": downloaded_files, "message": message}


# -------------------------------------------------------------------
# Tool: compute_index
# -------------------------------------------------------------------


def _compute_index(
    args: ComputeIndexParams,
) -> Dict[str, Any]:
    """Compute a spectral index over a region."""
    region = _build_region(args)
    collection = _build_collection(
        args.dataset,
        args.start_date,
        args.end_date,
        region,
        args.max_cloud_cover,
        getattr(args, "cloud_filter_property", None),
    )

    count = collection.size().getInfo()
    if count == 0:
        return {"error": ("No images found matching the specified filters.")}

    composite = collection.median()

    # Determine which index to compute
    if args.index_name:
        name = args.index_name.upper()
        spec = SPECTRAL_INDICES[name]
        band_names = spec["bands"]

        # Two-band normalised difference indices
        if len(band_names) == 2:
            band_vals = list(band_names.values())
            index_image = composite.normalizedDifference(band_vals).rename(name)
        else:
            # Multi-band indices (EVI, SAVI)
            band_map = {k: composite.select(v) for k, v in band_names.items()}
            index_image = composite.expression(spec["formula"], band_map).rename(name)
        label = name
    else:
        # Custom expression
        band_info = composite.bandNames().getInfo()
        band_map = {b: composite.select(b) for b in band_info}
        index_image = composite.expression(args.expression, band_map).rename(
            "custom_index"
        )
        label = "custom"

    # Compute statistics
    combined_reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True)
    )

    stats = index_image.reduceRegion(
        reducer=combined_reducer,
        geometry=region,
        scale=args.scale,
        maxPixels=1e9,
    ).getInfo()

    result: Dict[str, Any] = {
        "index": label,
        "image_count_in_collection": count,
        "stats": stats,
    }

    if args.download:
        download_params = {
            "scale": args.scale,
            "region": json.dumps(region.bounds().getInfo()["coordinates"]),
            "format": "GEO_TIFF",
        }
        result["download_url"] = index_image.getDownloadURL(download_params)

    return result


@mcp.tool
@_flatten_model(ComputeIndexParams)
def compute_index(
    args: ComputeIndexParams,
) -> Dict[str, Any]:
    """Compute a spectral index (NDVI, NDWI, EVI, NBR, NDBI, SAVI)
    or a custom band math expression over a region of interest.

    Returns statistics (mean, min, max, stdDev, median) and optionally
    a GeoTIFF download link.
    """
    return _compute_index(args)


# -------------------------------------------------------------------
# Tool: zonal_statistics
# -------------------------------------------------------------------


def _zonal_statistics(
    args: ZonalStatsParams,
) -> Dict[str, Any]:
    """Compute zonal statistics for bands or index over a region."""
    region = _build_region(args)
    collection = _build_collection(
        args.dataset,
        args.start_date,
        args.end_date,
        region,
        args.max_cloud_cover,
        getattr(args, "cloud_filter_property", None),
    )

    count = collection.size().getInfo()
    if count == 0:
        return {"error": ("No images found matching the specified filters.")}

    composite = collection.median()

    # If an index is requested, compute it on the composite
    if args.index_name:
        name = args.index_name.upper()
        if name not in SPECTRAL_INDICES:
            return {
                "error": (
                    f"Unknown index_name '{name}'. "
                    "Supported: "
                    f"{', '.join(SPECTRAL_INDICES.keys())}."
                )
            }
        spec = SPECTRAL_INDICES[name]
        band_names = spec["bands"]
        if len(band_names) == 2:
            band_vals = list(band_names.values())
            image = composite.normalizedDifference(band_vals).rename(name)
        else:
            band_map = {k: composite.select(v) for k, v in band_names.items()}
            image = composite.expression(spec["formula"], band_map).rename(name)
    elif args.bands:
        band_list = [b.strip() for b in args.bands.split(",") if b.strip()]
        image = composite.select(band_list)
    else:
        image = composite

    # Build combined reducer from the requested list
    reducer_map = {
        "mean": ee.Reducer.mean,
        "median": ee.Reducer.median,
        "min": ee.Reducer.min,
        "max": ee.Reducer.max,
        "stdDev": ee.Reducer.stdDev,
        "sum": ee.Reducer.sum,
        "count": ee.Reducer.count,
    }

    requested = [r.strip() for r in args.reducers.split(",") if r.strip()]
    if not requested:
        return {"error": "No reducers specified."}

    combined = reducer_map[requested[0]]()
    for r_name in requested[1:]:
        if r_name in reducer_map:
            combined = combined.combine(reducer_map[r_name](), sharedInputs=True)

    stats = image.reduceRegion(
        reducer=combined,
        geometry=region,
        scale=args.scale,
        maxPixels=1e9,
    ).getInfo()

    area = region.area(maxError=1).getInfo()

    result: Dict[str, Any] = {
        "dataset": args.dataset,
        "region_area_m2": area,
        "image_count_in_collection": count,
        "statistics": stats,
    }

    if args.download:
        download_params = {
            "scale": args.scale,
            "region": json.dumps(region.bounds().getInfo()["coordinates"]),
            "format": "GEO_TIFF",
        }
        result["download_url"] = image.getDownloadURL(download_params)

    return result


@mcp.tool
@_flatten_model(ZonalStatsParams)
def zonal_statistics(
    args: ZonalStatsParams,
) -> Dict[str, Any]:
    """Compute zonal statistics (mean, median, min, max, stdDev, etc.)
    for image bands or a spectral index within a region of interest.

    Returns per-band or per-index statistics and optionally a GeoTIFF
    download link.
    """
    return _zonal_statistics(args)


# -------------------------------------------------------------------
# Tool: temporal_composite
# -------------------------------------------------------------------


def _temporal_composite(
    args: TemporalCompositeParams,
) -> Dict[str, Any]:
    """Create a temporal composite and download it."""
    region = _build_region(args)
    collection = _build_collection(
        args.dataset,
        args.start_date,
        args.end_date,
        region,
        args.max_cloud_cover,
        getattr(args, "cloud_filter_property", None),
    )

    count = collection.size().getInfo()
    if count == 0:
        return {"error": ("No images found matching the specified filters.")}

    # Apply composite method
    if args.method == "median":
        composite = collection.median()
    elif args.method == "mosaic":
        composite = collection.sort("system:time_start", False).mosaic()
    elif args.method == "greenest":

        def _add_ndvi(img):
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            return img.addBands(ndvi)

        composite = collection.map(_add_ndvi).qualityMosaic("NDVI")
    elif args.method == "most_recent":
        composite = collection.sort("system:time_start", False).first()
    else:
        return {"error": f"Unknown method '{args.method}'."}

    # Select bands and clip
    if args.bands:
        band_list = [b.strip() for b in args.bands.split(",") if b.strip()]
        composite = composite.select(band_list)

    composite = composite.clip(region)

    # Download via fallback helper
    prefix = f"composite_{args.method}" f"_{args.start_date}_{args.end_date}"
    files = _download_with_fallback(composite, region, args.scale, prefix)

    return {
        "method": args.method,
        "files": files,
        "message": (f"Created {args.method} composite " f"from {count} images."),
        "image_count_in_collection": count,
    }


@mcp.tool
@_flatten_model(TemporalCompositeParams)
def temporal_composite(
    args: TemporalCompositeParams,
) -> Dict[str, Any]:
    """Create a cloud-free temporal composite from a satellite image
    collection using median, mosaic, greenest pixel, or most recent
    methods. Downloads the result as a GeoTIFF.
    """
    return _temporal_composite(args)


# -------------------------------------------------------------------
# Tool: mask_by_raster
# -------------------------------------------------------------------


def _mask_by_raster(
    args: MaskByRasterParams,
) -> Dict[str, Any]:
    """Mask imagery by an ancillary raster and compute stats."""
    region = _build_region(args)
    collection = _build_collection(
        args.dataset,
        args.start_date,
        args.end_date,
        region,
        args.max_cloud_cover,
        args.cloud_filter_property,
    )

    count = collection.size().getInfo()
    if count == 0:
        return {"error": ("No images found matching the specified filters.")}

    composite = collection.median()
    target = _resolve_target_image(
        composite, args.index_name, args.expression, args.bands
    )
    masked = _apply_ancillary_mask(
        target,
        args.mask_dataset,
        args.mask_band,
        args.mask_min,
        args.mask_max,
    )

    combined = _build_reducer(args.reducers)
    stats = masked.reduceRegion(
        reducer=combined,
        geometry=region,
        scale=args.scale,
        maxPixels=1e9,
    ).getInfo()

    mask_range: Dict[str, Any] = {}
    if args.mask_min is not None:
        mask_range["min"] = args.mask_min
    if args.mask_max is not None:
        mask_range["max"] = args.mask_max

    result: Dict[str, Any] = {
        "mask_dataset": args.mask_dataset,
        "mask_band": args.mask_band,
        "mask_range": mask_range,
        "image_count_in_collection": count,
        "stats": stats,
    }

    if args.download:
        download_params = {
            "scale": args.scale,
            "region": json.dumps(region.bounds().getInfo()["coordinates"]),
            "format": "GEO_TIFF",
        }
        result["download_url"] = masked.getDownloadURL(download_params)

    return result


@mcp.tool
@_flatten_model(MaskByRasterParams)
def mask_by_raster(
    args: MaskByRasterParams,
) -> Dict[str, Any]:
    """Load an ancillary raster (DEM, land cover, etc.) and apply a
    value-range mask to satellite imagery. Returns statistics on the
    masked result and optionally a GeoTIFF download link.
    """
    return _mask_by_raster(args)


# -------------------------------------------------------------------
# Tool: threshold_area
# -------------------------------------------------------------------


def _threshold_area(
    args: ThresholdAreaParams,
) -> Dict[str, Any]:
    """Compute the area of pixels meeting a threshold condition."""
    region = _build_region(args)
    collection = _build_collection(
        args.dataset,
        args.start_date,
        args.end_date,
        region,
        args.max_cloud_cover,
        args.cloud_filter_property,
    )

    count = collection.size().getInfo()
    if count == 0:
        return {"error": ("No images found matching the specified filters.")}

    if args.pixel_mask_band:
        collection = collection.map(
            lambda img: _apply_pixel_mask(
                img,
                args.pixel_mask_band,
                args.pixel_mask_min,
                args.pixel_mask_max,
            )
        )

    composite = collection.median()
    target = _resolve_target_image(
        composite, args.index_name, args.expression, args.band
    )

    if args.mask_dataset:
        target = _apply_ancillary_mask(
            target,
            args.mask_dataset,
            args.mask_band,
            args.mask_min,
            args.mask_max,
        )

    ops = {
        "gte": target.gte,
        "gt": target.gt,
        "lte": target.lte,
        "lt": target.lt,
        "eq": target.eq,
    }
    binary = ops[args.operator](args.threshold)

    area_m2 = (
        ee.Image.pixelArea()
        .updateMask(binary)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=args.scale,
            maxPixels=1e9,
        )
        .get("area")
        .getInfo()
    )
    area_m2 = area_m2 or 0.0

    region_area_m2 = region.area(maxError=1).getInfo()
    area_km2 = area_m2 / 1e6
    region_area_km2 = region_area_m2 / 1e6
    fraction = area_m2 / region_area_m2 if region_area_m2 > 0 else 0.0

    return {
        "area_km2": area_km2,
        "area_m2": area_m2,
        "region_area_km2": region_area_km2,
        "fraction": fraction,
        "threshold": args.threshold,
        "operator": args.operator,
        "image_count_in_collection": count,
    }


@mcp.tool
@_flatten_model(ThresholdAreaParams)
def threshold_area(
    args: ThresholdAreaParams,
) -> Dict[str, Any]:
    """Count the area (km2) of pixels that exceed or fall below a
    threshold on a band, spectral index, or custom expression.
    Optionally apply an ancillary raster mask first.
    """
    return _threshold_area(args)


# -------------------------------------------------------------------
# Tool: multi_period_analysis
# -------------------------------------------------------------------


def _multi_period_analysis(
    args: MultiPeriodParams,
) -> Dict[str, Any]:
    """Run the same analysis across multiple date ranges."""
    region = _build_region(args)
    results = []

    for period in args.periods:
        collection = _build_collection(
            args.dataset,
            period.start_date,
            period.end_date,
            region,
            args.max_cloud_cover,
            args.cloud_filter_property,
        )

        count = collection.size().getInfo()
        if count == 0:
            results.append(
                {
                    "label": period.label,
                    "start_date": period.start_date,
                    "end_date": period.end_date,
                    "error": "No images found.",
                }
            )
            continue

        # Select band(s) before .median() to avoid inhomogeneous
        # band-ordering errors (e.g. MODIS collections where band
        # order varies).
        select_bands = args.band or args.bands
        if select_bands and not args.index_name and not args.expression:
            band_list = [b.strip() for b in select_bands.split(",") if b.strip()]
            collection = collection.select(band_list)

        if args.pixel_mask_band:
            collection = collection.map(
                lambda img: _apply_pixel_mask(
                    img,
                    args.pixel_mask_band,
                    args.pixel_mask_min,
                    args.pixel_mask_max,
                )
            )

        if args.analysis == "threshold_area" and args.temporal_method == "daily_mean":
            # Map threshold-area over individual images, average.
            op_name = args.operator
            band_name = select_bands or "custom_index"

            def _compute_image_area(img):
                target = img.select([band_name])
                if args.mask_dataset:
                    target = _apply_ancillary_mask(
                        target,
                        args.mask_dataset,
                        args.mask_band,
                        args.mask_min,
                        args.mask_max,
                    )
                binary = getattr(target, op_name)(args.threshold)
                area = (
                    ee.Image.pixelArea()
                    .updateMask(binary)
                    .reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=region,
                        scale=args.scale,
                        maxPixels=1e9,
                    )
                    .get("area")
                )
                return img.set("_daily_area", area)

            with_areas = collection.map(_compute_image_area)
            area_list = with_areas.aggregate_array("_daily_area")
            mean_area_m2 = ee.List(area_list).reduce(ee.Reducer.mean()).getInfo() or 0.0
            results.append(
                {
                    "label": period.label,
                    "start_date": period.start_date,
                    "end_date": period.end_date,
                    "image_count": count,
                    "mean_daily_area_km2": mean_area_m2 / 1e6,
                    "mean_daily_area_m2": mean_area_m2,
                }
            )

        elif args.analysis == "threshold_area":
            composite = collection.median()
            target = _resolve_target_image(
                composite,
                args.index_name,
                args.expression,
                select_bands,
            )

            if args.mask_dataset:
                target = _apply_ancillary_mask(
                    target,
                    args.mask_dataset,
                    args.mask_band,
                    args.mask_min,
                    args.mask_max,
                )

            ops = {
                "gte": target.gte,
                "gt": target.gt,
                "lte": target.lte,
                "lt": target.lt,
                "eq": target.eq,
            }
            binary = ops[args.operator](args.threshold)
            area_m2 = (
                ee.Image.pixelArea()
                .updateMask(binary)
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=region,
                    scale=args.scale,
                    maxPixels=1e9,
                )
                .get("area")
                .getInfo()
            )
            area_m2 = area_m2 or 0.0
            results.append(
                {
                    "label": period.label,
                    "start_date": period.start_date,
                    "end_date": period.end_date,
                    "image_count": count,
                    "area_km2": area_m2 / 1e6,
                    "area_m2": area_m2,
                }
            )
        else:
            composite = collection.median()
            target = _resolve_target_image(
                composite,
                args.index_name,
                args.expression,
                select_bands,
            )

            if args.mask_dataset:
                target = _apply_ancillary_mask(
                    target,
                    args.mask_dataset,
                    args.mask_band,
                    args.mask_min,
                    args.mask_max,
                )

            # zonal_stats or index_stats
            combined = _build_reducer(args.reducers)
            stats = target.reduceRegion(
                reducer=combined,
                geometry=region,
                scale=args.scale,
                maxPixels=1e9,
            ).getInfo()
            results.append(
                {
                    "label": period.label,
                    "start_date": period.start_date,
                    "end_date": period.end_date,
                    "image_count": count,
                    "stats": stats,
                }
            )

    return {
        "analysis": args.analysis,
        "period_count": len(args.periods),
        "results": results,
    }


@mcp.tool
@_flatten_model(MultiPeriodParams)
def multi_period_analysis(
    args: MultiPeriodParams,
) -> Dict[str, Any]:
    """Run the same analysis across multiple date ranges and return
    structured results for comparison. Supports zonal_stats,
    threshold_area, and index_stats analyses.
    """
    return _multi_period_analysis(args)


@mcp.tool(
    description=(
        "Analyze a Google Earth Engine Python script and extract what aspects or issues "
        "are making scientific or data assumptions either explicitly or implicitly and might"
        "require factual verification. The extract aspects or issues would be presented to"
        "an Earth Observation expert for verification to increase the trustworthiness of the"
        "GEE Python code and its results."
        "" 
        "The response is a a list of json structures, each one describing a specific aspect"
        "identified  and containing the following fields: 'title', 'description', 'facts', "
        "and 'question_for_expert'."
        ""        
    )
)
def extract_factuality_issues(question: str, python_code: str) -> str:

    from .analysis import _extract_factuality_issues

    return _extract_factuality_issues(question=question, python_code=python_code)




@mcp.tool(
    description=(
        "Make an assessment of a factuality issue identified in a Google "
        "Earth Engine Python script that attempts to answer an Earth "
        "Observation question. The answer includes a textual assessment, "
        "as well as possible suggestions to update the GEE Python code "
        "to improve the reliability of the answer."
    )
)
async def assess_factuality_issue(
    question: str,
    python_code: str,
    issue_title: str,
    issue_description: str,
    issue_facts: str,
    issue_question_for_expert: str,
) -> str:
    from .analysis import _assess_factuality_issue

    return await _assess_factuality_issue(
        question=question,
        python_code=python_code,
        issue_title=issue_title,
        issue_description=issue_description,
        issue_facts=issue_facts,
        issue_question_for_expert=issue_question_for_expert,
    )



@mcp.tool(
    description=(
        """
        Given an earth observation question this tools determines what are the Google
        Earth Engine datasets which could be used to solve the question together with
        the time periods and spatial locations where they would be needed. This tool 
        does not make any assumption on wheather these datasets actually have data in
        such times and locations, it just determines what would be needed to answer the
        question.
        
        It also accepts a list of available Google Earth Engine as arguments so that
        its output will only be restricted to those.

        The output will be a list of json strings with fields 'dataset', 'date_periods', 'aois' where
        
        - 'dataset' is a string with the Google Earth Engine dataset anem
        - 'date_periods' is a list of dictionaries with keys 'from' and 'to'
        - 'explanation' an explanation on why this dataset is needed to answer the question
        - 'aois' is a list of dicttionaries with keys 'north-west' and 'south-east' each one with a pair (longitude, latitude)
            representing the coordinates of the aoi bounding box
        
        Note that a dataset might be required more than once, each time possibly in different locations and time periods
        
        **Example output**
        
        [ 
            {{
            'dataset': 'ESA/WorldCover/v200/2021',
            'date_periods': [ {{'from': '2023-10-01', 'to': '2023-12-31'}},
                                {{'from: '204-03-10', 'to': '2024-05-01' }} ],
            'explanation': 'this datasets provides a vegeation landcover class which is usefull to detect ....',
            'aois': [ {{'north-west': (5,40.3), 'south-east': (4.5, 41.7)}}]
        
            }}
        ]
    
        """
    )
)
def get_datasets_locations_and_periods(question: str, 
                                       gee_datasets: list[dict] = None
                                       ) -> dict:
    
    from . import analysis

    return analysis._get_datasets_locations_and_periods(question=question,
                                       gee_datasets=gee_datasets)    




@mcp.tool(
    description=(
        """
        Performs a sensitivity analysis of a GEE Python code with respect on
        how it implements an answer to an earth observation question.

        It identifies the variables and constants within the code whose values
        might impact the final result, and runs the code multiple times 
        changes those values and recording how the final result changes or not.

        It returns a markdown string with the description of the sensible 
        variables (input variables) and the output variables (results of execution
        the GEE Python code). The string also contains an image with the graphic
        representation for each input-output variable pair of the change in
        the output variable as the input variable is changed.

        """
    )
)
def sensitivity_analysis(question: str,
                          python_code: str, 
                          baseline_answer: str) -> str:
    from . import analysis

    return analysis._sensitivity_analysis(question = question,
                                          python_code = python_code,
                                          baseline_answer = baseline_answer)

@mcp.tool(
    description=(
        """
        It identifies the variables and constants within the GEE Python code 
        whose values might impact the final result when executing the code.

        It returns a list of json structures
        """
    )
)
def identify_sensible_variables(question: str,
                          python_code: str, 
                          baseline_answer: str) -> str:
    from . import analysis

    return analysis._identify_sensible_variables(question = question,
                                          python_code = python_code,
                                          baseline_answer = baseline_answer)