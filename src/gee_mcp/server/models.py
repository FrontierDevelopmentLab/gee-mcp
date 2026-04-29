"""Pydantic models for the GEE MCP server tools."""

from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator

from .constants import SPECTRAL_INDICES


class RegionParams(BaseModel):
    """Base model for parameters that define a region of interest."""

    latitude: Optional[float] = Field(
        None, description="Latitude of the point of interest."
    )
    longitude: Optional[float] = Field(
        None, description="Longitude of the point of interest."
    )
    region_geojson: Optional[str] = Field(
        None,
        description=("A GeoJSON string defining the region of interest."),
    )
    bounding_box: Optional[List[float]] = Field(
        None,
        description=(
            "A bounding box as a list of four coordinates: "
            "[min_lon, min_lat, max_lon, max_lat]."
        ),
    )
    buffer_m: int = Field(
        1000,
        description=(
            "Buffer distance in meters around the point "
            "(only used with latitude/longitude)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def check_coords_or_geojson(cls, data: Any) -> Any:
        """Ensure exactly one location type is provided."""
        if not isinstance(data, dict):
            return data

        lat, lon = data.get("latitude"), data.get("longitude")
        geojson = data.get("region_geojson")
        bbox = data.get("bounding_box")
        coords_provided = lat is not None and lon is not None
        geojson_provided = geojson is not None
        bbox_provided = bbox is not None

        provided_options = sum(
            [coords_provided, geojson_provided, bbox_provided]
        )

        if provided_options == 0:
            raise ValueError(
                "Either latitude/longitude, a region_geojson string, "
                "or a bounding_box must be provided."
            )
        if provided_options > 1:
            raise ValueError(
                "Provide only one of latitude/longitude, "
                "a region_geojson string, or a bounding_box."
            )

        if bbox_provided and len(bbox) != 4:
            raise ValueError(
                "The bounding_box must contain exactly four "
                "coordinates: "
                "[min_lon, min_lat, max_lon, max_lat]."
            )

        return data


class DownloadParams(RegionParams):
    """Parameters for downloading satellite imagery."""

    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    scale: int = Field(10, description="Pixel resolution in meters")
    bands: Optional[str] = Field(
        "B4,B3,B2",
        description=(
            "Comma-separated band names to download, " "e.g., B4,B3,B2"
        ),
    )
    dataset: str = Field(
        "COPERNICUS/S2_SR_HARMONIZED",
        description=(
            "ImageCollection ID, e.g., COPERNICUS/S2_SR_HARMONIZED "
            "or LANDSAT/LC08/C02/T1_L2"
        ),
    )
    max_cloud_cover: Optional[float] = Field(
        20.0, description="Maximum cloud cover percentage (0-100)."
    )
    cloud_filter_property: Optional[str] = Field(
        None,
        description=(
            "Custom cloud cover metadata property name. "
            "Auto-detected for Sentinel-2 and Landsat if omitted."
        ),
    )
    image_count: Optional[int] = Field(
        1, description="Number of least cloudy images to download."
    )


class ComputeIndexParams(RegionParams):
    """Parameters for computing a spectral index."""

    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    dataset: str = Field(
        "COPERNICUS/S2_SR_HARMONIZED",
        description="ImageCollection ID.",
    )
    index_name: Optional[str] = Field(
        None,
        description=(
            "Well-known spectral index: " "NDVI, NDWI, EVI, NBR, NDBI, SAVI."
        ),
    )
    expression: Optional[str] = Field(
        None,
        description=(
            "Custom band math expression, " 'e.g. "(B8 - B4) / (B8 + B4)".'
        ),
    )
    scale: int = Field(10, description="Pixel resolution in metres.")
    max_cloud_cover: Optional[float] = Field(
        20.0, description="Maximum cloud cover percentage (0-100)."
    )
    cloud_filter_property: Optional[str] = Field(
        None,
        description=(
            "Custom cloud cover metadata property name. "
            "Auto-detected for Sentinel-2 and Landsat if omitted."
        ),
    )
    download: bool = Field(
        False,
        description="If true, return a GeoTIFF download link.",
    )

    @model_validator(mode="before")
    @classmethod
    def check_index_or_expression(cls, data: Any) -> Any:
        """Ensure exactly one of index_name or expression."""
        # Run the parent validator first
        data = super(
            ComputeIndexParams, ComputeIndexParams
        ).check_coords_or_geojson(data)
        if not isinstance(data, dict):
            return data
        index_name = data.get("index_name")
        expression = data.get("expression")
        has_index = index_name is not None
        has_expr = expression is not None
        if not has_index and not has_expr:
            raise ValueError(
                "Exactly one of index_name or expression " "must be provided."
            )
        if has_index and has_expr:
            raise ValueError(
                "Provide only one of index_name or expression, " "not both."
            )
        if has_index and index_name.upper() not in SPECTRAL_INDICES:
            raise ValueError(
                f"Unknown index_name '{index_name}'. "
                f"Supported: "
                f"{', '.join(SPECTRAL_INDICES.keys())}."
            )
        return data


class ZonalStatsParams(RegionParams):
    """Parameters for computing zonal statistics."""

    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    dataset: str = Field(
        "COPERNICUS/S2_SR_HARMONIZED",
        description="ImageCollection ID.",
    )
    bands: Optional[str] = Field(
        None,
        description=(
            "Comma-separated bands to compute stats for. "
            "If None, computes for all bands."
        ),
    )
    index_name: Optional[str] = Field(
        None,
        description=(
            "Compute stats for a spectral index " "instead of raw bands."
        ),
    )
    scale: int = Field(10, description="Pixel resolution in metres.")
    max_cloud_cover: Optional[float] = Field(
        20.0, description="Maximum cloud cover percentage (0-100)."
    )
    cloud_filter_property: Optional[str] = Field(
        None,
        description=(
            "Custom cloud cover metadata property name. "
            "Auto-detected for Sentinel-2 and Landsat if omitted."
        ),
    )
    reducers: str = Field(
        "mean,median,min,max,stdDev",
        description="Comma-separated list of reducers.",
    )
    download: bool = Field(
        False,
        description=(
            "If true, include GeoTIFF download link " "for the composite."
        ),
    )


class TemporalCompositeParams(RegionParams):
    """Parameters for creating a temporal composite."""

    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    dataset: str = Field(
        "COPERNICUS/S2_SR_HARMONIZED",
        description="ImageCollection ID.",
    )
    method: str = Field(
        "median",
        description=(
            "Composite method: " "median, mosaic, greenest, most_recent."
        ),
    )
    bands: Optional[str] = Field(
        "B4,B3,B2",
        description="Comma-separated bands to include.",
    )
    scale: int = Field(10, description="Pixel resolution in metres.")
    max_cloud_cover: Optional[float] = Field(
        20.0, description="Maximum cloud cover percentage (0-100)."
    )
    cloud_filter_property: Optional[str] = Field(
        None,
        description=(
            "Custom cloud cover metadata property name. "
            "Auto-detected for Sentinel-2 and Landsat if omitted."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def check_method(cls, data: Any) -> Any:
        """Validate composite method."""
        data = super(
            TemporalCompositeParams, TemporalCompositeParams
        ).check_coords_or_geojson(data)
        if not isinstance(data, dict):
            return data
        method = data.get("method", "median")
        valid = {"median", "mosaic", "greenest", "most_recent"}
        if method not in valid:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: "
                f"{', '.join(sorted(valid))}."
            )
        return data


class MaskByRasterParams(RegionParams):
    """Parameters for masking imagery by an ancillary raster."""

    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    dataset: str = Field(
        "COPERNICUS/S2_SR_HARMONIZED",
        description="ImageCollection ID.",
    )
    mask_dataset: str = Field(
        ...,
        description=("Ancillary dataset ID, e.g. USGS/SRTMGL1_003."),
    )
    mask_band: str = Field(..., description="Band to mask on, e.g. elevation.")
    mask_min: Optional[float] = Field(
        None,
        description="Inclusive lower bound for the mask range.",
    )
    mask_max: Optional[float] = Field(
        None,
        description="Inclusive upper bound for the mask range.",
    )
    bands: Optional[str] = Field(
        None, description="Comma-separated bands to analyse."
    )
    index_name: Optional[str] = Field(
        None, description="Spectral index to compute."
    )
    expression: Optional[str] = Field(
        None, description="Custom band math expression."
    )
    scale: int = Field(10, description="Pixel resolution in metres.")
    max_cloud_cover: Optional[float] = Field(
        20.0, description="Maximum cloud cover percentage (0-100)."
    )
    cloud_filter_property: Optional[str] = Field(
        None,
        description=(
            "Custom cloud cover metadata property name. "
            "Auto-detected for Sentinel-2 and Landsat if omitted."
        ),
    )
    reducers: str = Field(
        "mean,median,min,max,stdDev",
        description="Comma-separated list of reducers.",
    )
    download: bool = Field(
        False,
        description="If true, include GeoTIFF download link.",
    )

    @model_validator(mode="before")
    @classmethod
    def check_mask_range(cls, data: Any) -> Any:
        """Ensure at least one of mask_min/mask_max is provided."""
        data = super(
            MaskByRasterParams, MaskByRasterParams
        ).check_coords_or_geojson(data)
        if not isinstance(data, dict):
            return data
        if data.get("mask_min") is None and data.get("mask_max") is None:
            raise ValueError(
                "At least one of mask_min or mask_max " "must be provided."
            )
        return data


class ThresholdAreaParams(RegionParams):
    """Parameters for computing area above/below a threshold."""

    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    dataset: str = Field(
        "COPERNICUS/S2_SR_HARMONIZED",
        description="ImageCollection ID.",
    )
    band: Optional[str] = Field(None, description="Single band name.")
    index_name: Optional[str] = Field(
        None, description="Spectral index to compute."
    )
    expression: Optional[str] = Field(
        None, description="Custom band math expression."
    )
    threshold: float = Field(..., description="Threshold value.")
    operator: str = Field(
        "gte",
        description="Comparison operator: gte, gt, lte, lt, eq.",
    )
    mask_dataset: Optional[str] = Field(
        None, description="Optional ancillary mask dataset."
    )
    mask_band: Optional[str] = Field(
        None, description="Band for ancillary mask."
    )
    mask_min: Optional[float] = Field(
        None, description="Ancillary mask lower bound."
    )
    mask_max: Optional[float] = Field(
        None, description="Ancillary mask upper bound."
    )
    scale: int = Field(10, description="Pixel resolution in metres.")
    max_cloud_cover: Optional[float] = Field(
        20.0, description="Maximum cloud cover percentage (0-100)."
    )
    cloud_filter_property: Optional[str] = Field(
        None,
        description=(
            "Custom cloud cover metadata property name. "
            "Auto-detected for Sentinel-2 and Landsat if omitted."
        ),
    )
    pixel_mask_band: Optional[str] = Field(
        None,
        description=(
            "Band for per-pixel QA masking (e.g. mask invalid "
            "values before compositing)."
        ),
    )
    pixel_mask_min: Optional[float] = Field(
        None, description="Minimum valid pixel value (inclusive)."
    )
    pixel_mask_max: Optional[float] = Field(
        None, description="Maximum valid pixel value (inclusive)."
    )

    @model_validator(mode="before")
    @classmethod
    def check_target_and_operator(cls, data: Any) -> Any:
        """Validate exactly one target and a valid operator."""
        data = super(
            ThresholdAreaParams, ThresholdAreaParams
        ).check_coords_or_geojson(data)
        if not isinstance(data, dict):
            return data

        has_band = data.get("band") is not None
        has_index = data.get("index_name") is not None
        has_expr = data.get("expression") is not None
        provided = sum([has_band, has_index, has_expr])
        if provided != 1:
            raise ValueError(
                "Exactly one of band, index_name, or expression "
                "must be provided."
            )

        op = data.get("operator", "gte")
        valid_ops = {"gte", "gt", "lte", "lt", "eq"}
        if op not in valid_ops:
            raise ValueError(
                f"Invalid operator '{op}'. "
                f"Must be one of: {', '.join(sorted(valid_ops))}."
            )
        return data


class DateRange(BaseModel):
    """A labelled date range for multi-period analysis."""

    label: str = Field(..., description="Label for this period.")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class MultiPeriodParams(RegionParams):
    """Parameters for running analysis across date ranges."""

    periods: List[DateRange] = Field(
        ..., description="Two or more labelled date ranges."
    )
    dataset: str = Field(
        "COPERNICUS/S2_SR_HARMONIZED",
        description="ImageCollection ID.",
    )
    analysis: str = Field(
        "zonal_stats",
        description=(
            "Analysis type: " "zonal_stats, threshold_area, index_stats."
        ),
    )
    band: Optional[str] = Field(
        None,
        description=("Single band name (threshold_area analysis)."),
    )
    bands: Optional[str] = Field(
        None, description="Bands for zonal_stats analysis."
    )
    index_name: Optional[str] = Field(
        None, description="Spectral index to compute."
    )
    expression: Optional[str] = Field(
        None, description="Custom band math expression."
    )
    threshold: Optional[float] = Field(
        None,
        description="Required if analysis=threshold_area.",
    )
    operator: str = Field(
        "gte",
        description="Threshold operator: gte, gt, lte, lt, eq.",
    )
    reducers: str = Field(
        "mean,median,min,max",
        description="Reducers for stats analyses.",
    )
    mask_dataset: Optional[str] = Field(
        None, description="Optional ancillary mask dataset."
    )
    mask_band: Optional[str] = Field(
        None, description="Band for ancillary mask."
    )
    mask_min: Optional[float] = Field(
        None, description="Ancillary mask lower bound."
    )
    mask_max: Optional[float] = Field(
        None, description="Ancillary mask upper bound."
    )
    scale: int = Field(10, description="Pixel resolution in metres.")
    max_cloud_cover: Optional[float] = Field(
        20.0, description="Maximum cloud cover percentage (0-100)."
    )
    cloud_filter_property: Optional[str] = Field(
        None,
        description=(
            "Custom cloud cover metadata property name. "
            "Auto-detected for Sentinel-2 and Landsat if omitted."
        ),
    )
    pixel_mask_band: Optional[str] = Field(
        None,
        description=(
            "Band for per-pixel QA masking (e.g. mask invalid "
            "values before compositing)."
        ),
    )
    pixel_mask_min: Optional[float] = Field(
        None, description="Minimum valid pixel value (inclusive)."
    )
    pixel_mask_max: Optional[float] = Field(
        None, description="Maximum valid pixel value (inclusive)."
    )
    temporal_method: str = Field(
        "composite",
        description=(
            "Temporal aggregation: 'composite' (median then "
            "analyse) or 'daily_mean' (analyse each image, "
            "average results). "
            "Only affects threshold_area analysis."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def check_periods_and_analysis(cls, data: Any) -> Any:
        """Validate periods count and threshold requirement."""
        data = super(
            MultiPeriodParams, MultiPeriodParams
        ).check_coords_or_geojson(data)
        if not isinstance(data, dict):
            return data

        periods = data.get("periods", [])
        if len(periods) < 2:
            raise ValueError("At least two periods must be provided.")

        analysis = data.get("analysis", "zonal_stats")
        valid_analyses = {
            "zonal_stats",
            "threshold_area",
            "index_stats",
        }
        if analysis not in valid_analyses:
            raise ValueError(
                f"Invalid analysis '{analysis}'. "
                f"Must be one of: "
                f"{', '.join(sorted(valid_analyses))}."
            )

        if analysis == "threshold_area" and data.get("threshold") is None:
            raise ValueError(
                "threshold is required when analysis is " "'threshold_area'."
            )

        return data
