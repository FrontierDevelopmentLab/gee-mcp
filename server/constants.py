"""Constants shared across the GEE MCP server modules."""

import ee

MAX_PIXELS_PER_DOWNLOAD = 4e7
BASE_URL = "https://developers.google.com/earth-engine/datasets/catalog"
STAC_BASE_URL = "https://earthengine-stac.storage.googleapis.com/catalog"

# ------------------------------------------------------------------
# Spectral index definitions (default band mappings for Sentinel-2)
# ------------------------------------------------------------------
SPECTRAL_INDICES = {
    "NDVI": {
        "formula": "(NIR - RED) / (NIR + RED)",
        "bands": {"NIR": "B8", "RED": "B4"},
    },
    "NDWI": {
        "formula": "(GREEN - NIR) / (GREEN + NIR)",
        "bands": {"GREEN": "B3", "NIR": "B8"},
    },
    "EVI": {
        "formula": ("2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)"),
        "bands": {"NIR": "B8", "RED": "B4", "BLUE": "B2"},
    },
    "NBR": {
        "formula": "(NIR - SWIR2) / (NIR + SWIR2)",
        "bands": {"NIR": "B8", "SWIR2": "B12"},
    },
    "NDBI": {
        "formula": "(SWIR1 - NIR) / (SWIR1 + NIR)",
        "bands": {"SWIR1": "B11", "NIR": "B8"},
    },
    "SAVI": {
        "formula": "1.5 * (NIR - RED) / (NIR + RED + 0.5)",
        "bands": {"NIR": "B8", "RED": "B4"},
    },
}

# ------------------------------------------------------------------
# Reducer lookup (name -> ee.Reducer factory)
# ------------------------------------------------------------------
_REDUCER_MAP = {
    "mean": ee.Reducer.mean,
    "median": ee.Reducer.median,
    "min": ee.Reducer.min,
    "max": ee.Reducer.max,
    "stdDev": ee.Reducer.stdDev,
    "sum": ee.Reducer.sum,
    "count": ee.Reducer.count,
}
