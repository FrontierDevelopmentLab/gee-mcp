"""Unified FastMCP server for GEE.

Registers all tools under a single ``FastMCP("gee-mcp")`` instance.

Tools
-----
- ``download_satellite_image``
- ``compute_index``
- ``zonal_statistics``
- ``temporal_composite``
- ``mask_by_raster``
- ``threshold_area``
- ``multi_period_analysis``
- ``list_datasets``
- ``get_dataset_info``
- ``get_dataset_metadata``
- ``check_imagery_availability``
- ``extract_metadata``
- ``analyze_metadata``
- ``generate_python_from_question``
- ``generate_python_from_reasoning_steps``
- ``generate_python_from_abstract_graph``
- ``generate_abstract_graph_from_question``
- ``extract_factuality_issues``
- ``assess_factuality_issue``
- ``get_datasets_locations_and_periods``
- ``identify_sensible_variables``
- ``sensitivity_analysis``
- ``execute_gee_python``

Prompts
-------
- ``get_llm_prompt``
- ``get_metadata_prompt``

Run standalone::

    python -m gee_mcp.server
"""

from .auth import setup_gee

# ------------------------------------------------------------------
# GEE initialisation (must happen before submodule imports that
# reference ``ee`` at module level via constants.py / helpers.py).
# ------------------------------------------------------------------
setup_gee()



# ------------------------------------------------------------------
# Re-export everything from submodules for convenience.
# ------------------------------------------------------------------

from .app import mcp  # noqa: E402
from .constants import (  # noqa: E402
    _REDUCER_MAP,
    BASE_URL,
    MAX_PIXELS_PER_DOWNLOAD,
    SPECTRAL_INDICES,
    STAC_BASE_URL,
)
from .helpers import (  # noqa: E402
    _apply_ancillary_mask,
    _apply_pixel_mask,
    _build_collection,
    _build_reducer,
    _build_region,
    _check_and_split_region,
    _download_with_fallback,
    _fetch_stac_json,
    _flatten_model,
    _resolve_target_image,
    _split_region_quadrants,
    _stac_cache,
)
from .models import (  # noqa: E402
    ComputeIndexParams,
    DateRange,
    DownloadParams,
    MaskByRasterParams,
    MultiPeriodParams,
    RegionParams,
    TemporalCompositeParams,
    ThresholdAreaParams,
    ZonalStatsParams,
)

# Importing the tool modules triggers @mcp.tool registration.
from .tools_analysis import (  # noqa: E402, F401
    _compute_index,
    _mask_by_raster,
    _multi_period_analysis,
    _temporal_composite,
    _threshold_area,
    _zonal_statistics,
    compute_index,
    download_satellite_image,
    get_datasets_locations_and_periods,
    identify_sensible_variables,
    mask_by_raster,
    multi_period_analysis,
    sensitivity_analysis,
    temporal_composite,
    threshold_area,
    zonal_statistics,
    extract_factuality_issues,
    assess_factuality_issue,
)
from .analysis import (  # noqa: E402, F401
    _extract_factuality_issues,
    _get_datasets_locations_and_periods,
    _identify_sensible_variables,
    _sensitivity_analysis,
)
from .tools_catalogue import (  # noqa: E402, F401
    _get_dataset_info,
    analyze_metadata,
    check_imagery_availability,
    extract_metadata,
    get_dataset_info,
    get_dataset_metadata,
    get_llm_prompt,
    get_metadata_prompt,
    list_datasets,
)

from .tools_execution import (  # noqa: E402, F401
    generate_python_from_question,
    generate_python_from_reasoning_steps,
    generate_python_from_abstract_graph,
    generate_abstract_graph_from_question,
    execute_gee_python
)

from .codegen import (  # noqa: E402, F401
    _generate_python_from_question,
    _generate_python_from_reasoning_steps,
    _generate_python_from_abstract_graph,
    _generate_abstract_graph_from_question,    
)

from .coderun import (
    _execute_gee_python
)

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
