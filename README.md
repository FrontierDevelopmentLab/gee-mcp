# GEE MCP Server

This is a standalone Model Context Protocol (MCP) server for Google Earth Engine (GEE). It provides tools for dataset discovery, metadata extraction, and executing GEE Python code.

## Project Structure

- `server/`: The MCP server package.
- `client.py`: An example MCP client to test the server.
- `requirements.txt`: List of dependencies.

## Available Tools

The server exposes several Model Context Protocol (MCP) tools categorised by their function:

### Catalogue & Metadata
- `list_datasets`: List all available Google Earth Engine datasets.
- `get_dataset_info`: Get detailed Markdown information about a specific GEE dataset.
- `get_dataset_metadata`: Get structured STAC metadata (bands, temporal interval, etc.) for a dataset.
- `check_imagery_availability`: Check imagery availability for a dataset within a date range and optional bounding box.
- `extract_metadata`: Extract structured metadata (bands, pixel size, availability, cadence) from a dataset page.
- `analyze_metadata`: Use Gemini AI to analyse a dataset description and extract structured metadata.

### Analysis & Data Processing
- `download_satellite_image`: Download satellite images from GEE.
- `compute_index`: Compute a spectral index (e.g., NDVI, NDWI) or custom band math expression over a region.
- `zonal_statistics`: Compute summary statistics (mean, median, min, etc.) for bands or an index within a region.
- `temporal_composite`: Create cloud-free temporal composites (median, mosaic, greenest, etc.).
- `mask_by_raster`: Apply a value-range mask (e.g., DEM, land cover) to imagery and compute statistics.
- `threshold_area`: Compute the area of pixels meeting a threshold condition on a band, index, or expression.
- `multi_period_analysis`: Run the same analysis across multiple date ranges for temporal comparisons.
- `execute_gee_python`: Execute a provided GEE Python script and return the result.

### AI Code Generation & Validation
- `generate_python_from_question`: Answer an Earth Observation question by generating GEE Python code with iterative error fixing.
- `generate_abstract_graph_from_question`: Generate an abstract graph (Mermaid) describing an EO pipeline to solve a question.
- `generate_python_from_reasoning_steps`: Generate GEE Python code based on a provided set of reasoning steps.
- `generate_python_from_abstract_graph`: Generate GEE Python code based on a provided Mermaid graph.
- `get_datasets_locations_and_periods`: Determine the GEE datasets, time periods, and AOIs required to answer a given question.
- `extract_factuality_issues`: Analyze GEE Python script and extract data/scientific assumptions that might require factual verification.
- `assess_factuality_issue`: Assess one of the factuality issies extracted in the previous function call, and generate recommendations and code change suggestions.
- `identify_sensible_variables`: Identify variables and constants in the GEE Python code whose values might impact the final result.
- `sensitivity_analysis`: Perform sensitivity analysis by tweaking variable values in the code and plotting the impacts on the final result.

## Example Tool Invocation

Here is an example of how an MCP client might format a JSON-RPC request to invoke the `generate_python_from_question` tool:

```json
{
  "method": "tools/call",
  "params": {
    "name": "generate_python_from_question",
    "arguments": {
      "question": "Calculate the average NDVI over the Amazon basin for the year 2023.",
      "fix_code": true
    }
  }
}
```

It returns a json structure with several objects, most notably 

- `python_code`: the actual GEE Python code generated
- `python_code_explanation`: an explanation of the code generated
- `python_code_fix_history`: the iterative fixes made to the code
- `python_code_result`: the result after executing the code

This is the code generated

```python
import ee
import geemap

def gee_main():
    # Define a point within the Amazon basin (near Manaus, Brazil) to intersect the basin polygon
    amazon_point = ee.Geometry.Point([-60.0, -3.0])
    
    # Load the WWF HydroATLAS Level 3 Basins dataset
    # We use level 3 which contains the major continental basins like the Amazon
    basins = ee.FeatureCollection('WWF/HydroATLAS/v1/Basins/level03')
    amazon_basin = basins.filterBounds(amazon_point)
    
    # Load MODIS monthly NDVI data for the year 2023
    modis_ndvi = ee.ImageCollection('MODIS/061/MOD13A3') \
        .filterDate('2023-01-01', '2024-01-01') \
        .select('NDVI')
    
    # Calculate the mean NDVI image over the year and apply the scaling factor of 0.0001
    mean_ndvi_image = modis_ndvi.mean().multiply(0.0001)
    
    # Calculate the average NDVI over the entire Amazon basin
    stats = mean_ndvi_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=amazon_basin.geometry(),
        scale=1000, # Matches the 1km resolution of MOD13A3
        maxPixels=1e12
    )
    
    # Extract the average NDVI value
    avg_ndvi = stats.get('NDVI').getInfo()
    
    # Initialize a geemap Map for visualization
    Map = geemap.Map()
    Map.centerObject(amazon_basin, 4)
    
    # Define NDVI visualization parameters
    ndvi_vis = {
        'min': 0.0,
        'max': 1.0,
        'palette': [
            'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
            '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
            '012E01', '011D01', '011301'
        ]
    }
    
    # Add the layers to the map
    Map.addLayer(amazon_basin.style(**{'fillColor': '00000000', 'color': 'FF0000'}), {}, 'Amazon Basin Boundary')
    Map.addLayer(mean_ndvi_image.clip(amazon_basin), ndvi_vis, 'Mean NDVI 2023')
    
    # Format the result as an XML string
    result_xml = f"""<RESULT>
<VARIABLE_NAME>average_ndvi</VARIABLE_NAME>
<VALUE>{avg_ndvi}</VALUE>
<UNITS>dimensionless</UNITS>
</RESULT>"""
    
    return result_xml, Map
  ```

and this is the result

```xml
<RESULT>
<VARIABLE_NAME>average_ndvi</VARIABLE_NAME>
<VALUE>0.7712281332887203</VALUE>
<UNITS>dimensionless</UNITS>
</RESULT>
```

Observe that the generated code also returns a `Map` object so that it can be displayed in appropriate environments, such as IPython Notebookes. For instance, for this question

    Characterize the morphometry and land cover of the Emme catchment in the Canton of Bern by determining its total area, maximum elevation, and forest cover percentage. Additionally, state the financial magnitude of the damages caused by the flash flood event in this catchment (specifically in Schangnau) during July 2014.

the generated code returns the following answer and map:

![map](imgs/gee-execution.png) 

## Integration

As any other MCP server GEE MCP can be integrated in any agentic tool. For instance in Gemini-CLI

![geminicli](imgs/geminicli.png) 


## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have authenticated with Google Earth Engine and your environment is set up.

## Environment Variables

You need to set up access to Gemini (via an api key or a vertex ai project) and to Google Earth Engine.

### Access to Gemini

You must set up either

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`: for the Gemini API key

or, for a Vertex AI project

- `VERTEXAI_PROJECT` (optional): The GCP project ID for Vertex AI (used if no API key is provided).
- `VERTEXAI_LOCATION` (optional): The GCP region for Vertex AI (defaults to `"global"`).

assuming you have authenticated within that project via

```bash
gcloud auth application-default login
```

### Access to Google Earth Engine

You must set

- `GEE_PROJECT`: The GEE project id (required by `auth.py`).


having previously authenticated via

```bash
earthengine authenticate
```

## Running the Server

You can run the server via stdio (as a module):

```bash
python -m server
```

## Testing with Client

You can run the example client to test connection and tools:

```bash
python client.py
```
