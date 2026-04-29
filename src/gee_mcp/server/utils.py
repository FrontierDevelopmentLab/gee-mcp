"""Dataset metadata extraction utilities.

Two functions used by the catalogue tools:

- :func:`extract_dataset_metadata` — parses the markdown form of a GEE
  dataset page into structured fields (bands table, pixel size,
  availability, cadence).
- :func:`analyze_dataset_metadata` — uses Gemini to analyse a dataset
  description and return structured metadata.
"""

import re

import pandas as pd


def extract_dataset_metadata(t: str):
    """Extract metadata from a markdown rendering of a GEE dataset page.

    :param t: markdown content of the dataset page.
    :return: a dict with ``data`` (DataFrame), ``pixel_size``,
        ``availability_start_date``, ``availability_end_date``, and
        ``cadence``.
    """
    i = 0
    r = []
    ti = t[:]

    while True:
        i = ti.find("\n#")
        if i == -1:
            break
        name = ti[i : i + ti[i + 1 :].find("\n") + 1]
        ti = ti[i + len(name) + 1 :][:]
        position = t.find(ti) - len(name)
        level = name.count("#")
        name = name.replace("\n", "").replace("#", "").strip()
        r.append(
            {"section": name, "header_level": level, "position": position}
        )

    for i in range(len(r)):
        a = r[i]["position"]
        b = r[i + 1]["position"] if i < len(r) - 1 else -1
        r[i]["content"] = t[a:b]

    r = pd.DataFrame(r)

    pixel_size = None
    if "bands" in r.section.str.lower().values:
        ri = r[r.section.str.lower() == "bands"].iloc[0]
        m = re.search(
            r"\*\*pixel size\*\*(.*?)\*\*bands",
            ri.content.lower(),
            flags=re.DOTALL,
        )
        if m:
            pixel_size = m.group(1).strip()

    availability_start_date, availability_end_date = None, None
    if "page summary" in r.section.str.lower().values:
        ri = r[r.section.str.lower() == "page summary"].iloc[0]
        m = re.search(
            r"dataset availability(\s+):(.*?)\n\n",
            ri.content.lower(),
            flags=re.DOTALL,
        )
        if m:
            dataset_availability = " ".join(m.groups()).strip()
            (
                availability_start_date,
                availability_end_date,
            ) = dataset_availability.split("–")

    cadence = None
    if "page summary" in r.section.str.lower().values:
        ri = r[r.section.str.lower() == "page summary"].iloc[0]
        m = re.search(
            r"\ncadence(\s+):(.*?)\n\n", ri.content.lower(), flags=re.DOTALL
        )
        if m:
            cadence = " ".join(m.groups()).strip()

    return {
        "data": r,
        "pixel_size": pixel_size,
        "availability_start_date": availability_start_date,
        "availability_end_date": availability_end_date,
        "cadence": cadence,
    }


def analyze_dataset_metadata(genai, t):
    """Use a Gemini client to extract structured metadata from a dataset page.

    :param genai: a Gemini-style client exposing ``call(prompt)``.
    :param t: raw text or markdown description of the dataset.
    :return: the model response (typically a dict with ``answer``).
    """
    prompt = f"""

    You are an expert extractor of information from earth observation metadata.
    This is the description of an earth observation dataset in Google Earth Engine.

    <DATASET_DESCRIPTION>
    {t}
    </DATASET_DESCRIPTION>

    Your tasks are:
    - the dataset id
    - compile a detailed description of the dataset
    - extract the availability start date and end date of the dataset
    - extract the pixel size of the different bands of the dataset
    - extract the cadence or time resolution of the dataset
    - extract a list of bands with all the data associated
    - infer a list of possible applications for this dataset.

    Take into consideration the following aspects:

    - some datasets have a single, global pixel size, whilst others have different pixel
    sizes for different bands. In either case extract a list of all pixel sizes
    present in the dataset.

    - bands usually come described in a table with different fields like band name,
    pixel size, description, value ranges, etc. Extract them all.

    - sometimes cadence might not be explicitly stated in the dataset. If this is the case
    try to infer it from the full description.

    Provide your answer as a json structure such as this one:


    {{
    "dataset_id": "COPERNICUS_S1_GRD",
    "description": "this dataset contains ....",
    "applications": "this dataset could be used to ....",
    "availability_start": "2020-01-01T00:00:00Z",
    "availability_end": "2026-01-31T06:00:00Z",
    "pixel_size": [ "10 meters", "60 meters" ],
    "cadence": "6 hours",
    "bands": [ {{"name": "B1", "description": "coastal aerosol", "pixel_size": "10 meters", "wavelength": " 0.43 - 0.45 μm"}},
                {{"name": "B2", "description": "green", "pixel_size": "60 meters", "wavelength": " 0.46 - 0.50 μm"}}
    ]
    }}

    """
    return genai.call(prompt)
