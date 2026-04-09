import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from .helpers import extract_xml_tag, extract_tag
import ee
from loguru import logger


def extract_dataset_metadata(t: str):
    """
    Extracts metadata from a markdown string representation of a GEE dataset page.

    Args:
        t (str): The markdown content of the dataset page.

    Returns:
        dict: A dictionary containing the extracted metadata, including
              the following keys: 'data' (a dataframe), 'pixel_size', 'availability_start_date', 'availability_end_date', and 'cadence'.
    """

    i = 0
    r = []
    ti = t[:]

    # Parse the markdown to identify sections based on headers
    while True:
        i = ti.find("\n#")
        if i == -1:
            break
        name = ti[i : i + ti[i + 1 :].find("\n") + 1]
        ti = ti[i + len(name) + 1 :][:]

        # Calculate the position of the section in the original text
        position = t.find(ti) - len(name)

        level = name.count("#")
        name = name.replace("\n", "").replace("#", "").strip()

        r.append(
            {"section": name, "header_level": level, "position": position}
        )

    # Extract content for each section
    for i in range(len(r)):
        a = r[i]["position"]
        b = r[i + 1]["position"] if i < len(r) - 1 else -1
        r[i]["content"] = t[a:b]

    r = pd.DataFrame(r)

    # extract pixel size
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

    # extract availability dates
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
            availability_start_date, availability_end_date = (
                dataset_availability.split("–")
            )

    # extract cadence
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
    """
    Uses a Generative AI model to analyze and extract structured metadata from a dataset description.

    This function constructs a prompt containing the dataset description and instructions
    for the AI to extract specific fields like ID, description, availability, pixel size,
    cadence, bands, and potential applications.

    Args:
        genai: The Generative AI client object used to generate the response.
        t (str): The raw text or markdown description of the dataset.

    Returns:
        str: A JSON-formatted string containing the extracted metadata.
    """

    # Construct the prompt with instructions for the model to extract metadata
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

    Provide your answer as a json structre such as this one:


    {{
    "dataset_id": "COPERNICUS_S1_GRD",
    "description": "this dataset contains ....",
    "applications": this dataset could be used to ....",
    "availability_start": "2020-01-01T00:00:00Z",
    "availability_end": "2026-01-31T06:00:00Z",
    "pixel_size": [ "10 meters", "60 meters" ],
    "cadence": "6 hours"
    "bands": [ {{"name": "B1", "description": "coastal aerosol", "pixel_size": "10 meters", "wavelength": " 0.43 - 0.45 μm"}},
                {{"name": "B2", "description": "green", "pixel_size": "60 meters", "wavelength": " 0.46 - 0.50 μm"}}
    }}

    """

    # Call the AI model to generate the response
    response = genai.call(prompt)
    return response


def timestamp_to_datetime(ts: int) -> datetime:
    """
    Converts a unix timestamp in seconds, milliseconds, microseconds, or nanoseconds
      to a datetime object. Assumes timezone is UTC.
    """

    if ts > 1e18:
        return datetime.fromtimestamp(ts / 1e9, tz=timezone.utc)  # nanoseconds
    elif ts > 1e15:
        return datetime.fromtimestamp(
            ts / 1e6, tz=timezone.utc
        )  # microseconds
    elif ts > 1e12:
        return datetime.fromtimestamp(
            ts / 1e3, tz=timezone.utc
        )  # milliseconds
    else:
        return datetime.fromtimestamp(ts, tz=timezone.utc)  # seconds




def distribute_points(a, b, x, n):
    """
    Distributes n points within (a, b) given an existing point x
    to maximize the minimum distance between all points.
    """
    if not (a < x < b):
        raise ValueError("x must be within the interval (a, b)")

    # Calculate the total length and the proportion of n points for each side
    total_length = b - a
    left_ratio = (x - a) / total_length
    
    # Estimate how many points should go to the left of x
    n_left = round(left_ratio * n)
    n_right = n - n_left

    # Generate points for the left segment (a, x)
    # We use n_left + 1 spaces
    left_points = [a + (i + 1) * (x - a) / (n_left + 1) for i in range(n_left)]
    
    # Generate points for the right segment (x, b)
    # We use n_right + 1 spaces
    right_points = [x + (i + 1) * (b - x) / (n_right + 1) for i in range(n_right)]

    # Combine and return sorted list
    return sorted(left_points + right_points)

def extract_standardized_numeric_answer(text):
    
    ex = []
    
    pattern = r"<RESULT>(.*?)</RESULT>"
    
    results = re.findall(pattern, text, flags=re.DOTALL)
    
    # Clean up and print
    for i, match in enumerate(results, 1):
        ss = match.strip()
        value = extract_xml_tag(ss, 'VALUE')

        # attempt to convert to float
        try:
            value = float(value)
        except:
            pass
        ex.append({
            'answer_field_name': extract_xml_tag(ss, 'VARIABLE_NAME'),
            'answer_field_value': value,
            'answer_field_units': extract_xml_tag(ss, 'UNITS'),
            
        })

    return ex

def extract_numeric_answer(genai_client, question, answer):
    prompt = f"""
    You are an expert in Earth Observation. You are given an Earth Observation question
    and a verbose answer to it.
    
    Your task is to extract the fields and their numeric values from the answer that are
    relevant to answering the question.
    
    Your answer should be a list of json structures each one having two entries 'answer_field_name'
    and 'answer_field_value', such as in this example output
    
    [{{'answer_field_name': 'ndvi difference', 'answer_field_value': -0.4012}} ]
    
    This is the question
    <QUESTION>
    {question}
    </QUESTION>
    
    And this is the answer
    <ANSWER>
    {answer}
    </ANSWER>
    
    """
    r = genai_client.call(prompt)

    return json.loads(extract_tag(r['answer'], 'json'))


def extract_field_meaning(genai_client, question, answer):
    prompt = f"""
    You are an expert in Earth Observation. You are given an Earth Observation question
    and a verbose answer to it.
    
    Your task is to extract the fields and their meanings that are
    relevant to answering the question.
    
    Your answer should be a list of json structures each one having two entries 'answer_field_name'
    and 'answer_field_meaning', such as in this example output
    
    [{{'answer_field_name': 'ndvi difference', 'answer_field_meaning': 'The difference in NDVI values between two time periods'}} ]
    
    This is the question
    <QUESTION>
    {question}
    </QUESTION>
    
    And this is the answer
    <ANSWER>
    {answer}
    </ANSWER>
    
    """
    r = genai_client.call(prompt)

    return json.loads(extract_tag(r['answer'], 'json'))


class ParameterSensitivity:
    """
    Sensitivity analysis of Google Earth Engine Python code for Earth Observation questions.

    First it identifies the variables and constants in the code that are most likely to affect the final result,
    then it changes the value of each variable to a value for sensitivity analysis, executes the modified code,

    The change in value for sensitivity analysis is computed as the original value plus two thirds of the distance 
    to the maximum value.
    
    We compute the linear range impact in output variable as the relative change in output variable 
    divided by the relative change in value of the code variable, when changing the code variable 
    from its original value to the value for sensitivity analysis.

    See doc in run method below.

    :var guidelines: Description
    """


    def __init__(self, genai_client, gee_client, question_record, n_samples_per_code_variable=3):

        self.genai_client = genai_client
        self.question_record = question_record
        self.gee_client = gee_client

        q = self.question_record
        self.question    = q["question"]
        self.python_code = q['python_code']
        self.baseline_answer = q['python_code_result']
        self.input_variables = None
        self.output_variables = None
        self.sensitivity_results = None
        self.n_samples_per_code_variable = n_samples_per_code_variable


    def gee_identify_sensible_variables(self):
        """
        identifies the variables and constants in the code that are most likely to affect the final result, 
        by calling genai with a prompt that includes the code and the question, and asking for a list of 
        variables and constants with their value range and an estimate of their impact on the final result.

        Then, for each variable, it generates a set of values for sensitivity analysis by distributing n 
        samples within the value range of the variable, given the original value of the variable, 
        to maximize the minimum distance between all points. The number of points n is defined in 
        self.n_samples_per_code_variable.
        """

        if self.input_variables is not None:
            return self.input_variables

        logger.info('identifying sensible variables in code')

        prompt = f"""
        You are an Earth Observation expert with excellent coding of the Google Earth Engine Python API.

        Your task is to analyze the following Google Earth Engine Python API Code and identify the
        constants and variables which are most likely to affect its final result. The Python code 
        below provides an answer to the following Earth Observation question

        <QUESTION>
        {self.question}
        </QUESTION>

        Follow these guidelines:

        - Assume the spatial and temporal extents are corrent. Do not consider those in your analysis.

        - Consider only continuous numerical variables and constants, not categorical variables or variables with string values.

        - Assume as well that the overall processing pipeline is corrent, including the datasets
        it uses.

        - For each variable you identify try to infer:
            (1) a value range with its minimum and maximum values.
            (2) what is the impact (LOW, MEDIUM, HIGH) you estimate that changes in this variable 
                will have on the final output when running the code, together with a justified 
                explanation of your estimate.
            
        - Provide a list of the names of the variables and constant values that you identify in a list
        with json format like in this example:

        [ {{'name': 'threshold', 'value': 40, 'explanation': 'this variable holds the limit by which
        pixels with buildinds are selected', 'value_range': (20,60), 'estimated_impact': 'LOW',
        'estimated_impact_justification': '......'}},
        {{'name': 'cloud_percentage', 'value': 20, 'explanation': 'this variable holds represents
        the maximum clouding percentage admitted for the calculation', 'value_range': (0,100),
        'estimated_impact': 'HIGH', 'estimated_impact_justification': '.....'}}
        ]

        <PYTHON_GOOGLE_EARTH_ENGINE_CODE>
        
        {self.python_code}
        
        </PYTHON_GOOGLE_EARTH_ENGINE_CODE>

        """

        r = self.genai_client.call(prompt)
        sensible_vars = eval(extract_tag(r['answer'], 'json'))

        s = pd.DataFrame(sensible_vars)


        rs = []
        for _,si in s.iterrows():
            v = si['value']
            try:
                vmin, vmax = eval(str(si.value_range))
                svals = distribute_points(vmin, vmax, v, self.n_samples_per_code_variable)
            except Exception as e:
                print (si.value_range, e)
                vmin, vmax, svals = np.nan, np.nan, np.nan
            rs.append({'vmin': vmin, 'vmax':vmax, 'values_for_sensitivity_analysis': svals})

        rs = pd.DataFrame(rs, index=s.index)
        s = s.join(rs)
        s.index = s['name']
        self.input_variables = s
        return s

    
    def run(self):
        """
        run sensitivity analysis by changing the value of each variable identified in 
        gee_identify_sensible_variables to the value for sensitivity analysis, and then 
        executing the modified code and extracting the answer.

        For each input variable (in the code) we do this self.n_samples_per_code_variable times, 
        changing the variable value to a different value for sensitivity analysis each time, 
        and extracting the answer each time.
        
        return: a dict with the sensitivity analysis results, with the following structure:
            {
                'output_variable_1': {
                    'input_variable_1': {
                        'samples': [
                            {'input': value_1, 'output': output_value_1}, is_baseline='yes'},
                            {'input': value_2, 'output': output_value_2},            
                            ]
                        }
                    ...
                }
                ...
            }

        is_baseline refers to the value_1 beding the original value of the variable in the original code, 
        and output_value_1 being the answer obtained when executing the original code with the 
        original variable value.
        """
        s = self.gee_identify_sensible_variables()


        logger.info('----- sensible variables identified in code')
        logger.info(s['name'].values)

        sensitivity = {}
        baseline = extract_standardized_numeric_answer(self.baseline_answer)
        self.output_variables = extract_field_meaning(self.genai_client, self.question, self.baseline_answer)

        # copy units for metadata
        for v in self.output_variables:
            for b in baseline:
                if b['answer_field_name'] == v['answer_field_name']:
                    v['answer_field_units'] = b['answer_field_units']


        for _, si in s.iterrows():
            variable_name = si['name']
            variable_values = si['values_for_sensitivity_analysis']

            # add baseline
            sensitivity[variable_name] = [{'value': si['value'],
                                           'is_baseline': 'yes',
                                           'output': baseline} ]

            logger.info('----')
            logger.info(f"---- analyzing variable {variable_name} within estimated range {si['value_range']}")
            for variable_value in variable_values:
            
                logger.info(f"---- changing {variable_name} value from {si['value']} to {variable_value:.4f}")
    
                if pd.isna(variable_value):
                    logger.info('skipping nan value')
                    continue
                
                
                prompt = f"""
                you are an expert Google Earth Engine Python code interpreter.
                
                your task is to modify the following Google Earth Engine Python code, so that
                the variable named "{variable_name}" has value "{variable_value}" all throughout
                the script.
                
                Output the fully modified code within <MODIFIED_PYTHON> and </MODIFIED_PYTHON> xml tags
                
                --- Python Google Earth Engine Code starts here ---
                
                {self.python_code}
                
                """
    
                logger.info('modifying python code with new variable value')
                r = self.genai_client.call(prompt)
    
                logger.info('executing new code')
                python_modified = extract_xml_tag(r['answer'], 'MODIFIED_PYTHON')
                rr = self.gee_client.exec(python_modified)
    
                logger.info('extracting answer')
                #sensitivity[variable_name] = extract_numeric_answer(self.genai_client, self.question, rr[0])
                sensitivity[variable_name].append( {'value': variable_value, 'output': extract_standardized_numeric_answer(rr[0])} )

            logger.info(f'new result {sensitivity[variable_name]}')


        # store in case in error it can be used for debugging
        self.sensitivity_results = sensitivity
        # gather all data in a two-level dict with first key input var, and second key output var
        ss = {}
        for output_variable in self.output_variables:
            output_variable = output_variable['answer_field_name']
            ss[output_variable] = {}
            for input_variable in s['name'].values:
                ssi = {}
                ssi['samples'] = [{'input': i['value'], 'output': j['answer_field_value']} for i in sensitivity[input_variable] for j in i['output']  if j['answer_field_name'] == output_variable]
                try:
                    ssi['baseline'] = {'input': si['value'], 'output': [i['answer_field_value'] for i in baseline if i['answer_field_name']==output_variable][0]}
                except:
                    ssi['baseline'] = {'input': si['value'], 'output': np.nan}
                ss[output_variable][input_variable] = ssi
                
        self.sensitivity_results = ss

        return ss
    