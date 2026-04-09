import re
import json
import pandas as pd
import numpy as np
from loguru import logger
import itertools
import matplotlib.pyplot as plt

from .helpers import extract_tag, extract_xml_tag
from .coderun import _execute_gee_python
from .genai import init_genai_client


def _get_datasets_locations_and_periods(question: str, 
                                       gee_datasets: list[dict] = None, 
                                       ) -> dict:
    
    genai_client = init_genai_client()


    dataset_instructions = f"""
    We also know that the following datasets are the only ones available in Google
    Earth Engine

    {gee_datasets}

    - you must use one or more of the Google Earth Engine datasets 
    - you do not need to use all the datasets in the list, but you can only use the 
    ones in that list
    """ if gee_datasets is not None else ''

    prompt = f"""
    you are an expert data scientist specialized in earth observation datasets and, specifically, in Google
    Earth Engine datasets and algorithms.
    
    **The context**
    
    We have the following question that we want to solve with Google Earth Engine
    
    <QUESTION>
    {question}
    </QUESTION>

    {dataset_instructions}
        
    **You task**
    
    Your must provide a list of the Google Earth Engine datasets required to answer the question, 
    together with the time periods and coordinates of the geographical areas of interest (AOIs) 
    required for each dataset. 
    
    Your output must be a list of json strings with fields 'dataset', 'date_periods', 'aois' where
    
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

    logger.debug('calling genai for extraction')
    
    r = genai_client.call(prompt)
    r['json'] = json.loads(extract_tag(r['answer'], 'json'))
        
    return r

def _extract_factuality_issues(question: str, python_code: str) -> str:

    genai_client = init_genai_client()

    prompt = f"""
    You are a helpful assistant for Earth Observation data analysis with Google Earth Engine.
    The Python code below was devised to answer the following question

    <QUESTION>
    {question}
    </QUESTION>

    Your task is to analyze the Google Earth Engine Python code below and extract what aspects 
    or issues are making scientific or data assumptions either explicitly or implicitly and might require 
    factual verification.

    <PYTHON_CODE>
    {python_code}
    </PYTHON CODE>

    Your response must be a list of json structures, each one describing a specific aspect you identify
    and containing the following fields:
    [ 
    {{"title": "A short title describing the aspect or assumption",
      "description": "A detailed description of the aspect or assumption, why it might require factual verification",
      "facts": "Data, information, constants or facts to be verified",
      "question_for_expert": "The question that should be posed to an expert to verify the aspect or assumption"
    }}
    {{ ... more issues ...}}
    ]

    """
    logger.debug('calling genai to extract factuality issues')
    r = genai_client.call(prompt)
    return extract_tag(r['answer'], 'json')

def distribute_points(a, b, x, n):
    """
    Distributes n points within (a, b) given an existing point x
    to maximize the minimum distance between all points.
    """
    if not (a <= x <= b):
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


class SensitivityAnalizer:
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


    def __init__(self, question, python_code, python_code_result, 
                  n_samples_per_code_variable=3):

        self.genai_client = init_genai_client()

        self.question    = question
        self.python_code = python_code
        self.baseline_answer = python_code_result
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

        logger.debug('identifying sensible variables in code')

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
            (1) the type of numeric variable: 'int' or 'float'
            (2) a value range with its minimum and maximum values.
            (3) what is the impact (LOW, MEDIUM, HIGH) you estimate that changes in this variable 
                will have on the final output when running the code, together with a justified 
                explanation of your estimate.
            
        - Provide a list of the names of the variables and constant values that you identify in a list
        with json format like in this example:

        [ {{'name': 'threshold', 'value': 40, 'type': 'int', 'explanation': 'this variable holds the limit by which
        pixels with buildinds are selected', 'value_range': (20,60), 'estimated_impact': 'LOW',
        'estimated_impact_justification': '......'}},
        {{'name': 'cloud_percentage', 'value': 20, 'type': 'float', 'explanation': 'this variable holds represents
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
                if si['type'] == 'int':
                    svals = [int(i) for i in svals]
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

        logger.info(f"sensible variables identified in code {s['name'].values}")

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

            logger.info(f"analyzing variable {variable_name} within estimated range {si['value_range']}")

            try:
                variable_values = [float(i) for i in variable_values]
            except Exception as e:
                logger.warning(f"skipping '{variable_name}' as could not convert values {variable_values} to float")
                continue

            for variable_value in variable_values:
            
                logger.debug(f"---- changing {variable_name} value from {si['value']} to {variable_value:.4f}")
    
                if pd.isna(variable_value):
                    logger.debug('skipping nan value')
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
    
                logger.debug('modifying python code with new variable value')
                r = self.genai_client.call(prompt)
    
                logger.debug('executing new code')
                python_modified = extract_xml_tag(r['answer'], 'MODIFIED_PYTHON')
                rr = _execute_gee_python(python_modified)
                rr = json.loads(rr)

                sensitivity[variable_name].append( {'value': variable_value, 'output': extract_standardized_numeric_answer(rr['result'])} )

            logger.debug(f'new result {sensitivity[variable_name]}')


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

    def get_analysis_summary(self):    

        metadata_output_vars = self.output_variables
        metadata_input_vars = self.input_variables
        
        output_vars = list(self.sensitivity_results.keys())
        input_vars = list(self.sensitivity_results[output_vars[0]].keys())
        
        h, w = len(output_vars), len(input_vars)
        fig, axs = plt.subplots(h, w, figsize=(w*4, h*3))
        
        axs = axs.flatten()
        
        md = f'## question \n\n {self.question}\n\n'
        md += '## output variables\n'
        for metadata_output_var in metadata_output_vars:
            md += f"- **{metadata_output_var['answer_field_name']}** ({metadata_output_var['answer_field_units']}): {metadata_output_var['answer_field_meaning']}\n"
            
        md += '## input variables (in the code)\n'
        for _,metadata_input_var in metadata_input_vars.iterrows():
            md += f"**{metadata_input_var.name}**:\n"\
                  f"- **meaning**: {metadata_input_var.explanation}\n"\
                  f"- **values range**: {metadata_input_var.value_range}\n"\
                  f"- **estimated impact**: {metadata_input_var.estimated_impact}\n"\
                  f"- **impact justification**: {metadata_input_var.estimated_impact_justification}\n\n"
        
        md += '\n## sensitivity with respect to variables in the code\n\n**output vars** in y-axes, **input vars** in x-axes'

        for (output_var, input_var, ), ax in zip(itertools.product(output_vars, input_vars), axs):
        
            metadata_output_var = [i for i in metadata_output_vars if i['answer_field_name'] == output_var][0]
            
            vv = self.sensitivity_results[output_var][input_var]
            zz = pd.DataFrame(vv['samples']).sort_values(by='input', ascending=True)
            
            ax.plot(zz['input'], zz['output'], marker='o')
            if input_var == input_vars[0]:
                ax.set_ylabel(f"OUTPUT\n{output_var} ({metadata_output_var['answer_field_units']})")
            ax.set_xlabel(f'INPUT  {input_var}')
            ax.grid()
        
            #ax.set_title(f"OUTPUT\n{output_var} ({metadata_output_var['answer_field_units']})")
        
            
            # set uniform output range
            all_outputs = np.r_[[ii['output'] for i in self.sensitivity_results[output_var].values() for ii in i['samples']]]
            try:
                omin, omax = all_outputs.min(), all_outputs.max()
                vmin = omin - (omax - omin) * 0.05
                vmax = omax + (omax - omin) * 0.05
                ax.set_ylim(vmin, vmax)
            except:
                # in case of categorical variables, there is no max or min
                pass    
            
        fig.tight_layout()

        import io
        import base64
        strbytes = io.BytesIO()
        fig.savefig(strbytes, format='jpg')
        strbytes.seek(0)
        img_b64 = base64.b64encode(strbytes.read()).decode()

        md = f'{md}\n\n![Hello World](data:image/png;base64,{img_b64})'
        return md


# functions for direct calling with strings
def _sensitivity_analysis(question: str,
                          python_code: str, 
                          baseline_answer: str) -> str:
    
    rs = SensitivityAnalizer(question=question, 
                             python_code=python_code, 
                             python_code_result=baseline_answer)

    rs.run()
    markdown_answer = rs.get_analysis_summary()
    return markdown_answer


def _identify_sensible_variables(question: str,
                                 python_code: str, 
                                 baseline_answer: str) -> str:
    
    rs = SensitivityAnalizer(question=question, 
                             python_code=python_code, 
                             python_code_result=baseline_answer)

    rs.run()
    result = [i.to_json() for _m,i in rs.input_variables.iterrows()]
    return result

def _extract_factuality_issues(question: str, python_code: str) -> str:

    from .genai import init_genai_client

    genai_client = init_genai_client()

    prompt = f"""
    You are a helpful assistant for Earth Observation data analysis with Google Earth Engine.
    The Python code below was devised to answer the following question

    <QUESTION>
    {question}
    </QUESTION>

    Your task is to analyze the Google Earth Engine Python code below and extract what aspects 
    or issues are making scientific or data assumptions either explicitly or implicitly and might require 
    factual verification.

    <PYTHON_CODE>
    {python_code}
    </PYTHON CODE>

    Your response must be a list of json structures, each one describing a specific aspect you identify
    and containing the following fields:
    [ 
    {{"title": "A short title describing the aspect or assumption",
      "description": "A detailed description of the aspect or assumption, why it might require factual verification",
      "facts": "Data, information, constants or facts to be verified",
      "question_for_expert": "The question that should be posed to an expert to verify the aspect or assumption"
    }}
    {{ ... more issues ...}}
    ]

    """
    logger.info('calling genai to extract factuality issues')
    r = genai_client.call(prompt)
    return extract_tag(r['answer'], 'json')

