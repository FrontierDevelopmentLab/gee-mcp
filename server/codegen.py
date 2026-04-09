import re
import mermaid as md
from loguru import logger
from glob import glob
import json
import numpy as np
import os
import ee
from .helpers import extract_tag, extract_xml_tag, remove_leading_spaces, NoTagFoundError
from .genai import init_genai_client
from  .coderun import GEEPythonExecution

class QuestionRecord:

    """
    class to manage EO questions and associated artifacts (graph, generated code, etc.)
    """
    @classmethod
    def load_question(cls, question_file):
        with open(question_file) as f:
            q = json.load(f)

        if 'graph' in q.keys():
            q['graph_javascript'] = q['graph']
            del(q['graph'])

        if 'graph_fixed' in q.keys():
            q['graph_javascript_fixed'] = q['graph_fixed']
            del(q['graph_fixed'])

        if 'explanation' in q.keys():
            q['explanation_javascript'] = q['explanation']
            del(q['explanation'])

        if 'thinking' in q.keys():
            q['thiking_python'] = q['thinking']
            del(q['thinking'])

        dataset_dir = '/'.join(question_file.split('/')[:-1])
        
        qo = cls(question=q['question'], dataset_dir=dataset_dir, exists_ok=True)
        if 'remarks' in q.keys():
            qo.remarks_for_prompts = q['remarks']
        
        qo.record = q
        qo.question_file = question_file

        if 'reference_refined_question' in qo.record.keys():

            s = qo.record['reference_refined_question']['steps']
            s = "\n - "+"\n - ".join([i['sub_question'] for i in s])

            qo.record['reasoning_steps'] = s

        return qo    

    def __init__(self, 
                 question, 
                 dataset_dir=None, 
                 exists_ok=False):
        """
        if 'dataset_dir' is None, nothing will be saved (useful for transient calls)
        """
        self.question = remove_leading_spaces(question)
        self.dataset_dir = dataset_dir
        self.question_file = None

        if dataset_dir is not None:
            r = self.check_question_exists()
            
            if not exists_ok:
                if r:
                    raise ValueError(f'question already exists in {r}')
            else:
                self.question_file = r

        self.record = {'question': self.question}   

    def clean(self):
        for k in list(self.record.keys()):
            if k not in ['question', 'reference_refined_question', 'reasoning_steps']:
                del(self.record[k])
        return self

    def __setitem__(self, k,v):
        self.record[k] = v

    def __getitem__(self, k):
        return self.record[k]
    
    def keys(self):
        return self.record.keys()

    def items(self):
        return self.record.items()

    def check_question_exists(self):
        # checks if the question already exists in any of the entries
        # in the dataset (checks q*.json files under dataset_dir)
        q = remove_leading_spaces(self.question)
        q = self.question.replace('\n', ' ').strip()

        files = glob(f'{self.dataset_dir}/q*.json')
        for file in files:
            with open(file) as f:
                qj = json.load(f)['question']
                qj = remove_leading_spaces(qj)
                qj = qj.replace('\n', ' ').strip()
                if q == qj:
                    return file
                
        return False    

    def set_question_file(self, question_file):
        self.question_file = question_file
        return self

    def save(self):
        """ saves dict 'r' under 'dataset_dir' as qNNNN.json
            with a consecutive number wrt the files already
            existing in the folder
        """

        # skip if no dataset dir provided
        if self.dataset_dir is None:
            return

        if self.question_file is not None:
            if os.path.isfile(self.question_file):
                logger.info(f'saving record in existing file {self.question_file}')
            else:
                logger.info(f'saving record in new file {self.question_file}')
            with open(self.question_file, 'w') as f:
                json.dump(self.record, f, indent=4)
            return

        # if question file is not set, create a consecutive number file
        files = glob(f'{self.dataset_dir}/q*.json')
        if len(files)==0:
            next_qnumber = 1
        else:
            next_qnumber = np.max([int(f.split('/')[-1].split('.')[0][1:]) for f in files])+1
        fname = f'{self.dataset_dir}/q{next_qnumber:04d}.json'

        logger.info(f'saving record in new file {fname}')
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(self.record, f, ensure_ascii=False, indent=4)


class GeoQuestion:


    def __init__(self, 
                 question_record, 
                 gee_dataset_list=None, 
                 number_of_fix_iterations = 5,
                 ):
        
        """
        constructor
        
        :param question_record: a QA object
        :param genai: the genai object to generate assets
        :param gee_dataset_list: list of available GEE datasets to include in the prompts. If None, will not include that information in the prompts.
        """

        self.question_record = question_record
        self.qr = self.question_record
        self.genai_client = init_genai_client()
        self.remarks_for_prompts = ''
        self.gee_dataset_list = gee_dataset_list
        self.number_of_fix_iterations = number_of_fix_iterations
        self.fix_code = number_of_fix_iterations > 0

        if self.fix_code:
            logger.info(f'will attempt to fix any code generated in max {self.number_of_fix_iterations} iterations')
            self.gee_runtime = GEEPythonExecution(genai_client=self.genai_client)

    def set_remarks_for_prompts(self, remarks):
        self.remarks_for_prompts = remarks
        return self


    def get_dataset_prompt_instructions(self):
        dataset_instructions = f"""
        - you must use one or more of the Google Earth Engine datasets included in the 
          following list {self.gee_dataset_list}
        - you do not need to use all the datasets in the list, but you can only use the 
          ones in that list
        """ if self.gee_dataset_list is not None else ''

        return dataset_instructions

    def get_prompt_for_python_from_question(self):

        dataset_instructions = self.get_dataset_prompt_instructions()

        p = f"""
        You are an expert earth observation scientist and Python programmer. You are 
        given the following question and your task is to develop a Python script
        following Google Earth Engine Python API to solve the question below. 

        Follow these guidelines:

        {dataset_instructions}

        - Include in the graph nodes labels what Google Earth Engine datasets you would use.
        
        - Enclose the python code with a <PYTHON> xml tag

        - Create a variable named 'map' of type 'geemap.Map' at the end of the python 
          script so that I can display the map within a jupyter notebook.

        - Do no include in the python code any call to ee.Authenticate() or ee.Initialize() 
          since I will run the code in an environment where I have already authenticated and 
          initialized the Earth Engine API.

        - Include all code in a function called 'gee_main' that must return a tuple containing
          a string with the result and a geemap.Map object. Do not call that function 
          at the end of the script, I will call it directly later.

        - The string with the final numeric results that gee_main must return must follow 
          XML formatting such as in this example:
            <RESULT>
                <VARIABLE_NAME>flooded_area</VARIABLE_NAME>
                <VALUE>123.45</VALUE>
                <UNITS>km^2</UNITS>
            </RESULT>

        - Give the final results variable names that convey the meaning of their nature, for 
          instance "flooded_area" and state in a comment its measurement units.
            
        - If there are several results, print one <RESULT> block per result, with the 
        corresponding variable name, value, and units in each block.

        - Do not catch any exception in the python code, since I want to be able to debug 
          it if something goes wrong.


        <QUESTION>
        {self.qr['question']}
        </QUESTION>

        {self.remarks_for_prompts}
        """
        return remove_leading_spaces(p)      


    def get_prompt_for_abstract_graph_from_question(self):

        dataset_instructions = self.get_dataset_prompt_instructions()

        p = f"""
        You are an expert earth observation scientist. You are given the following 
        question and your task is to design a pipeline that, starting from selected 
        Google Earth Engine datasets (GEE) solves the question by providing a 
        specific answer. 

        You must generate two things:

        (1) a direct acyclic graph in mermaid language representing the pipeline that, 
            when implemented in the python code, would solve the question.

        (2) a high level textual explanation of what the pipeline does

        Follow these guidelines:

        {dataset_instructions}

        - Include in the relevant graph nodes labels of the Google Earth Engine datasets 
          you would use.
        
        - Enclose the mermaid graph with a <GRAPH> xml tag. Do not include any triple quotes
          tag, just the string defining the mermaid graph.

        - Enclose the high level explanation with a <EXPLANATION> xml tag.

        - The graph nodes must be text verbose and MUST not include any reference to 
          Python code or Google Earth Engine API operatons.

        - The graph must be targeted at producing a single number, or small set 
          of numbers that solves the question.

        <QUESTION>
        {self.qr['question']}
        </QUESTION>

        {self.remarks_for_prompts}
        """
        return remove_leading_spaces(p)      

    def get_prompt_for_python_from_abstract_graph(self):


        dataset_instructions = self.get_dataset_prompt_instructions()

        p = f"""
        You are an expert earth observation scientist and Python programmer. You are 
        given a processing pipeline as a Mermaid graph that is suposed to answer a 
        specific earth observation question. 

        Your task is to develop a Python script following Google Earth Engine 
        Python API that implements the pipeline. 

        Follow these guidelines:
        
        {dataset_instructions}

        - You will find below the processing pipeline and the question it pretends to answer.

        - Use mainly the graph to guide your code generation. Resort to the question
          just to make sure the code generates the answer it is being requested.

        - Enclose the python code with a <PYTHON> xml tag. Do not use triple quotes.

        - Create a variable named 'map' of type 'geemap.Map' at the end of the python 
          script so that I can display the map within a jupyter notebook.

        - Do no include in the python code any call to ee.Authenticate() or ee.Initialize() 
          since I will run the code in an environment where I have already authenticated and 
          initialized the Earth Engine API.

        - Include all code in a function called 'gee_main' that must return a tuple containing
          a string with the result and a geemap.Map object. Do not call that function 
          at the end of the script, I will call it directly later.

        - The string with the final numeric results that gee_main must return must follow 
          XML formatting such as in this example:
            <RESULT>
                <VARIABLE_NAME>flooded_area</VARIABLE_NAME>
                <VALUE>123.45</VALUE>
                <UNITS>km^2</UNITS>
            </RESULT>

        - Give the final results variable names that convey the meaning of their nature, for 
          instance "flooded_area" and state in a comment its measurement units.
            
        - If there are several results, print one <RESULT> block per result, with the 
        corresponding variable name, value, and units in each block.

        - Do not catch any exception in the python code, since I want to be able to debug 
          it if something goes wrong.


        <QUESTION>
        {self.qr['question']}
        </QUESTION>

        <GRAPH>
        {self.qr['abstract_graph']}
        </GRAPH>
        """
        return remove_leading_spaces(p)      
    

    def get_prompt_for_python_from_reasoning_steps(self):

        dataset_instructions = self.get_dataset_prompt_instructions()


        p = f"""
        You are an expert earth observation scientist and Python programmer. You are 
        given a set of reasoning steps that are suposed to help answer specific earth
        observation question. Both the question and the reasoning steps are provided 
        below.

        Your task is to develop a Python script following Google Earth Engine 
        Python API that follow those steps to answer the earth observation question. 

        Follow these guidelines:
        
        {dataset_instructions}

        - Use reasoning steps to guide your code generation. 
        
        - Use the question to (1) double check the steps do provide an procedure to
        arrive at a plausible anwer; and (2) make to sure the code generates the
        answer it is being requested.

        - Enclose the python code with a <PYTHON> xml tag. Do not use triple quotes.

        - Create a variable named 'map' of type 'geemap.Map' at the end of the python 
          script so that I can display the map within a jupyter notebook.

        - Do no include in the python code any call to ee.Authenticate() or ee.Initialize() 
          since I will run the code in an environment where I have already authenticated and 
          initialized the Earth Engine API.

        - Include all code in a function called 'gee_main' that must return a tuple containing
          a string with the result and a geemap.Map object. Do not call that function 
          at the end of the script, I will call it directly later.

        - The string with the final numeric results that gee_main must return must follow 
          XML formatting such as in this example:
            <RESULT>
                <VARIABLE_NAME>flooded_area</VARIABLE_NAME>
                <VALUE>123.45</VALUE>
                <UNITS>km^2</UNITS>
            </RESULT>

        - Give the final results variable names that convey the meaning of their nature, for 
          instance "flooded_area" and state in a comment its measurement units.
            
        - If there are several results, print one <RESULT> block per result, with the 
        corresponding variable name, value, and units in each block.

        - Do not catch any exception in the python code, since I want to be able to debug 
          it if something goes wrong.


        <QUESTION>
        {self.qr['question']}
        </QUESTION>

        <REASONING_STEPS>
        {self.qr['reasoning_steps']}
        </REASONING_STEPS>
        """
        return remove_leading_spaces(p)      
    

    def run_fix_code(self, python_code):
        r = self.gee_runtime.run_fix_code(python_code, 
                                          number_of_iterations=self.number_of_fix_iterations)
        z = {}
        
        if len(r)==0:
            z['python_code_status'] = 'error'
            z['python_code_error'] = 'no running iterations'
        elif 'result' in r[-1].keys():
            z['python_code_status'] = 'success'
            z['python_code'] = r[-1]['code_original']
            z['python_code_result'] = r[-1]['result']
        elif 'code_original_error' in r[-1].keys():
            z['python_code_status'] = 'error'
            z['python_code'] = r[-1]['code_original']
            z['python_code_error'] = r[-1]['code_original_error']
        else:
            z['python_code_status'] = 'error'
            z['python_code_error'] = 'internal error fixing code'

        z['python_code_fix_history'] = [{k:v for k,v in ri.items() if k!='map'} for ri in r]

        return z    


    def _generate_python(self, prompt):
        """
          calls genai to generate python expecting to find a tag <PYTHON> in
          the response (assuming the prompt requests it).

        it attempts a a few times if the tag is not found

        """
        for i in range(5):
            try:
                r = self.genai_client.call(prompt)
                self.genai_response = r

                # check we can extract the tag content
                try:
                    _ = extract_xml_tag(r['answer'], 'PYTHON')
                    logger.debug('<PYTHON> found in response')
                    return r
                except NoTagFoundError as e:
                    # in case python is with ```python mark up
                    _ = extract_tag(r['answer'], 'python')
                    r['answer'] = r['answer'].replace('```python', '<PYTHON>')\
                                             .replace('```', '</PYTHON>')
                    return r
            
            except NoTagFoundError as e:
                if i==2:
                    logger.debug(f'no PYTHON tag found in genai response, response content is\n\n{r["answer"]}')
                    raise e
                else:
                    logger.debug('no PYTHON tag found in genai response, trying again...')

    def generate_python_from_question(self):

        p = self.get_prompt_for_python_from_question()
        
        logger.info(f'calling genai to generate gee python code from question')
        r = self._generate_python(p)

        self.genai_response = r
        c = extract_xml_tag(r['answer'], 'PYTHON')
        e = r['answer'].replace(f'<PYTHON>{c}</PYTHON>', '')            
        
        if not 'question_derived' in self.qr.keys():
            self.qr['question_derived'] = {}

        self.qr['question_derived'].update( {
            'python_code': c,
            'python_code_thinking': r['thought'],
            'python_code_explanation': e,
            'python_code_remarks': self.remarks_for_prompts,
            'python_code_status': 'not_run'
        })

        self.qr.save()

        if self.fix_code:
            z = self.run_fix_code(c)
            self.qr['question_derived'].update(z)
            self.qr.save()

        return self.qr
    
    def generate_abstract_graph_from_question(self):

        if 'abstract_graph' in self.qr.keys():
            raise ValueError('abstract graph already present in question record')

        p = self.get_prompt_for_abstract_graph_from_question()

        logger.info(f'calling genai to generate abstract graph from question')
        r = self.genai_client.call(p)
        self.genai_response = r

        if not 'question_derived' in self.qr.keys():
            self.qr['question_derived'] = {}


        self.qr['question_derived'].update ({
            'abstract_graph': extract_xml_tag(r['answer'], 'GRAPH'),
            'abstract_graph_thinking': r['thought'],
            'abstract_graph_explanation': extract_xml_tag(r['answer'], 'EXPLANATION'),
            'abstract_graph_remarks': self.remarks_for_prompts  
        })

        self.qr['abstract_graph'] = self.qr['question_derived']['abstract_graph']

        self.qr.save()

        return self.qr
    

    def generate_python_from_abstract_graph(self):

        p = self.get_prompt_for_python_from_abstract_graph()

        logger.info(f'calling genai to generate gee python from graph')

        r = self._generate_python(p)
        self.genai_response = r
        c = extract_xml_tag(r['answer'], 'PYTHON')
        e = r['answer'].replace(f'<PYTHON>{c}</PYTHON>', '')            
        
        if not 'abstract_graph_derived' in self.qr.keys():
            self.qr['abstract_graph_derived'] = {}

        self.qr['abstract_graph_derived'] = {
            'python_code': c,
            'ptyhon_code_thinking': r['thought'],
            'python_code_explanation': e,
            'python_code_remarks': self.remarks_for_prompts,
            'python_code_status': 'not_run'
        }

        self.qr.save()

        if self.fix_code:
            z = self.run_fix_code(c)
            self.qr['abstract_graph_derived'].update(z)
            self.qr.save()

        return self.qr
    

    def generate_python_from_reasoning_steps(self):

        p = self.get_prompt_for_python_from_reasoning_steps()

        logger.info(f'calling genai to generate gee python from reasoning steps')

        r = self._generate_python(p)
        self.genai_response = r
        c = extract_xml_tag(r['answer'], 'PYTHON')
        e = r['answer'].replace(f'<PYTHON>{c}</PYTHON>', '')            
        
        if not 'reasoning_steps_derived' in self.qr.keys():
            self.qr['reasoning_steps_derived'] = {}

        self.qr['reasoning_steps_derived'] = {
            'python_code': c,
            'ptyhon_code_thinking': r['thought'],
            'python_code_explanation': e,
            'python_code_remarks': self.remarks_for_prompts,
            'python_code_status': 'not_run'
        }

        self.qr.save()

        if self.fix_code:
            z = self.run_fix_code(c)
            self.qr['reasoning_steps_derived'].update(z)
            self.qr.save()

        return self.qr

# functions for direct calling with strings

def _generate_python_from_question(question: str, gee_datasets: list = None, fix_code: bool = True) -> dict:

    q = QuestionRecord(question=question)  
    g = GeoQuestion(q, 
                    gee_dataset_list=gee_datasets, 
                    number_of_fix_iterations=5 if fix_code else 0)
    
    g.generate_python_from_question()
    
    return q.record['question_derived']

def _generate_abstract_graph_from_question(question: str, gee_datasets: list = None) -> dict:

    q = QuestionRecord(question=question)  
    g = GeoQuestion(q, 
                    gee_dataset_list=gee_datasets, 
                    )
    
    g.generate_abstract_graph_from_question()
    
    return q.record['question_derived']

def _generate_python_from_reasoning_steps(question: str, 
                                         reasoning_steps: str,
                                         gee_datasets: list = None, 
                                         fix_code: bool = True) -> dict:
    
    q = QuestionRecord(question=question)  
    q['reasoning_steps'] = reasoning_steps
    g = GeoQuestion(q, 
                    gee_dataset_list=gee_datasets, 
                    number_of_fix_iterations=5 if fix_code else 0)
    
    g.generate_python_from_reasoning_steps()
    
    return q.record['reasoning_steps_derived']

def _generate_python_from_abstract_graph(question: str, 
                                         abstract_graph: str,
                                         gee_datasets: list = None, 
                                         fix_code: bool = True) -> dict:

    q = QuestionRecord(question=question)  
    q['abstract_graph'] = abstract_graph
    g = GeoQuestion(q, 
                    gee_dataset_list=gee_datasets, 
                    number_of_fix_iterations=5 if fix_code else 0)
    
    g.generate_python_from_abstract_graph()
    
    return q.record['abstract_graph_derived']

