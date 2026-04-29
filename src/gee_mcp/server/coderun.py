import json

from loguru import logger

from .genai import init_genai_client
from .helpers import extract_xml_tag


class GEEPythonExecution:
    def __init__(self, genai_client=None):
        self.genai_client = genai_client or init_genai_client()

    def exec(self, code):
        namespace: dict = {}
        exec(code, namespace)  # pylint: disable=exec-used
        gee_main = namespace["gee_main"]
        return gee_main()

    def fix_code_iteration(self, code, previous_errors=[]):
        """
        assumes ee.Authenticate and ee.Initialized have been
        called previously.

        Fixes GEE Python code by calling genai with the error message and the original code,
        and asking for a fixed version of the code. The prompt includes specific guidelines
        to ensure that the generated code is correct and can be run in the user's environment.
        The method returns a dict with the original code, the fixed code, the original error
        message, the fixed graph, and an explanation of the fix.
        """

        error = ""

        try:
            r = self.exec(code)
            return {"result": r[0], "map": r[1], "code_original": code}
        except Exception as e:
            error = e
            logger.debug(f"error {e}")

            if len(previous_errors) > 0:
                logger.debug(f"previous errors: {previous_errors}")
                previous_errors = (
                    f"You already tried to fix these previous errors:\n"
                    + "\n - ".join(previous_errors)
                )
            else:
                previous_errors = ""

            fix_prompt = f"""
    You an expert Google Earth Engine Python API programmer and I have this python 
    Google Earth Engine code

    <PYTHON_CODE>
    {code}
    </PYTHON_CODE>

    but when I execute it I get this error

    <ERROR>
    {error}
    </ERROR>

    {previous_errors}

    your main task is to fix the error producing a complete full new verstion of code and 
    explain your fix.

    then, create also a direct acyclic graph representing the process implemented in the 
    fixed python code in mermaid language.

    Follow these guidelines:

        - If the error relates to computation being timed out, or to memory issues, or to any 
        other issue related to the size of the data being processed, then you should implement 
        a fix that consists in reducing the size of the data being processed (for example, by 
        reducing the spatial or temporal extent of the analysis) and explain that this is the 
        reason for the error and that this fix is just for testing purposes.

        - Enclose the new python code in a <FIXED_PYTHON> xml tag.

        - Enclose your explanation in a <FIX_EXPLANATION> xml tag.

        - Enclose the mermaid graph with a <FIXED_PYTHON_GRAPH> xml tag.

        - Include in the graph nodes labels what Google Earth Engine datasets you would use.

        - Respect the original script structure as much as possible, only changing what 
        is strictly necessary to fix the error. In particular:

            - Do no call ee.Authenticate() or ee.Initialize() in the fixed python code since 
            I will run the code in an environment where I have already authenticated and i
            nitialized the Earth Engine API.

            - Respect any variable named 'map' of type 'geemap.Map' at the end of the python 
                script so that I can display the map within a jupyter notebook.

            - Do no include in the python code any call to ee.Authenticate() or ee.Initialize() 
                since I will run the code in an environment where I have already authenticated and 
                initialized the Earth Engine API.

            - Respect all the code being within a function named 'gee_main' that must return 
            a tuple containing a string with the result and a geemap.Map object. Do not 
            call that function at the end of the script, I will call it directly later.

            - Do not catch any exception in the python code, since I want to be able to debug 
                it if something goes wrong.


        - Any error or invalid check detected by your code must result in a ValueException being 
        thrown, not just printing a message.

            """
            logger.info("calling genai to fix python code")
            rf = self.genai_client.call(fix_prompt)

            code_fixed = extract_xml_tag(rf["answer"], "FIXED_PYTHON")
            graph_fixed = extract_xml_tag(rf["answer"], "FIXED_PYTHON_GRAPH")
            explanation_fixed = extract_xml_tag(
                rf["answer"], "FIX_EXPLANATION"
            )

            return {
                "code_original": code,
                "code_fixed": code_fixed,
                "code_original_error": str(error),
                "graph_fixed": graph_fixed,
                "explanation_fixed": explanation_fixed,
            }

    def run_fix_code(self, code, number_of_iterations=5):
        """
        Runs the provided GEE Python code and if it raises an error, calls the fix_code_iteration method to try to fix it.
        This process is repeated for a maximum number of iterations until the code runs without errors or the maximum
        number of iterations is reached. The method returns the history of fixes attempted, and if successful,

        :param self: Description
        :param code: Python code to execute and fix if errors are found
        :param number_of_iterations: Number of iterations to try fixing the code if errors are found. Default is 5.
        :return: The history of fixes attempted. If successful, that last fix in history contains the fixed code and result.
        :rtype: list

        """

        fix_history = []
        error_history = []
        for i in range(number_of_iterations):
            logger.info(f"running code attempt {i+1}")
            rf = self.fix_code_iteration(code, previous_errors=error_history)
            fix_history.append(rf)
            if "result" in rf.keys():
                if i > 0:
                    logger.info("code is fixed")
                    return fix_history
                else:
                    logger.info("code is already correct, no fix needed")
                    return fix_history
            else:
                code = rf["code_fixed"]
                error_history.append(rf["code_original_error"])
        logger.info(f"could not fix code in {number_of_iterations} iterations")
        return fix_history


def _execute_gee_python(code: str):
    gee = GEEPythonExecution()
    try:
        r = gee.exec(code)
        return json.dumps({"result": r[0]}, indent=2, default=str)
    except Exception as e:
        return json.dumps({"errors": str(e)}, indent=2, default=str)
