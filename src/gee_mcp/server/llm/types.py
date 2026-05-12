from typing import Any, TypedDict


class LLMCallReturn(TypedDict):
    """
    Structure of return from `call`:
    {"answer": answer, "thought": thought, "response": response}
    """

    answer: str | None  # extracted answer text; None if the model returned no text
    thought: str | None
    response: Any  # raw provider SDK response object, or None for cache hits
