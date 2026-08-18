"""One place to build a ChatGroq, so the reasoning_format lesson lives once.

reasoning_format is only valid on reasoning models; passing it to a plain model
(llama-3.1-8b-instant) returns a 400. This factory sets it only where valid.
"""

from langchain_groq import ChatGroq

REASONING_MODEL_HINTS = ("gpt-oss", "qwen3", "deepseek-r1")


def make_llm(model: str, temperature: float = 0.0) -> ChatGroq:
    kwargs = {"model": model, "temperature": temperature}
    if any(hint in model for hint in REASONING_MODEL_HINTS):
        kwargs["reasoning_format"] = "hidden"
    return ChatGroq(**kwargs)