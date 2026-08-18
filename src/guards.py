"""Deterministic guard, used both to enforce and (later) to measure.

contains_worked_numbers answers one question mechanically: does this text
present worked numeric results, either a matrix of numbers or an arithmetic
calculation? The computational path is told to produce none; this catches it
when it disobeys. The same function will score the computational axis in the
eval, so the thing we enforce is exactly the thing we measure. No LLM judge,
so it never drifts between runs.

Known limit: it targets rendered numeric matrices and arithmetic-with-a-result,
which is how the observed fabrications appear. Invented values stated in bare
prose ("the diagonal entries are 1, 10, -1") are the residual case; the eval
will reveal any that slip through.
"""

import re

# A LaTeX matrix environment and its contents: \begin{bmatrix}...\end{bmatrix}.
_MATRIX_BLOCK = re.compile(
    r"\\begin\{[a-zA-Z]*matrix\}(.*?)\\end\{[a-zA-Z]*matrix\}", re.DOTALL
)
_DIGIT = re.compile(r"\d")

# A worked arithmetic result: an operator, a number, then eventually "= number".
# Uses arithmetic operators (+, -, *, ., \cdot) but NOT the dimension operators
# (x, ×, \times), so "a 2x2 matrix" is not mistaken for a calculation.
_WORKED_RESULT = re.compile(
    r"(?:\\cdot|[+\-−*])\s*\(?-?\d[\d.\s()+\-−*\\a-z]*?=\s*-?\d"
)


def contains_worked_numbers(text: str) -> bool:
    """True if the text renders a numeric matrix or a worked arithmetic result."""
    for block in _MATRIX_BLOCK.findall(text):
        if _DIGIT.search(block):
            return True
    return bool(_WORKED_RESULT.search(text))