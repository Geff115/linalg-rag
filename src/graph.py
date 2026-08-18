"""
Phase 3: the corrective RAG graph.

A linear chain cannot branch; each failure we measured needs a different branch,
which is why this is a graph. Flow:

    retrieve
      |-- top score < threshold ------------------> refuse   (far out-of-scope)
      `-- else --> grade (LLM: relevant? computational?)
                     |-- not relevant -------------> refuse   (near out-of-scope)
                     |-- relevant + computational -> generate_computational --> guard
                     `-- relevant + conceptual ----> generate_conceptual

The conceptual path is the same prompt as chain.py, untouched, because it
already scores 94% correct. We only add gates and a router around it.

    python src/graph.py "What is the rank of a matrix?"
    python src/graph.py -v "Show a worked example of subtracting matrices"
    python src/graph.py "What is the LU decomposition?"
"""

from __future__ import annotations

import argparse
import json
import re

from dotenv import load_dotenv
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

import config
from chain import SYSTEM, HUMAN, format_context, build_store
from llm import make_llm
from guards import contains_worked_numbers

load_dotenv()


class GraphState(TypedDict):
    question: str
    docs: NotRequired[list]
    top_score: NotRequired[float]
    relevant: NotRequired[bool]
    computational: NotRequired[bool]
    answer: NotRequired[str]
    route: NotRequired[str]


# Built once at import; nodes close over these.
_store = build_store()
_gen_llm = make_llm(config.LLM_MODEL, config.LLM_TEMPERATURE)
_grade_llm = make_llm(config.LLM_MODEL, 0.0)

import time


def _invoke_with_retry(llm, messages, tries: int = 4, base_sleep: float = 3.0):
    """Retry transient failures (per-minute rate limits, network blips).
    Raise immediately on a daily cap or a permanent 4xx, which retrying can't fix.
    """
    for attempt in range(tries):
        try:
            return llm.invoke(messages)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "per day" in msg or "TPD" in msg or "tokens per day" in msg:
                raise
            if ("Error code: 4" in msg) and ("429" not in msg):
                raise
            if attempt == tries - 1:
                raise
            time.sleep(base_sleep * (attempt + 1))

GRADE_SYSTEM = """You route a Linear Algebra study assistant. Given a student \
question and excerpts retrieved from the course book, decide two things.

Respond with ONLY a JSON object, no prose and no code fences, with exactly \
these keys:
  "relevant": true if the excerpts actually address the SPECIFIC topic of the \
question (they contain material about that exact concept), false if they are \
about adjacent but different topics or do not cover it.
  "computational": true if the question asks for a worked numeric example, a \
calculation, or a step-by-step numeric result; false if it asks for a concept, \
definition, or explanation.

Judge "relevant" by TOPIC MATCH, not completeness. Excerpts that are garbled or \
incomplete but still about the right topic count as relevant."""

GRADE_HUMAN = """QUESTION:
{question}

RETRIEVED EXCERPTS:
{context}

JSON:"""

COMPUTATIONAL_SYSTEM = """You are a Linear Algebra study assistant. The student \
is asking for a worked numeric example or calculation.

Critical constraint: this course book's equations and matrices do NOT extract \
reliably from the PDF, so specific numbers in the excerpts may be corrupted. \
You must NOT reproduce or compute specific numeric results. Do NOT write \
matrices of numbers, do NOT show arithmetic, do NOT state computed values.

Instead:
- Explain the METHOD in plain words: the steps and the rule to follow.
- Point the student to the page where the book works the actual example, using \
the citations shown in the excerpts.
- Keep it in prose. No equations, no numeric matrices, no calculations.

End with a line starting with "See:" giving the section and page to open."""

_grade_prompt = ChatPromptTemplate.from_messages(
    [("system", GRADE_SYSTEM), ("human", GRADE_HUMAN)])
_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", HUMAN)])
_comp_prompt = ChatPromptTemplate.from_messages(
    [("system", COMPUTATIONAL_SYSTEM), ("human", HUMAN)])


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _computational_fallback(docs) -> str:
    pages = sorted({int(d.metadata["page"]) for d in docs})[:3]
    top = docs[0].metadata
    return (
        f"This asks for a worked numeric example. The book works it through in "
        f"{top['section_title']} (Section {top['section']}), around page(s) "
        f"{', '.join(map(str, pages))}. I can't reproduce the numbers reliably "
        f"here because the equations don't extract cleanly from the PDF, and "
        f"reconstructing them risks showing you wrong arithmetic. Open those "
        f"pages for the actual calculation. I can explain the method in words "
        f"if that would help."
    )


# --- Nodes ---

def retrieve(state: GraphState) -> dict:
    scored = _store.similarity_search_with_score(
        state["question"], k=config.RETRIEVE_K)
    docs = [d for d, _ in scored]
    top = max((s for _, s in scored), default=0.0)
    return {"docs": docs, "top_score": top}


def grade(state: GraphState) -> dict:
    context = format_context(state["docs"])
    reply = _invoke_with_retry(_grade_llm, _grade_prompt.format_messages(
        question=state["question"], context=context)).content
    verdict = _extract_json(reply) or {}
    # Fail toward answering (relevant, conceptual) if the grader output is
    # unparseable, which is rare with the 120B model.
    return {"relevant": bool(verdict.get("relevant", True)),
            "computational": bool(verdict.get("computational", False))}


def generate_conceptual(state: GraphState) -> dict:
    context = format_context(state["docs"])
    answer = _invoke_with_retry(_gen_llm, _gen_prompt.format_messages(
        context=context, question=state["question"])).content
    return {"answer": answer, "route": "conceptual"}


def generate_computational(state: GraphState) -> dict:
    context = format_context(state["docs"])
    answer = _invoke_with_retry(_gen_llm, _comp_prompt.format_messages(
        context=context, question=state["question"])).content
    if contains_worked_numbers(answer):
        # The model emitted numbers despite the constraint. Replace with the
        # safe page-pointer so no fabricated arithmetic ever reaches the student.
        return {"answer": _computational_fallback(state["docs"]),
                "route": "computational_guarded"}
    return {"answer": answer, "route": "computational"}


def refuse(state: GraphState) -> dict:
    if state["top_score"] < config.SCORE_GATE_THRESHOLD:
        return {"answer": (
            "I don't find this topic in the course book. This tool only covers "
            "the Linear Algebra course material: systems of equations, matrices, "
            "vector spaces, linear and affine mappings, analytical geometry, and "
            "matrix decomposition."), "route": "gate_refuse"}
    top = state["docs"][0].metadata
    return {"answer": (
        f"I don't find this specific topic in the course book. The nearest "
        f"material is {top['section_title']} (Section {top['section']}, "
        f"p.{int(top['page'])}), but it doesn't address your question."),
        "route": "grade_refuse"}


# --- Routing ---

def route_after_retrieve(state: GraphState) -> str:
    return "refuse" if state["top_score"] < config.SCORE_GATE_THRESHOLD else "grade"


def route_after_grade(state: GraphState) -> str:
    if not state["relevant"]:
        return "refuse"
    return "computational" if state["computational"] else "conceptual"


# --- Assemble ---

_builder = StateGraph(GraphState)
_builder.add_node("retrieve", retrieve)
_builder.add_node("grade", grade)
_builder.add_node("generate_conceptual", generate_conceptual)
_builder.add_node("generate_computational", generate_computational)
_builder.add_node("refuse", refuse)

_builder.add_edge(START, "retrieve")
_builder.add_conditional_edges("retrieve", route_after_retrieve,
                               {"refuse": "refuse", "grade": "grade"})
_builder.add_conditional_edges("grade", route_after_grade,
                               {"refuse": "refuse",
                                "computational": "generate_computational",
                                "conceptual": "generate_conceptual"})
_builder.add_edge("generate_conceptual", END)
_builder.add_edge("generate_computational", END)
_builder.add_edge("refuse", END)

app = _builder.compile()


def ask(question: str) -> dict:
    """Run one question through the graph; returns the full final state."""
    return app.invoke({"question": question})


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the Linear Algebra course book (graph).")
    ap.add_argument("question", nargs="*")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    def run(q: str) -> None:
        state = ask(q)
        if args.verbose:
            pages = [int(d.metadata["page"]) for d in state["docs"]]
            print(f"\n[route={state['route']}  top_score={state['top_score']:.3f}"
                  f"  relevant={state.get('relevant')}"
                  f"  computational={state.get('computational')}"
                  f"  retrieved_pages={pages}]")
        print("\n" + state["answer"] + "\n")

    if args.question:
        run(" ".join(args.question))
        return
    print("Linear Algebra assistant (graph). Empty line or Ctrl-C to quit.\n")
    while True:
        try:
            q = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    main()