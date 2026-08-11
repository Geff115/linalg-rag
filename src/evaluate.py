"""
Phase 2: measure retrieval accuracy and answer faithfulness.

Scores four behaviors separately:
  - retrieval:    Hit@k and MRR against hand-labeled expected pages
  - faithfulness: grounded (supported by context) AND correct (factually right)
  - fabrication:  do computational answers invent numbers instead of deferring
  - refusal:      do out-of-scope questions get correctly declined

Grounded and correct are deliberately separate. An answer can be correct but
ungrounded (the model filled a gap from training knowledge) which is acceptable
for stable concepts, or ungrounded and incorrect (fabrication) which is not.

    python src/evaluate.py --mode retrieval
    python src/evaluate.py --mode full
    python src/evaluate.py --mode full -k 5 --sleep 1.5
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import config
from chain import SYSTEM, HUMAN, format_context, build_store

EVAL_FILE = Path("evals/questions.jsonl")
RESULTS_FILE = Path("evals/last_run.jsonl")

JUDGE_SYSTEM = """You evaluate a study assistant. You are given a question, the \
exact course-book excerpts the assistant was shown, and its answer.

Respond with ONLY a JSON object, no prose and no code fences, with exactly \
these four boolean keys:
  "attempts_answer": true if the assistant gives a substantive answer, false \
if it declines or says the material does not cover the question.
  "grounded": true if every substantive claim is supported by the EXCERPTS \
provided, false otherwise. Judge this only against the excerpts.
  "introduces_unsupported_numbers": true if the answer states specific numeric \
results (matrix or vector entries, computed values) not clearly present in the \
excerpts, false otherwise.
  "answer_is_correct": judged against YOUR OWN knowledge of linear algebra, not \
the excerpts: true if the answer's factual claims are correct, or if the \
assistant correctly declined an unanswerable question; false if it states \
anything mathematically wrong."""

JUDGE_HUMAN = """QUESTION:
{question}

EXCERPTS SHOWN TO THE ASSISTANT:
{context}

ASSISTANT ANSWER:
{answer}

JSON:"""


def load_questions() -> list[dict]:
    lines = EVAL_FILE.read_text().splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def invoke_with_retry(llm, messages, tries: int = 3, base_sleep: float = 4.0):
    for attempt in range(tries):
        try:
            return llm.invoke(messages)
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = base_sleep * (attempt + 1)
            print(f"    (LLM call failed: {e}; retrying in {wait:.0f}s)")
            time.sleep(wait)


def parse_judge(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def retrieval_score(expected_pages: list[int], docs) -> tuple[bool, int | None, list[int]]:
    pages = [int(d.metadata["page"]) for d in docs]
    hit = any(p in expected_pages for p in pages)
    rank = next((i for i, p in enumerate(pages, 1) if p in expected_pages), None)
    return hit, rank, pages


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    ap.add_argument("--mode", choices=["retrieval", "full"], default="full")
    ap.add_argument("-k", type=int, default=config.RETRIEVE_K)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    questions = load_questions()
    store = build_store()

    gen_llm = judge_llm = gen_prompt = judge_prompt = None
    if args.mode == "full":
        gen_llm = ChatGroq(model=config.LLM_MODEL,
                           temperature=config.LLM_TEMPERATURE,
                           reasoning_format="parsed")
        judge_llm = ChatGroq(model=config.LLM_MODEL, temperature=0,
                             reasoning_format="parsed")
        gen_prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM), ("human", HUMAN)])
        judge_prompt = ChatPromptTemplate.from_messages(
            [("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])

    results = []
    for q in questions:
        docs_scored = store.similarity_search_with_score(q["question"], k=args.k)
        docs = [d for d, _ in docs_scored]
        top_score = max((s for _, s in docs_scored), default=0.0)
        hit, rank, pages = retrieval_score(q.get("expected_pages", []), docs)

        row = {"id": q["id"], "type": q["type"], "question": q["question"],
               "answerable": q["answerable"], "top_score": round(top_score, 4),
               "retrieved_pages": pages, "hit": hit, "rank": rank}

        if args.mode == "full":
            context = format_context(docs)
            answer = invoke_with_retry(
                gen_llm, gen_prompt.format_messages(
                    context=context, question=q["question"])).content
            verdict = parse_judge(invoke_with_retry(
                judge_llm, judge_prompt.format_messages(
                    question=q["question"], context=context, answer=answer)).content)
            row["answer"] = answer
            row["judge"] = verdict
            print(f"  {q['id']} {q['type']:18s} hit={hit!s:5s} judge={verdict}")
            time.sleep(args.sleep)
        else:
            print(f"  {q['id']} {q['type']:18s} hit={hit!s:5s} "
                  f"rank={rank} top_score={top_score:.3f}")

        results.append(row)

    RESULTS_FILE.write_text("\n".join(json.dumps(r) for r in results))
    report(results, args.mode, args.k)
    print(f"\nPer-question results written to {RESULTS_FILE}")


def report(results: list[dict], mode: str, k: int) -> None:
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    answerable = [r for r in results if r["answerable"]]

    print("\n" + "=" * 60)
    print(f"RESULTS  (k={k}, mode={mode})")
    print("=" * 60)

    hits = [r for r in answerable if r["hit"]]
    mrr = sum((1.0 / r["rank"]) for r in answerable if r["rank"]) / len(answerable)
    print(f"\nRETRIEVAL (over {len(answerable)} answerable questions)")
    print(f"  Hit@{k}: {len(hits)}/{len(answerable)} = {len(hits)/len(answerable):.0%}")
    print(f"  MRR:    {mrr:.3f}")

    print("\nMEAN TOP SIMILARITY SCORE BY TYPE")
    for t in sorted(by_type):
        scores = [r["top_score"] for r in by_type[t]]
        print(f"  {t:18s}: {sum(scores)/len(scores):.3f}")

    if mode != "full":
        return

    def rate(rows, key, want=True):
        graded = [r for r in rows if r.get("judge")]
        if not graded:
            return None, 0
        good = sum(1 for r in graded if r["judge"].get(key) == want)
        return good / len(graded), len(graded)

    # Conceptual: grounded vs correct, and the gap between them.
    concept = by_type.get("conceptual", [])
    gr, n = rate(concept, "grounded", True)
    co, _ = rate(concept, "answer_is_correct", True)
    print(f"\nCONCEPTUAL ({n} graded)")
    print(f"  grounded (in excerpts): {gr:.0%}" if gr is not None else "  (none)")
    print(f"  correct (factually):    {co:.0%}" if co is not None else "")
    completion = [r["id"] for r in concept if r.get("judge")
                  and r["judge"].get("answer_is_correct")
                  and not r["judge"].get("grounded")]
    if completion:
        print(f"  ungrounded-but-correct (helpful completion): {', '.join(completion)}")

    # Computational fabrication, cross-checked by correctness.
    comp = by_type.get("computational", [])
    fab, n = rate(comp, "introduces_unsupported_numbers", True)
    cco, _ = rate(comp, "answer_is_correct", True)
    print(f"\nCOMPUTATIONAL FABRICATION ({n} graded)")
    print(f"  fabricated numbers: {fab:.0%}  (want 0%)" if fab is not None else "  (none)")
    print(f"  factually correct:  {cco:.0%}" if cco is not None else "")
    for r in comp:
        if r.get("judge") and r["judge"].get("introduces_unsupported_numbers"):
            print(f"    FABRICATED -> {r['id']}: {r['question']}")

    # Refusal, split far vs near.
    for t, label in [("out_of_scope_far", "FAR"), ("out_of_scope_near", "NEAR")]:
        rows = by_type.get(t, [])
        ref, n = rate(rows, "attempts_answer", want=False)
        print(f"\nREFUSAL {label} ({n} graded)")
        print(f"  correctly refused: {ref:.0%}" if ref is not None else "  (none)")
        for r in rows:
            if r.get("judge") and r["judge"].get("attempts_answer"):
                print(f"    FAILED TO REFUSE -> {r['id']}: {r['question']}")


if __name__ == "__main__":
    main()