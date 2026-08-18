"""
Phase 2/3: measure the pipeline. Targets the Phase 3 graph.

Fabrication is scored MECHANICALLY via guards.contains_worked_numbers, not by
the LLM judge, because we proved the judge cannot reliably detect it (it shares
the generator's garbled context). The judge is kept only for grounded, correct,
and refusal, where it is reliable.

    python src/evaluate.py --mode retrieval
    python src/evaluate.py --mode full --limit 5 --fresh
    python src/evaluate.py --mode full --fresh
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

import config
from llm import make_llm
from guards import contains_worked_numbers
from graph import ask as graph_ask

EVAL_FILE = Path("evals/questions.jsonl")
RESULTS_FILE = Path("evals/last_run.jsonl")

JUDGE_SYSTEM = """You evaluate a study assistant. You are given a question, the \
excerpts it was shown, and its answer.

Respond with ONLY a JSON object, no prose and no code fences, with exactly \
these three boolean keys:
  "attempts_answer": true if it gives a substantive answer, false if it \
declines or says the material does not cover the question.
  "grounded": true if every substantive claim is supported by the EXCERPTS, \
false otherwise. Judge only against the excerpts.
  "answer_is_correct": judged against YOUR OWN knowledge of linear algebra: \
true if the answer's factual claims are correct, or if it correctly declined \
an unanswerable question; false if it states anything mathematically wrong."""

JUDGE_HUMAN = """QUESTION:
{question}

EXCERPTS SHOWN:
{context}

ASSISTANT ANSWER:
{answer}

JSON:"""


def load_questions() -> list[dict]:
    return [json.loads(ln) for ln in EVAL_FILE.read_text().splitlines() if ln.strip()]


def load_checkpoint() -> dict[str, dict]:
    if not RESULTS_FILE.exists():
        return {}
    return {r["id"]: r for r in
            (json.loads(ln) for ln in RESULTS_FILE.read_text().splitlines() if ln.strip())}


def append_result(row: dict) -> None:
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(row) + "\n")


def invoke_with_retry(llm, messages, tries: int = 3, base_sleep: float = 4.0):
    for attempt in range(tries):
        try:
            return llm.invoke(messages)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "per day" in msg or "TPD" in msg or "tokens per day" in msg:
                raise RuntimeError(
                    "Daily token cap reached. Progress is checkpointed; "
                    "re-run after the cap resets to resume.") from e
            if ("Error code: 4" in msg) and ("429" not in msg):
                raise RuntimeError(f"Permanent request error, not retrying: {e}") from e
            if attempt == tries - 1:
                raise
            wait = base_sleep * (attempt + 1)
            print(f"    (transient failure: {e}; retrying in {wait:.0f}s)")
            time.sleep(wait)


def parse_judge(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def retrieval_score(expected_pages, docs):
    pages = [int(d.metadata["page"]) for d in docs]
    hit = any(p in expected_pages for p in pages)
    rank = next((i for i, p in enumerate(pages, 1) if p in expected_pages), None)
    return hit, rank, pages


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the graph pipeline.")
    ap.add_argument("--mode", choices=["retrieval", "full"], default="full")
    ap.add_argument("-k", type=int, default=config.RETRIEVE_K)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]

    if args.fresh and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
    done = load_checkpoint()
    if done:
        print(f"Resuming: {len(done)} already done, skipping.\n")

    judge_llm = judge_prompt = None
    if args.mode == "full":
        judge_llm = make_llm(config.EVAL_JUDGE_MODEL, 0.0)
        judge_prompt = ChatPromptTemplate.from_messages(
            [("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])

    try:
        for q in questions:
            if q["id"] in done:
                continue

            if args.mode == "full":
                # Run the whole graph. It retrieves internally, so we read the
                # docs and route back out of the returned state.
                state = graph_ask(q["question"])
                docs = state.get("docs", [])
                top_score = state.get("top_score", 0.0)
                answer = state["answer"]
                route = state.get("route", "?")
                hit, rank, pages = retrieval_score(q.get("expected_pages", []), docs)

                # Mechanical fabrication check (not the judge).
                fabricated = contains_worked_numbers(answer)

                # Judge only grounded / correct / attempts_answer.
                context = "\n\n".join(d.page_content for d in docs)
                verdict = parse_judge(invoke_with_retry(
                    judge_llm, judge_prompt.format_messages(
                        question=q["question"], context=context, answer=answer)).content)

                row = {"id": q["id"], "type": q["type"], "question": q["question"],
                       "answerable": q["answerable"], "top_score": round(top_score, 4),
                       "retrieved_pages": pages, "hit": hit, "rank": rank,
                       "route": route, "fabricated": fabricated,
                       "answer": answer, "judge": verdict}
                print(f"  {q['id']} {q['type']:18s} route={route:22s} "
                      f"hit={hit!s:5s} fab={fabricated!s:5s} judge={verdict}")
                time.sleep(args.sleep)
            else:
                # Retrieval-only: call the store directly for speed, no LLM.
                from graph import _store
                scored = _store.similarity_search_with_score(q["question"], k=args.k)
                docs = [d for d, _ in scored]
                top_score = max((s for _, s in scored), default=0.0)
                hit, rank, pages = retrieval_score(q.get("expected_pages", []), docs)
                row = {"id": q["id"], "type": q["type"], "question": q["question"],
                       "answerable": q["answerable"], "top_score": round(top_score, 4),
                       "retrieved_pages": pages, "hit": hit, "rank": rank}
                print(f"  {q['id']} {q['type']:18s} hit={hit!s:5s} "
                      f"rank={rank} top_score={top_score:.3f}")

            append_result(row)
            done[q["id"]] = row
    except RuntimeError as e:
        print(f"\n{e}")

    report(list(done.values()), args.mode, args.k)
    print(f"\nPer-question results in {RESULTS_FILE}")


def report(results, mode, k):
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    answerable = [r for r in results if r["answerable"]]
    if not answerable:
        print("\n(no results yet)")
        return

    print("\n" + "=" * 60)
    print(f"RESULTS  (k={k}, mode={mode}, {len(results)} questions)")
    print("=" * 60)

    hits = [r for r in answerable if r["hit"]]
    mrr = sum((1.0 / r["rank"]) for r in answerable if r["rank"]) / len(answerable)
    print(f"\nRETRIEVAL (over {len(answerable)} answerable)")
    print(f"  Hit@{k}: {len(hits)}/{len(answerable)} = {len(hits)/len(answerable):.0%}")
    print(f"  MRR:    {mrr:.3f}")

    if mode != "full":
        return

    # Route breakdown: which node handled each question type.
    print("\nROUTE BREAKDOWN")
    for t in sorted(by_type):
        routes = defaultdict(int)
        for r in by_type[t]:
            routes[r.get("route", "?")] += 1
        summary = ", ".join(f"{rt}:{n}" for rt, n in sorted(routes.items()))
        print(f"  {t:18s}: {summary}")

    def rate(rows, key, want=True, judged=True):
        pool = [r for r in rows if r.get("judge")] if judged else rows
        if not pool:
            return None, 0
        return sum(1 for r in pool if (r["judge"].get(key) if judged else r.get(key)) == want) / len(pool), len(pool)

    concept = by_type.get("conceptual", [])
    gr, n = rate(concept, "grounded", True)
    co, _ = rate(concept, "answer_is_correct", True)
    if n:
        print(f"\nCONCEPTUAL ({n} graded)")
        print(f"  grounded (in excerpts): {gr:.0%}")
        print(f"  correct (factually):    {co:.0%}")

    # Fabrication: mechanical, over computational questions.
    comp = by_type.get("computational", [])
    if comp:
        fab = sum(1 for r in comp if r.get("fabricated")) / len(comp)
        print(f"\nCOMPUTATIONAL FABRICATION ({len(comp)}, mechanical check)")
        print(f"  fabricated numbers: {fab:.0%}  (want 0%)")
        for r in comp:
            if r.get("fabricated"):
                print(f"    FABRICATED -> {r['id']}: {r['question']}")

    for t, label in [("out_of_scope_far", "FAR"), ("out_of_scope_near", "NEAR")]:
        rows = by_type.get(t, [])
        ref, n = rate(rows, "attempts_answer", want=False)
        if n:
            print(f"\nREFUSAL {label} ({n} graded)")
            print(f"  correctly refused: {ref:.0%}")
            for r in rows:
                if r.get("judge") and r["judge"].get("attempts_answer"):
                    print(f"    FAILED TO REFUSE -> {r['id']}: {r['question']}")


if __name__ == "__main__":
    main()