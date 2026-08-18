"""
Phase 2: measure retrieval accuracy and answer faithfulness.

Resumable and quota-aware. Each question's result is written to disk as it
completes; re-running skips questions already done. Use --fresh whenever you
change the questions or the eval models, so stale checkpoints are discarded.

    python src/evaluate.py --mode retrieval
    python src/evaluate.py --mode full --limit 5 --fresh   # cheap smoke test
    python src/evaluate.py --mode full --fresh             # full run
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

# Only reasoning models accept reasoning_format. Applying it to a plain model
# (e.g. llama-3.1-8b-instant) returns a 400. Keep the list of reasoning-model
# name fragments here so we set the parameter only where it is valid.
REASONING_MODEL_HINTS = ("gpt-oss", "qwen3", "deepseek-r1")


def make_llm(model: str, temperature: float) -> ChatGroq:
    """Build a ChatGroq, passing reasoning_format only to reasoning models."""
    kwargs = {"model": model, "temperature": temperature}
    if any(hint in model for hint in REASONING_MODEL_HINTS):
        kwargs["reasoning_format"] = "hidden"
    return ChatGroq(**kwargs)


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
    return [json.loads(ln) for ln in EVAL_FILE.read_text().splitlines() if ln.strip()]


def load_checkpoint() -> dict[str, dict]:
    if not RESULTS_FILE.exists():
        return {}
    done = {}
    for ln in RESULTS_FILE.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            done[r["id"]] = r
    return done


def append_result(row: dict) -> None:
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(row) + "\n")


def invoke_with_retry(llm, messages, tries: int = 3, base_sleep: float = 4.0):
    """Retry only TRANSIENT failures (per-minute rate limits, network blips).

    Give up immediately on:
      - a daily token cap, which will not clear in any retry window
      - any other 4xx client error (400 bad request, 401 auth), which is
        permanent: the request is malformed and retrying cannot help.
    """
    for attempt in range(tries):
        try:
            return llm.invoke(messages)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "per day" in msg or "TPD" in msg or "tokens per day" in msg:
                raise RuntimeError(
                    "Daily token cap reached. Progress is checkpointed; "
                    "re-run after the cap resets to resume."
                ) from e
            # Permanent client errors (except transient 429 rate limits).
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
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
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
        print(f"Resuming: {len(done)} question(s) already done, skipping them.\n")

    store = build_store()
    gen_llm = judge_llm = gen_prompt = judge_prompt = None
    if args.mode == "full":
        gen_llm = make_llm(config.EVAL_GEN_MODEL, config.LLM_TEMPERATURE)
        judge_llm = make_llm(config.EVAL_JUDGE_MODEL, 0)
        gen_prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM), ("human", HUMAN)])
        judge_prompt = ChatPromptTemplate.from_messages(
            [("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])

    try:
        for q in questions:
            if q["id"] in done:
                continue
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
        return sum(1 for r in graded if r["judge"].get(key) == want) / len(graded), len(graded)

    concept = by_type.get("conceptual", [])
    gr, n = rate(concept, "grounded", True)
    co, _ = rate(concept, "answer_is_correct", True)
    if n:
        print(f"\nCONCEPTUAL ({n} graded)")
        print(f"  grounded (in excerpts): {gr:.0%}")
        print(f"  correct (factually):    {co:.0%}")
        comp_ids = [r["id"] for r in concept if r.get("judge")
                    and r["judge"].get("answer_is_correct")
                    and not r["judge"].get("grounded")]
        if comp_ids:
            print(f"  ungrounded-but-correct: {', '.join(comp_ids)}")

    comp = by_type.get("computational", [])
    fab, n = rate(comp, "introduces_unsupported_numbers", True)
    cco, _ = rate(comp, "answer_is_correct", True)
    if n:
        print(f"\nCOMPUTATIONAL FABRICATION ({n} graded)")
        print(f"  fabricated numbers: {fab:.0%}  (want 0%)")
        print(f"  factually correct:  {cco:.0%}")
        for r in comp:
            if r.get("judge") and r["judge"].get("introduces_unsupported_numbers"):
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