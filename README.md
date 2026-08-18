# linalg-rag

A corrective RAG system that answers questions about a Linear Algebra course
book, cites the section and page it used, refuses questions the book does not
cover, and does not fabricate worked calculations.

Built to learn LangChain and LangGraph properly by measuring every change
instead of trusting it. The eval harness is the real deliverable; the graph is
what it measures.

## The short version

A naive retrieve-then-answer chain over this course book fabricated numbers on
about 80% of computational questions: asked for a worked matrix subtraction, it
invented confident, wrong arithmetic and attributed it to the textbook. The
cause is that the book's equations do not extract cleanly from the PDF, so the
retrieved context is full of corrupted numbers that lure the model into
reconstructing them.

Rather than patch that on instinct, the project builds an evaluation harness
first, then a LangGraph pipeline whose every node targets a measured failure.
The result, measured against a locked baseline:

| Axis | Naive chain | Graph |
|---|---|---|
| Retrieval Hit@5 | 100% | 100% |
| Conceptual answers factually correct | 94% | 94% |
| Computational answers that fabricate numbers | ~80% | 0% |
| Out-of-scope (far) correctly refused | 100% | 100% |
| Out-of-scope (near) correctly refused | leaked | 100% |

Fabrication is scored by a deterministic check, not an LLM judge, because the
project found the judge could not reliably detect it (the judge sees the same
garbled context the generator does).

## How it works

Ingestion (`ingest.py`) parses the 154-page PDF, tags each chunk with its unit,
section, and page from a hand-verified structure map, embeds with Pinecone's
hosted `multilingual-e5-large`, and upserts to a Pinecone index.

Answering is a LangGraph state graph (`graph.py`):

    retrieve
      |-- top similarity < 0.82 --------------------> refuse   (far out-of-scope)
      `-- else --> grade (LLM: relevant? computational?)
                     |-- not relevant --------------> refuse   (near out-of-scope)
                     |-- relevant + computational --> constrained answer + guard
                     `-- relevant + conceptual -----> normal grounded answer

Each branch exists because a measured failure needed it:

- The **score gate** refuses far out-of-scope questions before any LLM call,
  using a 0.82 threshold derived from the eval (every in-scope question scored
  at least 0.839; every far one at most 0.810).
- The **relevance grader** catches near out-of-scope questions that score above
  the gate but whose retrieved chunks are about an adjacent topic (asking for
  QR decomposition retrieves Cholesky content). It judges topic match, not
  completeness, so garbled-but-on-topic answers still pass.
- The **computational router** sends worked-example questions to a constrained
  path that explains the method and points to the page, forbidden from emitting
  numbers. A deterministic guard (`guards.py`) checks the output and replaces it
  with a page pointer if any worked numbers slipped through, so no fabricated
  arithmetic can reach the reader.
- The **conceptual path** is left untouched, because it already answered 94%
  correctly.

## Evaluation

`evaluate.py` runs a hand-written, page-labeled question set
(`evals/questions.jsonl`, 30 questions across conceptual, computational, and two
kinds of out-of-scope) through the graph and scores four axes: retrieval
(Hit@k, MRR), conceptual grounding and correctness (LLM judge), computational
fabrication (deterministic), and refusal (LLM judge). Results checkpoint per
question so a rate-limit interruption resumes instead of restarting.

    python src/evaluate.py --mode retrieval          # fast, no LLM calls
    python src/evaluate.py --mode full --fresh        # full run through the graph

## Stack

- LangChain 1.x, LangGraph 1.x
- Pinecone serverless (hosted `multilingual-e5-large`, 1024-dim, cosine)
- Groq (`openai/gpt-oss-120b`)
- pdfplumber for parsing, LangSmith for tracing

## Setup

Requires Python 3.10+.

    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # add Pinecone, Groq, LangSmith keys
    # place the course book at data/course_book.pdf (gitignored)
    python src/ingest.py        # build the index
    python src/graph.py "What is the rank of a matrix?"

## Repository layout

    src/
      config.py       shared constants (index, models, threshold)
      chunking.py     PDF parsing, structure tagging, chunking
      ingest.py       embed + upsert to Pinecone
      chain.py        naive baseline chain (Phase 1)
      evaluate.py     the eval harness
      llm.py          ChatGroq factory
      guards.py       deterministic fabrication check
      graph.py        the corrective RAG graph (Phase 3)
    evals/
      questions.jsonl labeled eval set

## Known limitations

- The book's equations do not extract cleanly from the PDF. This is the root
  cause of everything above. The honest fix is vision OCR of the equation
  regions (not built here); the guard is a mitigation, not a cure.
- The constrained computational path is blunt: on some worked-example questions
  it defers to a page pointer rather than engaging, trading helpfulness for
  safety. Correct for a study tool, but a real limitation.
- One formula-heavy conceptual question (Cholesky) can still pull the model into
  emitting bad numbers, because the guard runs only on the computational path.

## What this book is

IU International University course book DLBDSMFLA01 (Linear Algebra). It is
copyrighted material and is not included in this repository; `data/` is
gitignored. This tool is for personal study only.