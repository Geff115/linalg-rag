# linalg-rag

A retrieval-augmented question-answering system over the IU *Mathematics:
Linear Algebra* (DLBDSMFLA01) course book. Ask a question in plain language,
get an answer grounded in the book, cited back to the unit, section, and page.

Built as a learning project to understand RAG architecture end to end:
document processing, embeddings, vector search, and agentic control flow.

## What it does

- Answers conceptual questions about linear algebra from the course book
- Cites the unit, section, and page each answer draws from
- Says "this isn't covered in the book" instead of inventing an answer

## Known limitation (by design)

The course book typesets equations in a custom font that does not survive
text extraction cleanly. This system answers **conceptual** questions well
(definitions, explanations, "where is X covered"). It does **not** reproduce
worked numerical examples or render matrices. For those, it points you to the
page number so you can open the PDF. Handling the math is a later phase.

## Architecture

The project is built in four phases, each a separate script:

| Phase | Script         | Purpose                                          |
|-------|----------------|--------------------------------------------------|
| 0     | `ingest.py`    | Parse, chunk, tag, embed, upsert to Pinecone     |
| 1     | `chain.py`     | Naive retrieve-then-answer                        |
| 2     | `evaluate.py`  | Hand-written eval set + retrieval scoring         |
| 3     | `graph.py`     | LangGraph: grade retrieval, retry, refuse         |

Ingestion is deliberately separate from querying. Run `ingest.py` once
(or when the book changes); it is not part of the query path.

## Stack

- **Framework:** LangChain 1.x, LangGraph 1.x
- **Vector DB:** Pinecone (serverless)
- **Embeddings:** Pinecone-hosted `multilingual-e5-large` (1024 dimensions)
- **LLM:** Groq, Llama 3.3 70B (`llama-3.3-70b-versatile`)
- **Tracing:** LangSmith

## Setup

Requires Python 3.10 or newer.

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd linalg-rag

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Configure keys
cp .env.example .env
# then edit .env with your Pinecone, Groq, and LangSmith keys

# 5. Add the course book (gitignored, not committed)
#    place the PDF at: data/course_book.pdf

# 6. Build the index (Phase 0)
python src/ingest.py
```

The Pinecone index is created automatically by `ingest.py` with **1024
dimensions** and **cosine** metric, matching the embedding model. If you ever
change embedding models, you must delete and recreate the index, because
dimension is fixed at creation.

## Data and copyright

The course book is IU copyrighted material. It stays out of version control
(`data/` is gitignored) and this project is for personal study only. Do not
deploy it publicly or redistribute the book's contents.

## Status

- [ ] Phase 0: ingestion
- [ ] Phase 1: naive RAG
- [ ] Phase 2: evaluation
- [ ] Phase 3: corrective RAG with LangGraph