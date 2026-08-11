"""
Phase 1: naive retrieve-then-answer over the course book.

Retrieves the k most similar chunks from Pinecone, hands them to the LLM as
context, and asks it to answer using only those excerpts and to cite the
section and page it used.

This is deliberately the SIMPLE version. Notice its main weakness as you test
it: it always retrieves k chunks, even for a question the book does not cover,
and then leans entirely on the LLM's willingness to say "not covered." That
reliance is unreliable, and hardening it is the whole point of Phase 3.

    python src/chain.py "What is the rank of a matrix?"
    python src/chain.py -v "Explain linear independence"   # -v shows sources
    python src/chain.py                                     # interactive loop
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import config

load_dotenv()

SYSTEM = """You are a study assistant for an IU International University Linear \
Algebra course. Answer the student's question using ONLY the course-book \
excerpts provided in the context. Rules:

- Use only the excerpts. Do not add outside knowledge or invent facts.
- If the excerpts do not contain enough to answer, say plainly that the \
provided course-book excerpts do not cover this. Do not guess.
- Explain in clear, plain language aimed at understanding the concept.
- These excerpts come from a PDF where equations and matrices do not extract \
cleanly. If a complete answer needs a formula or a worked calculation, say so \
and point to the page number so the student can read it in the PDF directly.
- End with a line starting with "Sources:" listing the section and page of \
each excerpt you actually used."""

HUMAN = """Context excerpts from the course book:

{context}

Student question: {question}"""


def build_store() -> PineconeVectorStore:
    """Connect to the existing index/namespace for reading (no writes here)."""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    embeddings = PineconeEmbeddings(model=config.EMBED_MODEL)
    return PineconeVectorStore(
        index=pc.Index(config.INDEX_NAME),
        embedding=embeddings,
        namespace=config.NAMESPACE,
    )


def format_context(docs) -> str:
    """Render retrieved chunks with their citations for the prompt.

    Note the int() casts on unit and page: Pinecone returns numeric metadata as
    floats, so without this a citation would read 'page 60.0'.
    """
    blocks = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        cite = (f"Unit {int(m['unit'])}, Section {m['section']} "
                f"{m['section_title']}, page {int(m['page'])}")
        blocks.append(f"[Excerpt {i} - {cite}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def ask(question: str, k: int = config.RETRIEVE_K, verbose: bool = False):
    """Retrieve, then answer. Returns (answer_text, retrieved_docs)."""
    store = build_store()
    docs = store.similarity_search(question, k=k)

    if verbose:
        print(f"\nRetrieved {len(docs)} chunks:")
        for d in docs:
            m = d.metadata
            print(f"  - {m['section']} {m['section_title']} (p.{int(m['page'])})")
        print()

    llm = ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        reasoning_format="parsed",  # keep the model's scratch reasoning out of .content
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM), ("human", HUMAN)]
    )
    messages = prompt.format_messages(
        context=format_context(docs), question=question
    )
    response = llm.invoke(messages)
    return response.content, docs


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the Linear Algebra course book.")
    ap.add_argument("question", nargs="*", help="Your question. Omit for interactive mode.")
    ap.add_argument("-k", type=int, default=config.RETRIEVE_K, help="Chunks to retrieve.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Show retrieved sources.")
    args = ap.parse_args()

    if args.question:
        answer, _ = ask(" ".join(args.question), k=args.k, verbose=args.verbose)
        print("\n" + answer)
        return

    print("Linear Algebra study assistant. Empty line or Ctrl-C to quit.\n")
    while True:
        try:
            q = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        answer, _ = ask(q, k=args.k, verbose=args.verbose)
        print("\n" + answer + "\n")


if __name__ == "__main__":
    main()