"""
Phase 0, part 2: embed the tagged chunks and upsert them into Pinecone.

Run this once (or after changing the chunking). It is not part of the query
path. Because chunk IDs are deterministic, re-running overwrites vectors in
place instead of creating duplicates. Use --reset to wipe the namespace first
if the number of chunks has changed and you want no stale vectors left behind.

    python src/ingest.py            # create index if needed, embed, upsert
    python src/ingest.py --reset    # clear the namespace, then re-ingest
"""

from __future__ import annotations

import argparse
import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

import config
from chunking import load_chunks

load_dotenv()


def get_index(pc: Pinecone):
    """Return the Pinecone index, creating it at the right dimension if absent.

    Guards against the classic footgun: an existing index built for a
    different embedding model (wrong dimension). We refuse rather than fail
    cryptically deep inside the upsert.
    """
    if not pc.has_index(config.INDEX_NAME):
        print(f"Creating index '{config.INDEX_NAME}' "
              f"(dim={config.EMBED_DIMENSION}, metric={config.METRIC}) ...")
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBED_DIMENSION,
            metric=config.METRIC,
            spec=ServerlessSpec(cloud=config.CLOUD, region=config.REGION),
        )
        while not pc.describe_index(config.INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("Index ready.")
    else:
        desc = pc.describe_index(config.INDEX_NAME)
        if desc.dimension != config.EMBED_DIMENSION:
            raise SystemExit(
                f"Index '{config.INDEX_NAME}' has dimension {desc.dimension}, "
                f"but the model needs {config.EMBED_DIMENSION}. Delete the "
                f"index in the Pinecone console (or change INDEX_NAME in "
                f"config.py) and re-run."
            )
        print(f"Using existing index '{config.INDEX_NAME}'.")
    return pc.Index(config.INDEX_NAME)


def main(reset: bool) -> None:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise SystemExit("PINECONE_API_KEY is not set. Fill it into .env.")

    print("Loading and chunking the course book ...")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks ready.\n")

    pc = Pinecone(api_key=api_key)
    index = get_index(pc)

    if reset:
        print(f"Clearing namespace '{config.NAMESPACE}' ...")
        try:
            index.delete(delete_all=True, namespace=config.NAMESPACE)
        except Exception as e:
            # A brand-new namespace has nothing to delete; that is fine.
            print(f"  (nothing to clear: {e})")

    embeddings = PineconeEmbeddings(model=config.EMBED_MODEL)
    store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=config.NAMESPACE,
    )

    print(f"Embedding + upserting {len(chunks)} chunks into namespace "
          f"'{config.NAMESPACE}' ...")
    store.add_documents(documents=chunks, ids=[c.id for c in chunks])
    print("  Upsert complete.")

    # Serverless indexing lags a few seconds after upsert; wait before peeking.
    time.sleep(5)
    stats = index.describe_index_stats()
    total = getattr(stats, "total_vector_count", None)
    print(f"\nTotal vectors in index: {total}")

    # End-to-end smoke test: one real query through the full path.
    print("\nSmoke test -> 'What is the rank of a matrix?'")
    for d in store.similarity_search("What is the rank of a matrix?", k=3):
        m = d.metadata
        preview = d.page_content[:90].replace("\n", " ").strip()
        print(f"  [{m['section']} {m['section_title']} p.{m['page']}] {preview}...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Embed and upsert course-book chunks.")
    ap.add_argument("--reset", action="store_true",
                    help="Delete the namespace before ingesting.")
    args = ap.parse_args()
    main(reset=args.reset)