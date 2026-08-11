"""Shared configuration for ingestion and retrieval.

Both ingest.py and the Phase 1 retriever import from here so they always
agree on which index, namespace, and embedding model to use. A mismatch
between writer and reader is a common cause of "retrieval returns nothing".
"""

# Pinecone index (created automatically by ingest.py if missing).
INDEX_NAME = "linalg-rag"

# Namespace keeps experiments isolated inside one index. Change this (not the
# index) when you want a clean slate for a different chunking strategy.
NAMESPACE = "linalg-v1"

# Embedding model. Hosted by Pinecone, so no local model download and no
# OpenAI key. This model fixes the index dimension at 1024.
EMBED_MODEL = "multilingual-e5-large"
EMBED_DIMENSION = 1024
METRIC = "cosine"

# Serverless location. aws / us-east-1 is the Pinecone free-tier default.
CLOUD = "aws"
REGION = "us-east-1"