"""Shared configuration for ingestion and retrieval.

Both ingest.py and chain.py import from here so they always agree on which
index, namespace, and models to use. Centralizing this also means that when a
model name is deprecated (which happens often), you change one line here.
"""

# --- Vector store ---
INDEX_NAME = "linalg-rag"
NAMESPACE = "linalg-v1"

# --- Embeddings (hosted by Pinecone; fixes index dimension at 1024) ---
EMBED_MODEL = "multilingual-e5-large"
EMBED_DIMENSION = 1024
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"

# --- LLM (served by Groq) ---
# Groq deprecated llama-3.3-70b-versatile for free-tier use in June 2026.
# openai/gpt-oss-120b is their recommended replacement. If this string is ever
# rejected, check https://console.groq.com/docs/models and update it here.
LLM_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0          # deterministic answers for a study tool

# --- Retrieval ---
RETRIEVE_K = 5               # how many chunks to pull per question