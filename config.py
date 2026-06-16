"""Configuration settings for Medical RAG Summarizer."""

import os
from pathlib import Path

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Retrieval configuration
TOP_K_RETRIEVAL = 5

# LLM configuration
LLM_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

# File paths
FAISS_INDEX_PATH = "./faiss_index.bin"
METADATA_PATH = "./metadata.json"
DOCS_FOLDER = "./sample_docs"

# System prompt for LLM
SYSTEM_PROMPT = """You are a medical expert assistant. Summarize the following medical context clearly and concisely for a healthcare professional. Focus on key findings, diagnoses, medications, and recommendations."""

# Ensure paths exist
Path(DOCS_FOLDER).mkdir(exist_ok=True)
