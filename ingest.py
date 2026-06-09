"""
ingest.py
---------
Loads sample documents, splits them into chunks,
generates embeddings using HuggingFace, and stores in FAISS.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_PATH    = "sample_docs.md"
FAISS_PATH   = "faiss_index"
EMBED_MODEL  = "all-MiniLM-L6-v2"   # ~80MB, auto-downloaded from HuggingFace
CHUNK_SIZE   = 300
CHUNK_OVERLAP = 50
# ──────────────────────────────────────────────────────────────────────────────


def ingest():
    print("📄 Loading documents...")
    loader = TextLoader(DOCS_PATH, encoding="utf-8")
    documents = loader.load()

    print("✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"   → {len(chunks)} chunks created")

    print("🤗 Loading HuggingFace embedding model (downloads on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

    print("📦 Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"💾 Saving FAISS index to '{FAISS_PATH}'...")
    vectorstore.save_local(FAISS_PATH)

    print("✅ Ingestion complete! Vector store is ready.")
    return vectorstore


if __name__ == "__main__":
    ingest()