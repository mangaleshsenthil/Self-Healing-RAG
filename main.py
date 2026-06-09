"""
main.py
-------
Entry point for the Self-Healing RAG pipeline.

Usage:
    # First time only — build the FAISS index:
    python main.py --ingest

    # Ask a question:
    python main.py --query "What is machine learning?"

    # Interactive mode:
    python main.py
"""

import argparse
import os
from ingest import ingest
from rag_graph import run_query


def main():
    parser = argparse.ArgumentParser(description="Self-Healing RAG Pipeline")
    parser.add_argument("--ingest", action="store_true",
                        help="Ingest documents and build FAISS index")
    parser.add_argument("--query", type=str,
                        help="Ask a single question and exit")
    args = parser.parse_args()

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    if args.ingest:
        ingest()
        return

    # ── Check FAISS index exists ──────────────────────────────────────────────
    if not os.path.exists("faiss_index"):
        print("⚠️  FAISS index not found. Running ingestion first...\n")
        ingest()

    # ── Step 2: Single query mode ─────────────────────────────────────────────
    if args.query:
        run_query(args.query)
        return

    # ── Step 3: Interactive mode ──────────────────────────────────────────────
    print("\n🧠 Self-Healing RAG — Interactive Mode")
    print("   Type your question and press Enter.")
    print("   Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        run_query(question)


if __name__ == "__main__":
    main()
