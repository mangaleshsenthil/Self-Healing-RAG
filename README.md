# 🧠 Self-Healing RAG Pipeline & Agentic Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=Ollama&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" />
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" />
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" />
</p>

An advanced Retrieval-Augmented Generation (RAG) system and Agentic Coding Assistant that critiques its own output and self-heals. Built with LangGraph, FastAPI, FAISS, and local LLMs (Ollama + HuggingFace).

## ✨ Features

- **Self-Healing Q&A (RAG)**: Retrieves context, generates an answer, and runs a strict critique. If the answer hallucinates or lacks context, it rewrites the search query and tries again.
- **Self-Healing Code Execution**: Generates Python code, executes it in a sandboxed subprocess, critiques the `stdout`/`stderr`, and iteratively fixes any bugs until the code passes.
- **Smart Intent Routing**: Automatically detects whether your prompt requires the Q&A pipeline or the Code Execution pipeline.
- **Continuous Learning**: Upload `.txt`/`.md` files or directly teach the agent new information via the UI. The agent logically validates new knowledge before integrating it into its vector database.
- **Web UI & Session Management**: A sleek, interactive web interface served by FastAPI with chat history persistence.

## 🏗️ Architectures

### RAG Pipeline (`rag_graph.py`)
```text
[retrieve] → [generate] → [critique]
                               ↓
                         PASS → [finalize] → Answer
                         FAIL → [rewrite_query] → [retrieve] → ...
```

### Code Pipeline (`code_graph.py`)
```text
[generate_code] → [patch_inputs] → [execute_code] → [critique_code]
                                                          ↓
                                                    PASS → END
                                                    FAIL → loop back to generate...
```

## 🛠️ Tech Stack

| Component       | Tool |
|-----------------|------|
| **Backend**     | FastAPI, Uvicorn, Python |
| **Orchestration**| LangGraph, LangChain |
| **LLMs**        | Mistral 7B (RAG), Qwen2.5-Coder 7B (Code/Intent) via Ollama |
| **Embeddings**  | all-MiniLM-L6-v2 (HuggingFace) |
| **Vector Store**| FAISS (local) |
| **Frontend**    | HTML, CSS, Vanilla JS (`ui.html`) |

## 🚀 Setup & Installation

### 1. Install Ollama & Pull Models
Download Ollama from [ollama.com](https://ollama.com/) and pull the required local models:
```bash
ollama pull mistral
ollama pull qwen2.5-coder:7b
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the FastAPI server:
```bash
uvicorn api:app --reload --port 8000
```
Then, open your browser and navigate to: [http://localhost:8000/ui](http://localhost:8000/ui)

*(Note: The first time you run it or ask a question, it will automatically download the HuggingFace embeddings model and build the FAISS index if it doesn't exist.)*

### Optional: CLI Mode
If you prefer testing the pure RAG pipeline in your terminal:
```bash
# Ingest documents manually
python main.py --ingest

# Interactive CLI chat
python main.py
```

## 📁 Project Structure

```text
self-healing-rag/
├── api.py               # FastAPI backend & unified endpoints
├── ui.html              # Web interface frontend
├── rag_graph.py         # Self-healing RAG LangGraph pipeline
├── code_graph.py        # Self-healing Code generation & execution pipeline
├── intent_detector.py   # LLM-based router (Code vs QA)
├── session_manager.py   # Chat history persistence
├── ingest.py            # Document ingestion → FAISS
├── main.py              # Legacy CLI entry point
├── sample_docs.md       # Default sample knowledge base
├── requirements.txt     # Python dependencies
├── faiss_index/         # Generated vector store
├── uploaded_docs/       # User uploaded knowledge files
└── sessions/            # Saved chat histories (JSON)
```

## ⚙️ Configuration

You can tweak the parameters at the top of the respective files:

| Variable | File | Default | Description |
|----------|------|---------|-------------|
| `LLM_MODEL` | `rag_graph.py` | `mistral` | LLM for Q&A and critique |
| `LLM_MODEL` | `code_graph.py` | `qwen2.5-coder:7b` | LLM for code generation |
| `MAX_RETRIES` | `rag_graph.py` | `2` | Max RAG self-healing retries |
| `SAFETY_CAP` | `code_graph.py` | `10` | Max code execution retries |
| `TOP_K` | `api.py` / `rag_graph.py`| `3` | Number of context chunks to retrieve |
