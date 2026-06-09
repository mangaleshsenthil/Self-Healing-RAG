"""
api.py
------
FastAPI server for the Self-Healing RAG pipeline.
Exposes every step of the pipeline in the response:
  - Retrieved chunks
  - Generated answer
  - Critique verdict
  - Retry attempts (rewritten queries)
  - Final answer

Usage:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /              → Health check
    POST /query         → Ask a question
    POST /ingest        → Re-ingest documents
    POST /upload        → Upload a new .txt or .md document
    GET  /docs          → Auto-generated Swagger UI
"""

import os
import shutil
import time
import threading
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── LangChain / RAG imports ───────────────────────────────────────────────────
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing import TypedDict

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
FAISS_PATH   = "faiss_index"
EMBED_MODEL  = "all-MiniLM-L6-v2"
LLM_MODEL    = "qwen2.5-coder:7b"
MAX_RETRIES  = 2
TOP_K        = 3
CHUNK_SIZE   = 300
CHUNK_OVERLAP= 50
UPLOAD_DIR   = "uploaded_docs"
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Per-session cancellation flags ────────────────────────────────────────────
_cancel_flags: dict = {}
_cancel_lock  = threading.Lock()

def _get_cancel_flag(session_id: str) -> threading.Event:
    with _cancel_lock:
        if session_id not in _cancel_flags:
            _cancel_flags[session_id] = threading.Event()
        return _cancel_flags[session_id]
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="🧠 Self-Healing RAG API",
    description="A RAG pipeline that critiques its own output and retries. Every step is visible.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class RetrievedChunk(BaseModel):
    content: str
    source:  str

class RetryAttempt(BaseModel):
    attempt_number:   int
    rewritten_query:  str
    retrieved_chunks: List[RetrievedChunk]
    generated_answer: str
    critique_verdict: str
    critique_reason:  str

class QueryResponse(BaseModel):
    question:          str
    status:            str           # "success" | "insufficient_info" | "error"
    total_time_sec:    float

    # ── First attempt ──────────────────────────────────────────────────────
    initial_query:     str
    retrieved_chunks:  List[RetrievedChunk]
    generated_answer:  str
    critique_verdict:  str           # "PASS" | "FAIL"
    critique_reason:   str

    # ── Retries (if any) ───────────────────────────────────────────────────
    retries_triggered: int
    retry_attempts:    List[RetryAttempt]

    # ── Final ──────────────────────────────────────────────────────────────
    final_answer:      str

class IngestResponse(BaseModel):
    status:      str
    chunks_created: int
    message:     str

class UploadResponse(BaseModel):
    status:   str
    filename: str
    message:  str

class LearnRequest(BaseModel):
    question:  str
    new_info:  str

class LearnResponse(BaseModel):
    status:       str
    validation:   str
    reason:       str
    saved_to:     str
    rerun_answer: str


# ── LangGraph State ───────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question:       str
    query:          str
    documents:      List[Document]
    answer:         str
    critique:       str
    retry_count:    int
    final_answer:   Optional[str]
    # tracking
    steps:          list   # list of step dicts recorded during run


# ── Load resources ────────────────────────────────────────────────────────────

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

def get_vectorstore():
    if not os.path.exists(FAISS_PATH):
        raise HTTPException(status_code=503, detail="FAISS index not found. POST /ingest first.")
    return FAISS.load_local(FAISS_PATH, get_embeddings(), allow_dangerous_deserialization=True)

def get_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0)


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def make_retrieve_node(vectorstore):
    def retrieve(state: RAGState) -> RAGState:
        docs = vectorstore.similarity_search(state["query"], k=TOP_K)
        step = {
            "step": "retrieve",
            "query": state["query"],
            "chunks": [
                {"content": d.page_content, "source": d.metadata.get("source", "sample_docs.md")}
                for d in docs
            ]
        }
        return {**state, "documents": docs, "steps": state["steps"] + [step]}
    return retrieve


def make_generate_node(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the context does not contain enough information, say "I don't have enough information".

Context:
{context}

Question: {question}

Answer:""")

    def generate(state: RAGState) -> RAGState:
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": state["question"]})
        answer = response.content.strip()
        step = {"step": "generate", "answer": answer}
        return {**state, "answer": answer, "steps": state["steps"] + [step]}
    return generate


def make_critique_node(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a strict fact-checker. Verify if the answer is grounded in the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Instructions:
- Reply with PASS if the answer is fully supported by the context.
- Reply with FAIL if the answer contains information NOT in the context, or if it's vague/made up.
- Start your response with either PASS or FAIL, then briefly explain why in one sentence.

Verdict:""")

    def critique(state: RAGState) -> RAGState:
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "question": state["question"],
            "answer": state["answer"]
        })
        verdict_full = response.content.strip()
        # Split verdict into PASS/FAIL and reason
        parts = verdict_full.split(None, 1)
        verdict = parts[0].upper().rstrip(".,:")
        reason  = parts[1].strip() if len(parts) > 1 else ""
        step = {"step": "critique", "verdict": verdict, "reason": reason}
        return {**state, "critique": verdict_full, "steps": state["steps"] + [step]}
    return critique


def make_rewrite_node(llm):
    prompt = ChatPromptTemplate.from_template("""
The previous retrieval did not find good enough information to answer the question.
Rewrite the question as a better search query to find more relevant documents.
Return ONLY the rewritten query, nothing else.

Original question: {question}
Previous query: {query}
Critique: {critique}

Rewritten search query:""")

    def rewrite_query(state: RAGState) -> RAGState:
        chain = prompt | llm
        response = chain.invoke({
            "question": state["question"],
            "query":    state["query"],
            "critique": state["critique"]
        })
        new_query = response.content.strip()
        step = {"step": "rewrite", "new_query": new_query}
        return {
            **state,
            "query":       new_query,
            "retry_count": state["retry_count"] + 1,
            "steps":       state["steps"] + [step]
        }
    return rewrite_query


def finalize(state: RAGState) -> RAGState:
    critique_upper = state["critique"].upper()
    if critique_upper.startswith("PASS"):
        final = state["answer"]
    else:
        final = "I don't have enough information in my knowledge base to answer this question accurately."
    step = {"step": "finalize", "final_answer": final}
    return {**state, "final_answer": final, "steps": state["steps"] + [step]}


def route_after_critique(state: RAGState) -> str:
    if state["critique"].upper().startswith("PASS"):
        return "finalize"
    elif state["retry_count"] >= MAX_RETRIES:
        return "finalize"
    else:
        return "rewrite_query"


def build_graph(vectorstore, llm):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve",      make_retrieve_node(vectorstore))
    graph.add_node("generate",      make_generate_node(llm))
    graph.add_node("critique",      make_critique_node(llm))
    graph.add_node("rewrite_query", make_rewrite_node(llm))
    graph.add_node("finalize",      finalize)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",       "generate")
    graph.add_edge("generate",       "critique")
    graph.add_conditional_edges("critique", route_after_critique, {
        "finalize":      "finalize",
        "rewrite_query": "rewrite_query"
    })
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("finalize",      END)
    return graph.compile()


# ── Parse steps into structured response ─────────────────────────────────────

def parse_steps(steps: list, question: str) -> dict:
    initial_chunks  = []
    initial_answer  = ""
    initial_verdict = ""
    initial_reason  = ""
    retries         = []

    retrieve_steps  = [s for s in steps if s["step"] == "retrieve"]
    generate_steps  = [s for s in steps if s["step"] == "generate"]
    critique_steps  = [s for s in steps if s["step"] == "critique"]
    rewrite_steps   = [s for s in steps if s["step"] == "rewrite"]
    finalize_step   = next((s for s in steps if s["step"] == "finalize"), {})

    # First attempt
    if retrieve_steps:
        initial_chunks = [
            RetrievedChunk(content=c["content"], source=c["source"])
            for c in retrieve_steps[0]["chunks"]
        ]
    if generate_steps:
        initial_answer = generate_steps[0]["answer"]
    if critique_steps:
        initial_verdict = critique_steps[0]["verdict"]
        initial_reason  = critique_steps[0]["reason"]

    # Retry attempts
    for i, rewrite in enumerate(rewrite_steps):
        idx = i + 1   # retry index
        retry_chunks  = []
        retry_answer  = ""
        retry_verdict = ""
        retry_reason  = ""

        if idx < len(retrieve_steps):
            retry_chunks = [
                RetrievedChunk(content=c["content"], source=c["source"])
                for c in retrieve_steps[idx]["chunks"]
            ]
        if idx < len(generate_steps):
            retry_answer = generate_steps[idx]["answer"]
        if idx < len(critique_steps):
            retry_verdict = critique_steps[idx]["verdict"]
            retry_reason  = critique_steps[idx]["reason"]

        retries.append(RetryAttempt(
            attempt_number   = idx,
            rewritten_query  = rewrite["new_query"],
            retrieved_chunks = retry_chunks,
            generated_answer = retry_answer,
            critique_verdict = retry_verdict,
            critique_reason  = retry_reason
        ))

    final_answer = finalize_step.get("final_answer", "")
    status = "success" if initial_verdict == "PASS" or any(
        r.critique_verdict == "PASS" for r in retries
    ) else "insufficient_info"

    return {
        "initial_query":     question,
        "retrieved_chunks":  initial_chunks,
        "generated_answer":  initial_answer,
        "critique_verdict":  initial_verdict,
        "critique_reason":   initial_reason,
        "retries_triggered": len(retries),
        "retry_attempts":    retries,
        "final_answer":      final_answer,
        "status":            status,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")

@app.get("/ui", tags=["UI"], include_in_schema=False)
def serve_ui():
    return FileResponse("ui.html")

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "running",
        "model":  LLM_MODEL,
        "index":  "ready" if os.path.exists(FAISS_PATH) else "not ingested"
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(req: QueryRequest):
    """
    Ask a question. Returns the full pipeline trace:
    retrieved chunks, generated answer, critique verdict,
    any retry attempts, and the final answer.
    """
    start = time.time()
    try:
        vectorstore = get_vectorstore()
        llm         = get_llm()
        graph_app   = build_graph(vectorstore, llm)

        initial_state: RAGState = {
            "question":    req.question,
            "query":       req.question,
            "documents":   [],
            "answer":      "",
            "critique":    "",
            "retry_count": 0,
            "final_answer": None,
            "steps":       []
        }

        result    = graph_app.invoke(initial_state)
        elapsed   = round(time.time() - start, 2)
        parsed    = parse_steps(result["steps"], req.question)

        return QueryResponse(
            question       = req.question,
            total_time_sec = elapsed,
            **parsed
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
def ingest_documents():
    """
    Re-ingests all documents (sample_docs.md + any uploaded files)
    and rebuilds the FAISS index.
    """
    try:
        all_chunks = []
        splitter   = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        embeddings = get_embeddings()

        # Load sample docs
        sources = ["sample_docs.md"]

        # Load uploaded docs
        if os.path.exists(UPLOAD_DIR):
            for f in os.listdir(UPLOAD_DIR):
                if f.endswith((".txt", ".md")):
                    sources.append(os.path.join(UPLOAD_DIR, f))

        for src in sources:
            if os.path.exists(src):
                loader = TextLoader(src, encoding="utf-8")
                docs   = loader.load()
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No documents found to ingest.")

        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local(FAISS_PATH)

        return IngestResponse(
            status         = "success",
            chunks_created = len(all_chunks),
            message        = f"Ingested {len(sources)} file(s) → {len(all_chunks)} chunks saved to FAISS."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a .txt or .md file to expand the knowledge base.
    After uploading, call POST /ingest to rebuild the index.
    """
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .txt and .md files are supported.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return UploadResponse(
        status   = "success",
        filename = file.filename,
        message  = f"File saved. Now call POST /ingest to rebuild the FAISS index."
    )


@app.post("/learn", response_model=LearnResponse, tags=["Knowledge"])
def learn(req: LearnRequest):
    """
    Validate new user-provided knowledge and, if logically consistent,
    save it to the knowledge base and re-ingest automatically.
    """
    try:
        llm = get_llm()

        # Step 1: Get existing context about the topic
        existing_context = ""
        if os.path.exists(FAISS_PATH):
            try:
                vs   = get_vectorstore()
                docs = vs.similarity_search(req.question, k=3)
                existing_context = "\n\n".join(d.page_content for d in docs)
            except Exception:
                existing_context = "No existing knowledge found."

        # Step 2: Ask Mistral to strictly validate the new info
        validation_prompt = ChatPromptTemplate.from_template("""
You are a strict knowledge validator. A user wants to add new information to a knowledge base.
Your job is to verify whether the new information is logically consistent and factually plausible.

Original Question: {question}

Existing knowledge about this topic (may be empty):
{existing_context}

New information the user wants to add:
{new_info}

Validation Rules (ALL must pass):
1. The new information must be factually plausible (not absurd or impossible).
2. It must NOT directly contradict the existing knowledge.
3. It must be logically related to the question or topic.
4. It must not be gibberish, spam, or harmful content.

Respond in this exact format:
VERDICT: VALID or INVALID
REASON: one clear sentence explaining why.
""")

        chain    = validation_prompt | llm
        response = chain.invoke({
            "question":         req.question,
            "existing_context": existing_context or "None",
            "new_info":         req.new_info
        })

        raw      = response.content.strip()
        lines    = raw.splitlines()
        verdict  = "INVALID"
        reason   = raw

        for line in lines:
            if line.upper().startswith("VERDICT:"):
                verdict = "VALID" if "VALID" in line.upper() else "INVALID"
            if line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        # Step 3: If INVALID, reject immediately
        if verdict == "INVALID":
            return LearnResponse(
                status       = "rejected",
                validation   = verdict,
                reason       = reason,
                saved_to     = "",
                rerun_answer = ""
            )

        # Step 4: Save new info to a file
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"learned_{timestamp}.md"
        save_path = os.path.join(UPLOAD_DIR, filename)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"## Learned: {req.question}\n\n{req.new_info}\n")

        # Step 5: Re-ingest everything
        all_chunks = []
        splitter   = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        embeddings = get_embeddings()
        sources    = ["sample_docs.md"]
        for fname in os.listdir(UPLOAD_DIR):
            if fname.endswith((".txt", ".md")):
                sources.append(os.path.join(UPLOAD_DIR, fname))
        for src in sources:
            if os.path.exists(src):
                loader = TextLoader(src, encoding="utf-8")
                docs   = loader.load()
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)

        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local(FAISS_PATH)

        # Step 6: Re-run the original question with updated knowledge
        graph_app = build_graph(vectorstore, llm)
        initial_state = {
            "question":    req.question,
            "query":       req.question,
            "documents":   [],
            "answer":      "",
            "critique":    "",
            "retry_count": 0,
            "final_answer": None,
            "steps":       []
        }
        result = graph_app.invoke(initial_state)

        return LearnResponse(
            status       = "accepted",
            validation   = verdict,
            reason       = reason,
            saved_to     = save_path,
            rerun_answer = result["final_answer"] or ""
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Code Generation Models ────────────────────────────────────────────────────

class CodeRequest(BaseModel):
    task:            str
    expected_output: str = ""

class AttemptDetail(BaseModel):
    attempt_number: int
    code:           str
    stdout:         str
    stderr:         str
    success:        bool
    critique:       str

class CodeResponse(BaseModel):
    task:            str
    status:          str        # "success" | "partial" | "failed"
    total_time_sec:  float
    total_attempts:  int
    attempts:        list
    final_code:      str
    final_output:    str


@app.post("/code", response_model=CodeResponse, tags=["Code"])
def generate_code(req: CodeRequest):
    """
    Generate Python code for a task, run it, critique the output,
    and self-heal if it fails — up to 3 times.
    Every attempt is visible in the response.
    """
    import time
    from code_graph import run_code_task

    start  = time.time()
    try:
        result  = run_code_task(req.task, req.expected_output)
        elapsed = round(time.time() - start, 2)

        return CodeResponse(
            task           = req.task,
            status         = result["final_status"] or "failed",
            total_time_sec = elapsed,
            total_attempts = len(result["attempts"]),
            attempts       = result["attempts"],
            final_code     = result["final_code"] or "",
            final_output   = result["final_output"] or ""
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat / Session Models ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message:    str

class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    intent:     str        # "code" or "qa"
    content:    str        # final answer or final code output
    meta:       dict       # full pipeline details
    total_time_sec: float


# ── Session Routes ────────────────────────────────────────────────────────────

@app.post("/sessions", tags=["Sessions"])
def create_session():
    """Create a new chat session."""
    from session_manager import create_session as _create
    return _create()


@app.get("/sessions", tags=["Sessions"])
def get_sessions():
    """List all sessions."""
    from session_manager import list_sessions
    return list_sessions()


@app.get("/sessions/{session_id}", tags=["Sessions"])
def get_session(session_id: str):
    """Load a session with full message history."""
    from session_manager import load_session
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(session_id: str):
    """Delete a session."""
    from session_manager import delete_session as _delete
    if _delete(session_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found.")


@app.post("/cancel/{session_id}", tags=["Chat"])
def cancel_request(session_id: str):
    """
    Signal the backend to stop the active pipeline for this session.
    The pipeline checks this flag cooperatively between steps.
    """
    flag = _get_cancel_flag(session_id)
    flag.set()
    print(f"\n⛔ [Cancel] Session {session_id} cancellation requested.")
    return {"status": "cancelling", "session_id": session_id}


# ── Unified Chat Route ────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest):
    """
    Unified chat endpoint.
    Auto-detects intent (code vs Q&A) and routes to the right pipeline.
    Saves everything to the session.
    """
    import time
    from session_manager import load_session, add_message
    from intent_detector import detect_intent

    start = time.time()

    # Set up cancellation flag for this session
    cancel_flag = _get_cancel_flag(req.session_id)
    cancel_flag.clear()   # reset any previous cancellation

    # Load session
    session = load_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Save user message
    add_message(req.session_id, "user", req.message)

    # Detect intent using history for context
    history  = session.get("messages", [])
    intent   = detect_intent(req.message, history)
    print(f"\n🎯 Intent detected: {intent.upper()}")

    meta    = {}
    content = ""

    # ── Route: Code pipeline ──────────────────────────────────────────────────
    if intent == "code":
        from code_graph import run_code_task
        result  = run_code_task(req.message, cancel_flag=cancel_flag)
        if result["final_status"] == "cancelled":
            raise HTTPException(status_code=499, detail="Request cancelled by user.")
        content = result["final_output"] or "Code generated but produced no output."
        if result["final_status"] == "failed":
            content = "I tried generating code but couldn't produce a working solution. Here's my best attempt."
        last_attempt = result["attempts"][-1] if result["attempts"] else {}
        meta = {
            "type":           "code",
            "status":         result["final_status"],
            "final_code":     result["final_code"],
            "final_output":   result["final_output"],
            "final_stderr":   last_attempt.get("stderr", ""),
            "note":           result.get("note", ""),
            "total_attempts": len(result["attempts"]),
            "attempts":       result["attempts"]
        }

    # ── Route: Q&A / RAG pipeline ─────────────────────────────────────────────
    else:
        if cancel_flag.is_set():
            raise HTTPException(status_code=499, detail="Request cancelled by user.")
        vectorstore = get_vectorstore()
        llm         = get_llm()
        graph_app   = build_graph(vectorstore, llm)

        initial_state: RAGState = {
            "question":     req.message,
            "query":        req.message,
            "documents":    [],
            "answer":       "",
            "critique":     "",
            "retry_count":  0,
            "final_answer": None,
            "steps":        []
        }
        result  = graph_app.invoke(initial_state)
        content = result["final_answer"] or ""
        parsed  = parse_steps(result["steps"], req.message)
        meta    = {
            "type":             "qa",
            "status":           parsed["status"],
            "retrieved_chunks": [{"content": c.content, "source": c.source} for c in parsed["retrieved_chunks"]],
            "generated_answer": parsed["generated_answer"],
            "critique_verdict": parsed["critique_verdict"],
            "critique_reason":  parsed["critique_reason"],
            "retries_triggered": parsed["retries_triggered"],
            "retry_attempts":   [r.dict() for r in parsed["retry_attempts"]]
        }

    elapsed = round(time.time() - start, 2)
    meta["total_time_sec"] = elapsed

    # Save assistant response
    msg = add_message(req.session_id, "assistant", content, meta)

    return ChatResponse(
        session_id     = req.session_id,
        message_id     = msg["id"],
        intent         = intent,
        content        = content,
        meta           = meta,
        total_time_sec = elapsed
    )


# ── Streaming Chat Route (SSE) ────────────────────────────────────────────────

@app.post("/chat/stream", tags=["Chat"])
def chat_stream(req: ChatRequest):
    """
    Streaming version of /chat using Server-Sent Events.
    Each pipeline step is yielded immediately as it completes.
    """
    import json as _json
    from fastapi.responses import StreamingResponse
    from session_manager import load_session, add_message as _add_message
    from intent_detector import detect_intent

    def _sse(obj: dict) -> str:
        return f"data: {_json.dumps(obj)}\n\n"

    def generate():
        start = time.time()

        session = load_session(req.session_id)
        if not session:
            yield _sse({"type": "error", "message": "Session not found."})
            return

        _add_message(req.session_id, "user", req.message)

        cancel_flag = _get_cancel_flag(req.session_id)
        cancel_flag.clear()

        # ── Intent detection ──────────────────────────────────────────────────
        yield _sse({"type": "status", "message": "Detecting intent…"})
        history = session.get("messages", [])
        intent  = detect_intent(req.message, history)
        yield _sse({"type": "intent", "intent": intent})

        if cancel_flag.is_set():
            yield _sse({"type": "cancelled"})
            return

        meta    = {}
        content = ""

        # ── Code pipeline ─────────────────────────────────────────────────────
        if intent == "code":
            from code_graph import run_code_task_stream
            all_attempts = []

            for event in run_code_task_stream(req.message, cancel_flag=cancel_flag):
                yield _sse(event)
                if event["type"] == "attempt":
                    all_attempts.append(event["attempt"])
                if event["type"] == "cancelled":
                    return
                if event["type"] == "final":
                    last = all_attempts[-1] if all_attempts else {}
                    meta = {
                        "type":           "code",
                        "status":         event["final_status"],
                        "final_code":     event["final_code"],
                        "final_output":   event["final_output"],
                        "final_stderr":   last.get("stderr", ""),
                        "note":           event.get("note", ""),
                        "total_attempts": event["total_attempts"],
                        "attempts":       all_attempts
                    }
                    content = event["final_output"] or "Code generated but produced no output."

        # ── QA / RAG pipeline ─────────────────────────────────────────────────
        else:
            try:
                vectorstore = get_vectorstore()
            except HTTPException as e:
                yield _sse({"type": "error", "message": e.detail})
                return

            llm = get_llm()

            state: RAGState = {
                "question":    req.message,
                "query":       req.message,
                "documents":   [],
                "answer":      "",
                "critique":    "",
                "retry_count": 0,
                "final_answer": None,
                "steps":       []
            }

            retry_log = []
            max_loops = MAX_RETRIES + 1

            for loop in range(max_loops):
                if cancel_flag.is_set():
                    yield _sse({"type": "cancelled"})
                    return

                # Retrieve
                yield _sse({"type": "status", "message": f"Retrieving documents{'  (retry #' + str(loop) + ')' if loop else ''}…"})
                state = make_retrieve_node(vectorstore)(state)
                chunks = [{"content": d.page_content, "source": d.metadata.get("source", "")}
                          for d in state["documents"]]
                yield _sse({"type": "qa_retrieve", "chunks": chunks, "query": state["query"], "loop": loop})

                if cancel_flag.is_set():
                    yield _sse({"type": "cancelled"})
                    return

                # Generate
                yield _sse({"type": "status", "message": "Generating answer…"})
                state = make_generate_node(llm)(state)
                yield _sse({"type": "qa_generate", "answer": state["answer"], "loop": loop})

                if cancel_flag.is_set():
                    yield _sse({"type": "cancelled"})
                    return

                # Critique
                yield _sse({"type": "status", "message": "Critiquing answer…"})
                state = make_critique_node(llm)(state)
                verdict_full = state["critique"]
                parts   = verdict_full.split(None, 1)
                verdict = parts[0].upper().rstrip(".,:") if parts else ""
                reason  = parts[1].strip() if len(parts) > 1 else ""
                yield _sse({"type": "qa_critique", "verdict": verdict, "reason": reason, "loop": loop})

                if verdict == "PASS" or state["retry_count"] >= MAX_RETRIES:
                    break

                # Rewrite for next loop
                yield _sse({"type": "status", "message": "Rewriting query for retry…"})
                state = make_rewrite_node(llm)(state)
                retry_log.append(state["query"])
                yield _sse({"type": "qa_rewrite", "new_query": state["query"]})

            # Finalize
            state   = finalize(state)
            content = state["final_answer"] or ""
            meta = {
                "type":             "qa",
                "status":           "success" if verdict == "PASS" else "insufficient_info",
                "retrieved_chunks": chunks,
                "generated_answer": state["answer"],
                "critique_verdict": verdict,
                "critique_reason":  reason,
                "retries_triggered": len(retry_log),
                "retry_attempts":   retry_log,
                "total_time_sec":   0
            }

        # ── Finalise & save ───────────────────────────────────────────────────
        elapsed = round(time.time() - start, 2)
        meta["total_time_sec"] = elapsed
        _add_message(req.session_id, "assistant", content, meta)
        yield _sse({"type": "done", "elapsed": elapsed, "meta": meta})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

