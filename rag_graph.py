"""
rag_graph.py
------------
The Self-Healing RAG pipeline built with LangGraph.

Flow:
  [retrieve] → [generate] → [critique]
                                ↓
                          PASS → END
                          FAIL → [rewrite_query] → [retrieve] → ...
"""

from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# ── Config ────────────────────────────────────────────────────────────────────
FAISS_PATH  = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "mistral"       # Ollama model name
MAX_RETRIES = 2               # Max re-retrieve attempts before giving up
TOP_K       = 3               # Number of chunks to retrieve
# ──────────────────────────────────────────────────────────────────────────────


# ── State ─────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:       str
    query:          str                  # may be rewritten on retry
    documents:      List[Document]
    answer:         str
    critique:       str                  # PASS or FAIL + reason
    retry_count:    int
    final_answer:   Optional[str]
# ──────────────────────────────────────────────────────────────────────────────


# ── Load shared resources ─────────────────────────────────────────────────────
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    return vectorstore, llm


# ── Node 1: Retrieve ──────────────────────────────────────────────────────────
def make_retrieve_node(vectorstore):
    def retrieve(state: RAGState) -> RAGState:
        print(f"\n🔍 [Retrieve] Query: '{state['query']}'")
        docs = vectorstore.similarity_search(state["query"], k=TOP_K)
        print(f"   → Retrieved {len(docs)} chunks")
        return {**state, "documents": docs}
    return retrieve


# ── Node 2: Generate ──────────────────────────────────────────────────────────
def make_generate_node(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the context does not contain enough information, say "I don't have enough information".

Context:
{context}

Question: {question}

Answer:""")

    def generate(state: RAGState) -> RAGState:
        print("\n🤖 [Generate] Generating answer...")
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "question": state["question"]
        })
        answer = response.content.strip()
        print(f"   → Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")
        return {**state, "answer": answer}
    return generate


# ── Node 3: Critique ──────────────────────────────────────────────────────────
def make_critique_node(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a strict fact-checker. Your job is to verify if the answer is grounded 
in the provided context, or if the model hallucinated.

Context:
{context}

Question: {question}

Answer: {answer}

Instructions:
- Reply with PASS if the answer is fully supported by the context.
- Reply with FAIL if the answer contains information NOT in the context, or if it's vague/made up.
- Start your response with either PASS or FAIL, then briefly explain why.

Verdict:""")

    def critique(state: RAGState) -> RAGState:
        print("\n🔎 [Critique] Evaluating answer...")
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "question": state["question"],
            "answer": state["answer"]
        })
        verdict = response.content.strip()
        print(f"   → Verdict: {verdict[:120]}{'...' if len(verdict) > 120 else ''}")
        return {**state, "critique": verdict}
    return critique


# ── Node 4: Rewrite Query ─────────────────────────────────────────────────────
def make_rewrite_node(llm):
    prompt = ChatPromptTemplate.from_template("""
The previous retrieval did not find good enough information to answer the question.
Rewrite the question as a better search query to find more relevant documents.
Return ONLY the rewritten query, nothing else.

Original question: {question}
Previous query used: {query}
Critique of the answer: {critique}

Rewritten search query:""")

    def rewrite_query(state: RAGState) -> RAGState:
        print("\n✏️  [Rewrite] Reformulating query...")
        chain = prompt | llm
        response = chain.invoke({
            "question": state["question"],
            "query": state["query"],
            "critique": state["critique"]
        })
        new_query = response.content.strip()
        print(f"   → New query: '{new_query}'")
        return {
            **state,
            "query": new_query,
            "retry_count": state["retry_count"] + 1
        }
    return rewrite_query


# ── Node 5: Finalize ──────────────────────────────────────────────────────────
def finalize(state: RAGState) -> RAGState:
    critique_upper = state["critique"].upper()

    if critique_upper.startswith("PASS"):
        print("\n✅ [Finalize] Answer passed critique!")
        final = state["answer"]
    else:
        print("\n⚠️  [Finalize] Max retries reached. Returning honest response.")
        final = "I don't have enough information in my knowledge base to answer this question accurately."

    return {**state, "final_answer": final}


# ── Router: after critique ────────────────────────────────────────────────────
def route_after_critique(state: RAGState) -> str:
    critique_upper = state["critique"].upper()
    if critique_upper.startswith("PASS"):
        return "finalize"
    elif state["retry_count"] >= MAX_RETRIES:
        print(f"\n🚫 Max retries ({MAX_RETRIES}) reached.")
        return "finalize"
    else:
        return "rewrite_query"


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_graph():
    vectorstore, llm = load_resources()

    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("retrieve",      make_retrieve_node(vectorstore))
    graph.add_node("generate",      make_generate_node(llm))
    graph.add_node("critique",      make_critique_node(llm))
    graph.add_node("rewrite_query", make_rewrite_node(llm))
    graph.add_node("finalize",      finalize)

    # Add edges
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",      "generate")
    graph.add_edge("generate",      "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "finalize":     "finalize",
            "rewrite_query":"rewrite_query"
        }
    )
    graph.add_edge("rewrite_query", "retrieve")   # loop back!
    graph.add_edge("finalize",      END)

    return graph.compile()


# ── Run a single query ────────────────────────────────────────────────────────
def run_query(question: str):
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    print('='*60)

    app = build_graph()

    initial_state: RAGState = {
        "question":     question,
        "query":        question,    # start with original question as query
        "documents":    [],
        "answer":       "",
        "critique":     "",
        "retry_count":  0,
        "final_answer": None
    }

    result = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"💬 Final Answer:\n{result['final_answer']}")
    print('='*60)
    return result["final_answer"]
