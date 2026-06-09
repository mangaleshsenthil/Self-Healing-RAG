"""
intent_detector.py
------------------
Uses Qwen to detect whether a user message is:
  - "code"  → needs code generation + execution
  - "qa"    → needs RAG Q&A pipeline
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

LLM_MODEL = "qwen2.5-coder:7b"


def detect_intent(message: str, history: list = None) -> str:
    """
    Returns "code" or "qa".
    Uses recent history for context if available.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # Build recent history context
    history_text = ""
    if history:
        recent = history[-4:]  # last 4 messages for context
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:100]}"
            for m in recent
        )

    prompt = ChatPromptTemplate.from_template("""
You are an intent classifier. Classify the user message as either "code" or "qa".

Rules:
- "code" = user wants to generate, write, fix, run, or debug any programming code
- "qa"   = user wants to ask a question, get information, or have something explained

Examples:
- "Write a Python function to sort a list" → code
- "Fix this JavaScript error" → code  
- "What is machine learning?" → qa
- "Explain how DNS works" → qa
- "How does a for loop work?" → qa
- "Write a for loop example" → code
- "What is the capital of France?" → qa
- "Create a script to read a CSV file" → code

Recent conversation:
{history}

User message: {message}

Reply with ONLY one word: code or qa""")

    chain    = prompt | llm
    response = chain.invoke({
        "message": message,
        "history": history_text or "None"
    })

    result = response.content.strip().lower()

    # Safely extract intent
    if "code" in result:
        return "code"
    return "qa"
