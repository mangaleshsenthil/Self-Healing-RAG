"""
code_graph.py
-------------
Self-Healing Code Generation pipeline.

Flow:
  [generate_code] → [patch_inputs] → [execute_code] → [critique_code]
                                                            ↓
                                                      PASS → END
                                                      FAIL → [rewrite_code] → loop...

run_code_task(task, expected_output="") returns:
    {
        "final_status":  "success" | "partial" | "failed",
        "final_code":    str,   # clean code (without mock patch header)
        "final_output":  str,
        "note":          str,   # hint if input() was mocked
        "attempts": [
            {
                "attempt_number": int,
                "code":           str,   # the clean code shown to user
                "stdout":         str,
                "stderr":         str,
                "success":        bool,
                "critique":       str,
                "note":           str    # non-empty if input() was mocked
            },
            ...
        ]
    }
"""

import subprocess
import sys
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ── Config ────────────────────────────────────────────────────────────────────
LLM_MODEL    = "qwen2.5-coder:7b"
SAFETY_CAP   = 10    # hard upper limit to prevent runaway loops
EXEC_TIMEOUT = 20    # seconds per execution
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_INPUT_HEADER = '''\
# ── AUTO-MOCK: input() replaced with sample values for demo ─────────────────
_mock_values = iter(["5", "hello", "10", "3", "yes", "2", "abc", "1", "7", "no", "0"])
def input(prompt=""):
    val = next(_mock_values, "0")
    print(f"[AUTO-INPUT] {prompt.strip()!r} → {val!r}")
    return val
# ─────────────────────────────────────────────────────────────────────────────

'''

_INPUT_NOTE = (
    "⚠️  This code uses input() — for the demo, input() was automatically mocked "
    "with sample values (5, hello, 10, 3, yes, …).\n"
    "To make it truly interactive, remove the AUTO-MOCK block at the top of the code."
)


def _get_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.2)


# ── Patch input() calls ───────────────────────────────────────────────────────

def patch_input_calls(code: str) -> tuple:
    """
    If code uses input(), prepend a mock that returns sample values.
    Returns (code_for_execution, display_code, was_patched, note)
    - code_for_execution: has the mock header (runs without hanging)
    - display_code:       the clean original code shown to the user
    """
    if "input(" not in code:
        return code, code, False, ""
    exec_code = _MOCK_INPUT_HEADER + code
    return exec_code, code, True, _INPUT_NOTE


# ── Step 1: Generate code ─────────────────────────────────────────────────────

_BASE_RULES = """\
IMPORTANT RULES:
- Do NOT use input() or any blocking prompt. Use hardcoded sample/random values instead.
- If the task inherently needs user input (e.g. "menu-driven"), use a fixed demo value
  and add a clearly marked comment block at the end showing exactly how to convert it
  to interactive (replacing your hardcoded value with input()).
- The code MUST run to completion without any user interaction.
- Return ONLY raw Python code — no markdown fences, no explanation."""


def generate_code(task: str, previous_code: str = "", error_feedback: str = "", llm=None) -> str:
    if llm is None:
        llm = _get_llm()

    if previous_code and error_feedback:
        prompt = ChatPromptTemplate.from_template("""\
You are an expert Python programmer. The previous attempt failed. Fix it.

Task: {task}

Previous code:
{previous_code}

Error / critique:
{error_feedback}

""" + _BASE_RULES)
        response = (prompt | llm).invoke({
            "task":           task,
            "previous_code":  previous_code,
            "error_feedback": error_feedback
        })
    else:
        prompt = ChatPromptTemplate.from_template("""\
You are an expert Python programmer.
Write clean, correct Python code to accomplish the following task.

Task: {task}

""" + _BASE_RULES)
        response = (prompt | llm).invoke({"task": task})

    code = response.content.strip()
    # Strip markdown fences if the model included them anyway
    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    return code


# ── Step 2: Execute code ──────────────────────────────────────────────────────

def execute_code(code: str) -> dict:
    """
    Runs code in a subprocess with a timeout.
    Returns {"stdout": str, "stderr": str, "success": bool}
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT,
            input=""        # close stdin so input() immediately raises EOFError instead of blocking
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        success = proc.returncode == 0 and not stderr
        return {"stdout": stdout, "stderr": stderr, "success": success}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Execution timed out after {EXEC_TIMEOUT}s.", "success": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "success": False}


# ── Step 3: Critique output ───────────────────────────────────────────────────

def critique_code(task: str, code: str, stdout: str, stderr: str,
                  expected_output: str = "", inputs_mocked: bool = False, llm=None) -> str:
    if llm is None:
        llm = _get_llm()

    mock_note = (
        "Note: input() calls were automatically mocked with sample values, "
        "so the code ran non-interactively. Judge it based on correctness of logic, not the demo values."
        if inputs_mocked else ""
    )

    prompt = ChatPromptTemplate.from_template("""\
You are a strict code reviewer. Evaluate whether the code correctly solves the task.

Task: {task}

Code:
{code}

Execution stdout:
{stdout}

Execution stderr:
{stderr}

Expected output (may be empty):
{expected_output}

{mock_note}

Rules:
- Reply PASS if the code logic is correct and ran without errors.
- Reply FAIL if there were errors, clearly wrong output, or no output when output was expected.
- Start with PASS or FAIL, then one sentence of explanation.

Verdict:""")

    response = (prompt | llm).invoke({
        "task":            task,
        "code":            code,
        "stdout":          stdout or "(no output)",
        "stderr":          stderr or "(none)",
        "expected_output": expected_output or "(not specified)",
        "mock_note":       mock_note
    })
    return response.content.strip()


# ── Main runner ───────────────────────────────────────────────────────────────

def run_code_task(task: str, expected_output: str = "", cancel_flag=None) -> dict:
    """
    Self-healing code generation loop — retries until PASS (or SAFETY_CAP).
    cancel_flag: optional threading.Event; if set, the loop stops immediately.
    Returns the structured result dict consumed by api.py.
    """
    llm         = _get_llm()
    attempts    = []
    code        = ""
    feedback    = ""
    attempt_num = 0
    last_note   = ""

    while True:
        attempt_num += 1

        # ── Check cancellation before each attempt ─────────────────────────────
        if cancel_flag and cancel_flag.is_set():
            print(f"\n⛔ [Code] Cancelled before attempt #{attempt_num}.")
            return {
                "final_status": "cancelled",
                "final_code":   code,
                "final_output": "",
                "note":         last_note,
                "attempts":     attempts
            }
        # ───────────────────────────────────────────────────────────────

        print(f"\n{'='*50}")
        print(f"💻 [Code] Attempt #{attempt_num} — generating code...")

        # Generate clean code
        code = generate_code(task, previous_code=code, error_feedback=feedback, llm=llm)
        print(f"   → Generated {len(code)} chars")

        # Check cancellation after LLM call (LLM calls are blocking — this is the earliest we can stop)
        if cancel_flag and cancel_flag.is_set():
            print(f"\n⛔ [Code] Cancelled after generation (attempt #{attempt_num}).")
            return {
                "final_status": "cancelled",
                "final_code":   code,
                "final_output": "",
                "note":         last_note,
                "attempts":     attempts
            }

        # Patch input() if needed (exec_code runs, display_code is shown to user)
        exec_code, display_code, inputs_mocked, note = patch_input_calls(code)
        last_note = note
        if inputs_mocked:
            print("   → input() detected — mocking with sample values")

        # Execute
        exec_result = execute_code(exec_code)
        stdout  = exec_result["stdout"]
        stderr  = exec_result["stderr"]
        print(f"   → Executed: returncode ok={exec_result['success']}")
        if stdout:
            print(f"   → stdout: {stdout[:120]}")
        if stderr:
            print(f"   → stderr: {stderr[:120]}")

        # Critique
        critique = critique_code(
            task, display_code, stdout, stderr,
            expected_output, inputs_mocked=inputs_mocked, llm=llm
        )
        passed = critique.upper().startswith("PASS")
        print(f"   → Critique: {critique[:120]}")

        attempts.append({
            "attempt_number": attempt_num,
            "code":           display_code,   # clean code shown to user
            "stdout":         stdout,
            "stderr":         stderr,
            "success":        passed,
            "critique":       critique,
            "note":           note
        })

        if passed:
            print(f"\n✅ [Code] Passed on attempt #{attempt_num}!")
            return {
                "final_status":  "success",
                "final_code":    display_code,
                "final_output":  stdout,
                "note":          note,
                "attempts":      attempts
            }

        # Hit safety cap — return best partial result
        if attempt_num >= SAFETY_CAP:
            print(f"\n⛔ [Code] Safety cap ({SAFETY_CAP}) reached.")
            status = "partial" if stdout else "failed"
            return {
                "final_status":  status,
                "final_code":    display_code,
                "final_output":  stdout,
                "note":          note,
                "attempts":      attempts
            }

        # Build feedback for next attempt
        if stderr:
            feedback = f"Critique: {critique}\nError output:\n{stderr}"
        else:
            feedback = critique
        print(f"   → Retrying (attempt {attempt_num + 1})...")


# ── Streaming generator version ───────────────────────────────────────────────

def run_code_task_stream(task: str, expected_output: str = "", cancel_flag=None):
    """
    Generator version of run_code_task.
    Yields dicts: {"type": "status"|"attempt"|"final"|"cancelled", ...}
    Each yield is one SSE event for the frontend.
    """
    llm         = _get_llm()
    code        = ""
    feedback    = ""
    last_note   = ""
    attempt_num = 0

    while True:
        attempt_num += 1

        if cancel_flag and cancel_flag.is_set():
            yield {"type": "cancelled"}
            return

        yield {"type": "status", "message": f"Generating code — attempt #{attempt_num}…"}
        code = generate_code(task, previous_code=code, error_feedback=feedback, llm=llm)

        if cancel_flag and cancel_flag.is_set():
            yield {"type": "cancelled"}
            return

        exec_code, display_code, inputs_mocked, note = patch_input_calls(code)
        last_note = note

        yield {"type": "status", "message": f"Running code — attempt #{attempt_num}…"}
        exec_result = execute_code(exec_code)
        stdout  = exec_result["stdout"]
        stderr  = exec_result["stderr"]

        if cancel_flag and cancel_flag.is_set():
            yield {"type": "cancelled"}
            return

        yield {"type": "status", "message": f"Critiquing result — attempt #{attempt_num}…"}
        critique = critique_code(
            task, display_code, stdout, stderr,
            expected_output, inputs_mocked=inputs_mocked, llm=llm
        )
        passed = critique.upper().startswith("PASS")

        yield {
            "type": "attempt",
            "attempt": {
                "attempt_number": attempt_num,
                "code":           display_code,
                "stdout":         stdout,
                "stderr":         stderr,
                "success":        passed,
                "critique":       critique,
                "note":           note
            }
        }

        if passed:
            yield {"type": "final", "final_status": "success",
                   "final_code": display_code, "final_output": stdout,
                   "note": last_note, "total_attempts": attempt_num}
            return

        if attempt_num >= SAFETY_CAP:
            status = "partial" if stdout else "failed"
            yield {"type": "final", "final_status": status,
                   "final_code": display_code, "final_output": stdout,
                   "note": last_note, "total_attempts": attempt_num}
            return

        feedback = f"Critique: {critique}\nError:\n{stderr}" if stderr else critique
