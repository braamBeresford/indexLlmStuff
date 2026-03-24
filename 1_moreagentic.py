import asyncio
import os
import subprocess

from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from helper import TEST_DESIGN, MockLLM2


# --- Events ---
class CompileEvent(Event):
    code: str
    attempt: int = 1


class CompileFailedEvent(Event):
    code: str
    error_log: str
    attempt: int


# --- Tools the agent can use ---
def read_file(filepath: str) -> str:
    """Read the contents of a file. Useful for inspecting source files,
    headers, or package definitions referenced in error messages."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file '{filepath}' not found."


def list_files(directory: str = ".") -> str:
    """List files in a directory. Useful for discovering available
    source files, include directories, or IP blocks."""
    try:
        entries = os.listdir(directory)
        return "\n".join(sorted(entries))
    except FileNotFoundError:
        return f"Error: directory '{directory}' not found."


def grep_source(pattern: str, filepath: str) -> str:
    """Search for a pattern in a file. Useful for finding signal
    declarations, module instantiations, or specific identifiers
    mentioned in error messages."""
    try:
        result = subprocess.run(
            ["grep", "-n", pattern, filepath],
            capture_output=True,
            text=True,
        )
        return (
            result.stdout
            if result.stdout
            else f"No matches for '{pattern}' in {filepath}"
        )
    except Exception as e:
        return f"Error running grep: {e}"


def run_lint(filepath: str) -> str:
    """Run a quick lint pass on a SystemVerilog file without full compilation.
    Returns warnings and errors for the given file only."""
    try:
        with open(filepath, "r") as f:
            code = f.read()
    except FileNotFoundError:
        return f"Error: file '{filepath}' not found."
    errors = []
    if "assign valid" in code and code.count("assign valid") > 1:
        errors.append("WARNING: signal 'valid' has multiple drivers")
    if "& |data_bus" in code and "& |data_bus;" not in code:
        errors.append("ERROR: missing semicolon after continuous assignment")
    return "\n".join(errors) if errors else "Lint clean — no warnings."


def write_file(filepath: str, content: str) -> str:
    """Write content to a file. Use this to apply your fix by writing
    the corrected SystemVerilog source."""
    with open(filepath, "w") as f:
        f.write(content)
    return f"Successfully wrote {len(content)} bytes to {filepath}"


def compile_design(filepath: str) -> str:
    """Compile a SystemVerilog file. Returns the compiler output.
    Use this to test whether your fix actually works."""
    r = _mock_compile_file(filepath)
    stats = f"[{r['runtime_s']}s | {r['peak_mem_mb']}MB]"
    return f"{r['output']}\n{stats}"


def _mock_compile_file(filepath: str) -> dict:
    """Mock compiler that checks for known SV errors. Returns a result dict."""
    try:
        with open(filepath, "r") as f:
            code = f.read()
    except FileNotFoundError:
        return {
            "success": False,
            "output": f"FAILED: file '{filepath}' not found.",
            "runtime_s": 0.0,
            "peak_mem_mb": 0,
        }
    errors = []
    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("assign") and not stripped.endswith(";") and stripped:
            errors.append(f"Error-[SE] Syntax error at line {i}: missing semicolon")
    if code.count("assign valid") > 1:
        errors.append("Error-[MDRV] Multiple drivers on signal 'valid'")

    lines = len(code.splitlines())
    if errors:
        return {
            "success": False,
            "output": "FAILED (exit 1):\n" + "\n".join(errors),
            "runtime_s": round(0.8 + lines * 0.02, 2),
            "peak_mem_mb": 64 + lines * 2,
        }
    return {
        "success": True,
        "output": "SUCCESS: Compilation passed.\n0 errors, 0 warnings",
        "runtime_s": round(1.2 + lines * 0.03, 2),
        "peak_mem_mb": 128 + lines * 3,
    }


AGENT_TOOLS = [
    # We're turning each fo the functions into a FunctionTool which can be used by the LLM
    FunctionTool.from_defaults(fn=read_file),
    FunctionTool.from_defaults(fn=list_files),
    FunctionTool.from_defaults(fn=grep_source),
    FunctionTool.from_defaults(fn=run_lint),
    FunctionTool.from_defaults(fn=write_file),
    FunctionTool.from_defaults(fn=compile_design),
]


# --- Workflow ---
class SVCompileWorkflow(Workflow):
    max_attempts: int = 5

    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def start(self, ev: StartEvent) -> CompileEvent:
        code = ev.get("code", "module top; endmodule")
        print(f"\n{'=' * 60}")
        print(f" STEP 1: INITIALIZE")
        print(f"{'=' * 60}")
        print(f"Starting agentic SV compile workflow")
        print(f"Design loaded ({len(code.splitlines())} lines)\n")
        return CompileEvent(code=code, attempt=1)

    @step
    async def compile(self, ev: CompileEvent) -> CompileFailedEvent | StopEvent:
        print(f"\n{'=' * 60}")
        print(f" STEP 2: COMPILE (Attempt {ev.attempt})")
        print(f"{'=' * 60}")

        sv_path = "design.sv"
        with open(sv_path, "w") as f:
            f.write(ev.code)

        print(f"[Attempt {ev.attempt}] Compiling {sv_path}...")
        await asyncio.sleep(4)  # simulate compile time

        r = _mock_compile_file(sv_path)

        if r["success"]:
            print(f"[Attempt {ev.attempt}] ✓ Compilation succeeded!")
            print(f"  Runtime: {r['runtime_s']}s | Peak memory: {r['peak_mem_mb']}MB")
            return StopEvent(
                result={
                    "status": "success",
                    "attempts": ev.attempt,
                    "runtime_s": r["runtime_s"],
                    "peak_mem_mb": r["peak_mem_mb"],
                }
            )

        error_log = r["output"]
        print(
            f"[Attempt {ev.attempt}] ✗ Compile failed ({r['runtime_s']}s | {r['peak_mem_mb']}MB):"
        )
        print(f"  {error_log[:500]}")

        return CompileFailedEvent(
            code=ev.code,
            error_log=error_log,
            attempt=ev.attempt,
        )

    @step
    async def debug_and_fix(self, ev: CompileFailedEvent) -> CompileEvent | StopEvent:
        if ev.attempt >= self.max_attempts:
            print(f"\n✗ Giving up after {ev.attempt} attempts.")
            return StopEvent(
                result={
                    "status": "failed",
                    "attempts": ev.attempt,
                    "last_error": ev.error_log[:1000],
                }
            )

        print(f"\n{'=' * 60}")
        print(f" STEP 3: DEBUG & FIX (Attempt {ev.attempt})")
        print(f"{'=' * 60}")
        print(f"Spawning ReAct agent to investigate and fix...")
        await asyncio.sleep(2)  # simulate agent startup

        agent = ReActAgent(
            tools=AGENT_TOOLS,
            llm=self.llm,
            system_prompt="""You are an expert SystemVerilog debug engineer.
You have tools to read files, search code, lint, write fixes, and compile.

Your workflow:
1. Analyze the error log to understand the root cause
2. Use grep/read_file to inspect the relevant code and context
3. Use lint to get additional diagnostics if helpful
4. Write the corrected file using write_file
5. Use compile_design to verify your fix works
6. If it still fails, iterate — read the new errors, fix, recompile

When the design compiles clean, respond with EXACTLY:
FIXED: <one-line summary of what you changed>

If you cannot fix it, respond with EXACTLY:
STUCK: <explanation of what's wrong and why you can't fix it>
""",
        )

        task = f"""The following SystemVerilog design failed to compile.

## File: design.sv
```systemverilog
{ev.code}
```

## Compiler error log:
```
{ev.error_log}
```

Debug the errors, fix the source, and verify the fix compiles.
"""

        print(f"[Debug] Agent analyzing errors and generating fix...")
        await asyncio.sleep(6)  # simulate LLM thinking time

        response = await agent.run(task)
        response_text = str(response)

        # Print just the summary line, not the full code block
        summary = response_text.split("\n")[0]
        print(f"\n[Debug] Agent response: {summary}")

        # Read back whatever the agent wrote to design.sv
        # If the agent used write_file, design.sv will have the fix.
        # Otherwise, try to extract code from the agent's text response.
        try:
            with open("design.sv", "r") as f:
                patched_code = f.read()
        except FileNotFoundError:
            patched_code = ev.code

        # If design.sv still has the original code, extract from response
        if patched_code.strip() == ev.code.strip():
            extracted = self._extract_code_block(response_text)
            if extracted:
                patched_code = extracted
                with open("design.sv", "w") as f:
                    f.write(patched_code)

        # If the agent already verified compilation, check its response
        if response_text.strip().startswith("FIXED:"):
            # Agent says it's fixed and it verified — do one final compile
            # in our controlled step to confirm
            return CompileEvent(code=patched_code, attempt=ev.attempt + 1)

        if response_text.strip().startswith("STUCK:"):
            return StopEvent(
                result={
                    "status": "stuck",
                    "attempts": ev.attempt,
                    "agent_message": response_text,
                    "final_code": patched_code,
                }
            )

        # Agent didn't follow format — just try compiling what it wrote
        return CompileEvent(code=patched_code, attempt=ev.attempt + 1)

    @staticmethod
    def _extract_code_block(text: str) -> str | None:
        """Pull the first ```systemverilog ... ``` block out of text."""
        import re

        for lang in (r"systemverilog", r"sv", r""):
            pattern = rf"```{lang}\s*\n(.*?)```"
            m = re.search(pattern, text, re.DOTALL)
            if m:
                return m.group(1).strip()
        return None


# --- Run it ---
async def main():
    # Mock LLM for PoC - Swap to Anthropic/OpenAi for real usage
    # from llama_index.llms.openai import OpenAI
    # llm = OpenAI(model="gpt-4o", max_tokens=4096)
    llm = MockLLM2()

    wf = SVCompileWorkflow(llm=llm, timeout=300)

    # TEST_DESIGN represents an SV top we are running compile against
    result = await wf.run(code=TEST_DESIGN)
    print(f"\n{'=' * 60}")
    print(f" RESULT")
    print(f"{'=' * 60}")
    print(f"Status:      {result['status']}")
    print(f"Attempts:    {result['attempts']}")
    if "runtime_s" in result:
        print(f"Runtime:     {result['runtime_s']}s")
        print(f"Peak memory: {result['peak_mem_mb']}MB")


if __name__ == "__main__":
    asyncio.run(main())
