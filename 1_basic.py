import asyncio
import subprocess

from llama_index.core.llms import LLM
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.llms.anthropic import Anthropic

# --- Events --- are how one step of the Worlflow sets off the next
# Steps will emit an Event and the next step which has the matching
# event type will run


# Compile event is emited by the start() step and caught by compile()
class CompileEvent(Event):
    code: str
    attempt: int = 1


# CompileFailedEvent is emited by
class CompileFailedEvent(Event):
    code: str
    error_log: str
    attempt: int


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
        print(f"Starting SV compile-debug workflow")
        print(f"{'=' * 60}\n")
        return CompileEvent(code=code, attempt=1)

    @step
    async def compile(self, ev: CompileEvent) -> CompileFailedEvent | StopEvent:
        print(f"\n--- Attempt {ev.attempt} ---")

        # Write the current source to disk
        sv_path = "design.sv"
        with open(sv_path, "w") as f:
            f.write(ev.code)

        # Run the actual compiler
        result = subprocess.run(
            ["vcs", "-sverilog", "-compile", sv_path],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"[Attempt {ev.attempt}] ✓ Compilation succeeded!")
            return StopEvent(
                result={
                    "status": "success",
                    "attempts": ev.attempt,
                    "final_code": ev.code,
                }
            )

        error_log = result.stderr + result.stdout
        print(f"[Attempt {ev.attempt}] ✗ Compile failed:\n{error_log[:500]}")

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
                    "final_code": ev.code,
                }
            )

        print(f"[Debug] Asking LLM to diagnose and fix...")

        prompt = f"""You are an expert SystemVerilog engineer. A design failed to compile.

## Current source code:
```systemverilog
{ev.code}
```

## Compiler error log:
```
{ev.error_log}
```

Analyze the error(s) and produce the COMPLETE corrected SystemVerilog source.
Return ONLY the corrected code inside a single ```systemverilog fenced block.
Do not explain anything outside the code block.
"""

        response = await self.llm.acomplete(prompt)
        response_text = response.text

        # Extract the code block from the LLM response
        patched_code = self._extract_code_block(response_text)

        if patched_code is None:
            print("[Debug] Warning: couldn't parse code block, using raw response")
            patched_code = response_text

        print(f"[Debug] LLM proposed fix (first 300 chars):\n{patched_code[:300]}")

        return CompileEvent(code=patched_code, attempt=ev.attempt + 1)

    @staticmethod
    def _extract_code_block(text: str) -> str | None:
        """Pull the first ```systemverilog ... ``` block out of the LLM response."""
        import re

        # Try systemverilog or sv or bare fence
        for lang in (r"systemverilog", r"sv", r""):
            pattern = rf"```{lang}\s*\n(.*?)```"
            m = re.search(pattern, text, re.DOTALL)
            if m:
                return m.group(1).strip()
        return None


# --- Run it ---
async def main():
    llm = Anthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
    )

    wf = SVCompileWorkflow(llm=llm, timeout=120)

    design = """\
module top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [15:0] data_bus,
    output logic        valid
);

    logic clk_en;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            clk_en <= 1'b0;
        else
            clk_en <= |data_bus;
    end

    assign valid = clk_en & |data_bus
    assign valid = rst_n;  // oops, double driver

endmodule
"""

    result = await wf.run(code=design)
    print(f"\n{'=' * 60}")
    print(f"Final result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
