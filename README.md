# LLM Workflows for Hardware Verification

This repo contains two examples that demonstrate how to build LLM-powered workflows for common hardware verification tasks using [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/). Each example builds on the last, introducing new concepts incrementally.

Both examples use mock LLMs so they run without API keys or EDA tools. To use a real LLM, swap the mock for an Anthropic or OpenAI client (commented-out examples are in each file).


## Example 0: Regression Report Generator (`0_reportgen.py`)

This example introduces the fundamentals of LlamaIndex Workflows: **Events**, **Steps**, and how they chain together.

### What it does

Simulates the morning routine of checking overnight regression results. The workflow collects test results across five suites (functional, coverage, timing, lint, power), has an LLM analyze failure patterns, and produces a health report with prioritized action items.

### Key concepts

- **Events** carry data between steps. Each event is a simple class with typed fields:

  ```python
  class GatherResultsEvent(Event):
      results: str  # JSON-encoded regression results
  ```

- **Steps** are async methods decorated with `@step`. The framework routes events automatically based on the type signature — a step that accepts `GatherResultsEvent` runs when that event is emitted:

  ```python
  @step
  async def analyze_results(self, ev: GatherResultsEvent) -> AnalyzeResultsEvent:
      # This runs automatically when gather_results emits a GatherResultsEvent
  ```

- **LLM calls** use `await self.llm.acomplete(prompt)` — you write a prompt describing the task and format, and the LLM returns structured text.

### Flow

```
StartEvent → gather_results → GatherResultsEvent
                                    ↓
                             analyze_results → AnalyzeResultsEvent
                                                      ↓
                                               generate_report → GenerateReportEvent
                                                                        ↓
                                                                    finalize → StopEvent
```

A straight-line pipeline: gather data, analyze it, write a report, output the result. Each step does one thing and passes its output to the next via an event.

### Run it

```bash
uv run python 0_reportgen.py
```

---

## Example 2: Agentic Compile-Debug Loop (`2_moreagentic.py`)

This example builds on the workflow concepts from Example 0 and introduces **agentic tool use** — giving the LLM access to tools it can call autonomously to investigate and fix problems.

### What it does

Takes a SystemVerilog design with intentional bugs (missing semicolon, double driver), compiles it, and when compilation fails, spawns a ReAct agent that can read files, search code, run lint, write fixes, and recompile — all on its own. The workflow loops until the design compiles clean or a retry limit is hit.

### Key concepts

- **Branching events** — a step can return different event types to take different paths through the workflow:

  ```python
  @step
  async def compile(self, ev: CompileEvent) -> CompileFailedEvent | StopEvent:
      # Return StopEvent if compilation passed, CompileFailedEvent if it didn't
  ```

- **Looping** — the `debug_and_fix` step emits a `CompileEvent`, which routes back to the `compile` step, creating a retry loop.

- **Tools** are plain Python functions wrapped with `FunctionTool.from_defaults()`. The LLM reads the docstring to understand when and how to use each tool:

  ```python
  def read_file(filepath: str) -> str:
      """Read the contents of a file. Useful for inspecting source files,
      headers, or package definitions referenced in error messages."""
      ...

  AGENT_TOOLS = [
      FunctionTool.from_defaults(fn=read_file),
      FunctionTool.from_defaults(fn=grep_source),
      FunctionTool.from_defaults(fn=run_lint),
      FunctionTool.from_defaults(fn=write_file),
      FunctionTool.from_defaults(fn=compile_design),
  ]
  ```

- **ReActAgent** — a reasoning-and-acting agent that decides which tools to call based on the error log. It receives a system prompt describing its role and a task prompt with the failing code and errors:

  ```python
  agent = ReActAgent(
      tools=AGENT_TOOLS,
      llm=self.llm,
      system_prompt="You are an expert SystemVerilog debug engineer...",
  )
  response = await agent.run(task)
  ```

### Flow

```
StartEvent → start → CompileEvent
                         ↓
                      compile ──── success ──→ StopEvent
                         │
                    CompileFailedEvent
                         ↓
                   debug_and_fix (ReAct agent investigates + fixes)
                         │
                    CompileEvent (loop back)
```

The compile → debug → recompile loop continues until the design passes or `max_attempts` is reached.

### Run it

```bash
uv run python 2_moreagentic.py
```

---

## Generating flow diagrams

You can visualize the workflow graphs as interactive HTML files:

```bash
uv run python draw_flows.py
```

This produces `flow_0_reportgen.html` and `flow_2_moreagentic.html` — open them in a browser to see the step and event relationships.

## Using a real LLM

Both examples use mock LLMs (`MockLLM1`, `MockLLM2` in `helper.py`) that return canned responses. To use a real model:

```python
# Anthropic
from llama_index.llms.anthropic import Anthropic
llm = Anthropic(model="claude-sonnet-4-20250514", max_tokens=4096)

# OpenAI
from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-4o", max_tokens=4096)
```

## File overview

| File | Purpose |
|------|---------|
| `0_reportgen.py` | Workflow basics — linear pipeline, LLM prompting |
| `2_moreagentic.py` | Agentic workflows — tool use, branching, retry loops |
| `helper.py` | Mock LLMs, test design, and canned response data |
| `draw_flows.py` | Generates interactive workflow diagrams |
| `1_basic.py` | Reference — minimal compile-debug loop (requires real LLM + VCS) |
