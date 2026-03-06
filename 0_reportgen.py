import asyncio
import json
import random
from datetime import datetime, timedelta
from llama_index.core.llms import LLM
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

from helper import MockLLM1


# --- Events ---
class GatherResultsEvent(Event):
    results: str  # JSON-encoded regression results


class AnalyzeResultsEvent(Event):
    results: str
    analysis: str


class GenerateReportEvent(Event):
    results: str
    analysis: str
    report: str


# --- Mock data generation ---
def generate_mock_regressions() -> list[dict]:
    """Generate realistic-looking overnight regression results."""
    suites = {
        "functional": [
            "alu_add_sub_test",
            "alu_mul_div_test",
            "cache_coherency_test",
            "cache_writeback_test",
            "interrupt_priority_test",
            "interrupt_nesting_test",
            "dma_burst_transfer_test",
            "dma_scatter_gather_test",
            "branch_predictor_test",
            "pipeline_hazard_test",
        ],
        "coverage": [
            "line_coverage_check",
            "toggle_coverage_check",
            "fsm_coverage_check",
            "assertion_coverage_check",
            "cross_coverage_check",
        ],
        "timing": [
            "setup_hold_500mhz",
            "setup_hold_1ghz",
            "clock_domain_crossing_check",
            "multicycle_path_check",
            "max_frequency_test",
        ],
        "lint": [
            "lint_rtl_top",
            "lint_rtl_subsystem_a",
            "lint_rtl_subsystem_b",
            "lint_tb_top",
        ],
        "power": [
            "dynamic_power_estimate",
            "leakage_power_estimate",
            "clock_gating_efficiency",
            "power_domain_switching_test",
        ],
    }

    failure_messages = {
        "functional": [
            "ASSERTION FAILED: expected data_out == 0xDEAD_BEEF, got 0x0000_0000 at time 14523ns",
            "TIMEOUT: test did not complete within 50000 cycles",
            "ERROR: scoreboard mismatch — expected 512 transactions, observed 509",
        ],
        "coverage": [
            "FAIL: line coverage 78.2% < threshold 80%",
            "FAIL: FSM state IDLE→ERROR transition never observed",
        ],
        "timing": [
            "VIOLATION: setup time -0.032ns on path clk→data_reg[7]/D (required 0.050ns)",
            "WARNING→FAIL: 3 unconstrained clock domain crossings detected",
        ],
        "lint": [
            "ERROR: signal 'debug_bus[31:16]' is never read (dead logic)",
            "WARNING: inferred latch for 'state_next' in always_comb block",
        ],
        "power": [
            "FAIL: dynamic power 1.82W exceeds budget 1.50W at TT/0.9V/85C",
            "FAIL: clock gating efficiency 62% < target 75%",
        ],
    }

    random.seed(42)  # reproducible mock data
    base_time = datetime(2026, 3, 6, 1, 0, 0)  # overnight run starts at 1 AM
    results = []

    for suite_name, tests in suites.items():
        for test_name in tests:
            duration = random.uniform(10, 600)
            start_time = base_time + timedelta(seconds=random.uniform(0, 3600))

            # ~20% failure rate overall, varies by suite
            fail_rate = {
                "functional": 0.15,
                "coverage": 0.3,
                "timing": 0.25,
                "lint": 0.2,
                "power": 0.3,
            }
            status = "fail" if random.random() < fail_rate[suite_name] else "pass"

            # Small chance of error (infra issue)
            if status == "pass" and random.random() < 0.03:
                status = "error"

            entry = {
                "suite": suite_name,
                "test": test_name,
                "status": status,
                "duration_s": round(duration, 1),
                "start_time": start_time.isoformat(),
            }

            if status == "fail":
                entry["error_log"] = random.choice(failure_messages[suite_name])
            elif status == "error":
                entry["error_log"] = "INFRA ERROR: license checkout timeout after 300s"

            results.append(entry)

    return results


# --- Workflow ---
class RegressionReportWorkflow(Workflow):
    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    # This step will run first because it accepts an Event of type StartEvent
    # we retun/emit a GatherResultsEvent() to call the analyze_results step
    async def gather_results(self, ev: StartEvent) -> GatherResultsEvent:
        print(f"\n{'=' * 60}")
        print(f"Regression Report Generator")
        print(f"{'=' * 60}\n")

        regressions = generate_mock_regressions()

        # Print quick summary
        total = len(regressions)
        passed = sum(1 for r in regressions if r["status"] == "pass")
        failed = sum(1 for r in regressions if r["status"] == "fail")
        errors = sum(1 for r in regressions if r["status"] == "error")
        print(
            f"[Gather] Loaded {total} test results: {passed} pass, {failed} fail, {errors} error"
        )

        results_json = json.dumps(regressions, indent=2)
        return GatherResultsEvent(results=results_json)

    @step
    # We catch a GatherResultsEvent from gather_results, we query an LLM to analyze the
    # result. We provide a "prompt" that tells the LLM what task to do and what format to use
    # Emit a AnalyzeResultsEvent to call generate_report
    async def analyze_results(self, ev: GatherResultsEvent) -> AnalyzeResultsEvent:
        print(f"[Analyze] Asking LLM to analyze regression patterns...")

        prompt = f"""You are a senior ASIC/FPGA verification engineer reviewing overnight regression results.

Analyze the following regression data and provide a structured analysis:

1. **Per-suite pass rates** — calculate pass/fail/error counts for each suite
2. **Failure categorization** — group failures by root cause pattern (e.g., timing violations, coverage gaps, functional bugs)
3. **Severity assessment** — rank which failures are most critical to address first
4. **Flaky/infra issues** — flag any tests that failed due to infrastructure rather than design issues

## Regression data:
```json
{ev.results}
```

Provide your analysis in a structured format. Be specific about test names and error messages.
"""

        # Feed the prompt to the ULM
        response = await self.llm.acomplete(prompt)
        analysis = response.text  # Extract text from llm response
        print(f"[Analyze] Analysis complete ({len(analysis)} chars)")

        return AnalyzeResultsEvent(results=ev.results, analysis=analysis)

    @step
    # In this example we have generate report as a seperate llm call, we can also just make
    # it part of analyze_results() instead of creating intermediary json file.
    # Returns GenerateReportEvent to summon finalize() step
    async def generate_report(self, ev: AnalyzeResultsEvent) -> GenerateReportEvent:
        print(f"[Report] Asking LLM to generate the health report...")

        prompt = f"""You are a senior verification engineer writing the daily regression health report
for the engineering team and management.

Based on the raw regression data and your analysis below, produce a concise,
actionable report in Markdown format.

## Report structure:
1. **Executive Summary** — 2-3 sentences: overall health, pass rate, top risk
2. **Suite Breakdown** — table with suite name, total tests, pass/fail/error counts, pass rate %
3. **Critical Failures** — list the most important failures with test name, suite, and error summary
4. **Infrastructure Issues** — any tests that failed due to infra (licenses, timeouts, etc.)
5. **Recommendations** — prioritized action items for the team
6. **Trend Note** — (since this is the first run, just note that trending data will be available after subsequent runs)

## Raw regression data:
```json
{ev.results}
```

## Analysis:
{ev.analysis}

Write the report now. Be concise and actionable.
"""
        # Async to prevent blocking
        response = await self.llm.acomplete(prompt)
        report = response.text
        print(f"[Report] Report generated ({len(report)} chars)")

        return GenerateReportEvent(
            results=ev.results, analysis=ev.analysis, report=report
        )

    @step
    # Catch GenerateReportEvent from generate_report()
    # Print out the details of the report.
    # Returns the StopEvent which indicates the end of this WorkFlow
    async def finalize(self, ev: GenerateReportEvent) -> StopEvent:
        print(f"\n{'=' * 60}")
        print(f"REGRESSION HEALTH REPORT")
        print(f"{'=' * 60}\n")
        print(ev.report)

        return StopEvent(
            result={
                "status": "complete",
                "report": ev.report,
            }
        )


# --- Run it ---
async def main():
    # Using a MockLLM with canned responses for PoC demo
    # Swap to Anthropic/OpenAi for real usage
    # from llama_index.llms.openai import OpenAI
    # llm = OpenAI(model="gpt-4o", max_tokens=4096)
    llm = MockLLM1()

    # Create the workflow with a timeout
    wf = RegressionReportWorkflow(llm=llm, timeout=120)
    # Run the workflow
    result = await wf.run()

    print(f"\n{'=' * 60}")
    print(f"Done. Status: {result['status']}")


if __name__ == "__main__":
    asyncio.run(main())
