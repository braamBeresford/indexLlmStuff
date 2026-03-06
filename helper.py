from typing import Any

from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.custom import CustomLLM


MOCK_ANALYSIS = """\
THIS IS AI GENERATED FILLER SLOP:
## Per-Suite Pass Rates

| Suite       | Total | Pass | Fail | Error | Pass Rate |
|-------------|-------|------|------|-------|-----------|
| functional  | 10    | 9    | 1    | 0     | 90%       |
| coverage    | 5     | 3    | 2    | 0     | 60%       |
| timing      | 5     | 4    | 1    | 0     | 80%       |
| lint        | 4     | 3    | 1    | 0     | 75%       |
| power       | 4     | 2    | 2    | 0     | 50%       |

## Failure Categorization

1. **Coverage gaps** (2 failures): line_coverage_check fell below 80% threshold; \
fsm_coverage_check missed IDLE→ERROR transition. Root cause: insufficient stimulus \
in testbench for error-path scenarios.
2. **Power budget violations** (2 failures): dynamic_power_estimate exceeded 1.50W budget; \
clock_gating_efficiency at 62% vs 75% target. Root cause: clock gating not applied to \
data_bus pipeline registers.
3. **Timing violation** (1 failure): setup_hold_1ghz has -0.032ns slack on clk→data_reg path. \
Root cause: likely missing pipeline stage at 1GHz target.
4. **Lint issue** (1 failure): lint_rtl_subsystem_b has dead logic on debug_bus[31:16]. \
Root cause: debug signals left unconnected after last RTL refactor.
5. **Functional bug** (1 failure): dma_scatter_gather_test scoreboard mismatch (509 vs 512 \
transactions). Root cause: possible off-by-one in DMA descriptor chain handling.

## Severity Assessment (highest first)

1. **CRITICAL** — dma_scatter_gather_test: data integrity issue, blocks tapeout
2. **HIGH** — setup_hold_1ghz: timing closure required for target frequency
3. **HIGH** — dynamic_power_estimate: power budget exceeded by 21%
4. **MEDIUM** — coverage gaps: need stimulus improvement before signoff
5. **LOW** — lint dead logic: cleanup item, no functional impact

## Infrastructure Issues

No infrastructure failures detected in this run. All failures are design-related.
"""

MOCK_REPORT = """\
THIS IS AI GENERATED FILLER SLOP:
# Daily Regression Health Report — 2026-03-06

## Executive Summary

Overnight regression completed 28 tests across 5 suites with a **75% overall pass rate** \
(21 pass, 7 fail, 0 infra errors). The top risk is a **functional data integrity bug** in \
the DMA scatter-gather path, followed by **timing closure failure** at the 1GHz target.

## Suite Breakdown

| Suite       | Total | Pass | Fail | Error | Pass Rate |
|-------------|-------|------|------|-------|-----------|
| functional  | 10    | 9    | 1    | 0     | 90%       |
| coverage    | 5     | 3    | 2    | 0     | 60%       |
| timing      | 5     | 4    | 1    | 0     | 80%       |
| lint        | 4     | 3    | 1    | 0     | 75%       |
| power       | 4     | 2    | 2    | 0     | 50%       |
| **Total**   | **28**| **21**| **7**| **0** | **75%**   |

## Critical Failures

1. **dma_scatter_gather_test** (functional) — Scoreboard mismatch: expected 512 transactions, \
observed 509. Likely off-by-one in descriptor chain. **Action: block until resolved.**
2. **setup_hold_1ghz** (timing) — Setup violation of -0.032ns on clk→data_reg[7]/D. \
Needs pipeline register insertion or clock adjustment.
3. **dynamic_power_estimate** (power) — 1.82W vs 1.50W budget at TT/0.9V/85C. \
Clock gating coverage insufficient on data pipeline.

## Infrastructure Issues

None. All failures are design-related.

## Recommendations

1. **[P0]** Investigate DMA scatter-gather transaction drop — assign to DMA owner immediately
2. **[P0]** Review 1GHz timing path; consider adding pipeline stage or relaxing constraint
3. **[P1]** Improve clock gating on data_bus pipeline to reduce dynamic power by ~20%
4. **[P1]** Add error-injection stimulus to cover FSM IDLE→ERROR transition
5. **[P2]** Clean up dead debug_bus signals in subsystem_b

## Trend Note

This is the first automated run. Trending data (pass-rate over time, flaky test detection) \
will be available after subsequent regression cycles.
"""

TEST_DESIGN = """\
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

FIXED_DESIGN = """\
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

    assign valid = clk_en & |data_bus;

endmodule
"""


class MockLLM1(CustomLLM):
    """Mock LLM for regression report workflow (0_reportgen.py)."""

    @classmethod
    def class_name(cls) -> str:
        return "MockLLM1"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(num_output=4096)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        if "structured analysis" in prompt.lower():
            return CompletionResponse(text=MOCK_ANALYSIS)
        elif "health report" in prompt.lower():
            return CompletionResponse(text=MOCK_REPORT)
        return CompletionResponse(text="[MockLLM1] No canned response for this prompt.")

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = self.complete(prompt, **kwargs)

        def gen() -> CompletionResponseGen:
            yield response

        return gen()


class MockLLM2(CustomLLM):
    """Mock LLM for SV compile-fix workflow (2_moreagentic.py)."""

    @classmethod
    def class_name(cls) -> str:
        return "MockLLM2"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(num_output=4096)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(
            text=(
                "The errors are: (1) missing semicolon after "
                "`assign valid = clk_en & |data_bus` and (2) double driver — "
                "`valid` is assigned twice. Fix: add semicolon and remove "
                "the second assign.\n\n"
                "```systemverilog\n" + FIXED_DESIGN + "```"
            )
        )

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = self.complete(prompt, **kwargs)

        def gen() -> CompletionResponseGen:
            yield response

        return gen()
