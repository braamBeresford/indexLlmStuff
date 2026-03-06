"""Generate workflow flow diagrams for all three examples."""

from helper import MockLLM1, MockLLM2
from llama_index.utils.workflow import draw_all_possible_flows

# 0_reportgen.py
from importlib import import_module

mod0 = import_module("0_reportgen")
wf0 = mod0.RegressionReportWorkflow(llm=MockLLM1(), timeout=120)
draw_all_possible_flows(wf0, filename="flow_0_reportgen.html")
print("Wrote flow_0_reportgen.html")

# 1_basic.py
mod1 = import_module("1_basic")
from llama_index.core.llms.mock import MockLLM
wf1 = mod1.SVCompileWorkflow(llm=MockLLM(), timeout=120)
draw_all_possible_flows(wf1, filename="flow_1_basic.html")
print("Wrote flow_1_basic.html")

# 2_moreagentic.py
mod2 = import_module("2_moreagentic")
wf2 = mod2.SVCompileWorkflow(llm=MockLLM2(), timeout=120)
draw_all_possible_flows(wf2, filename="flow_2_moreagentic.html")
print("Wrote flow_2_moreagentic.html")
