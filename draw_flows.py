"""Generate workflow flow diagrams for both examples."""

from helper import MockLLM1, MockLLM2
from llama_index.utils.workflow import draw_all_possible_flows
from importlib import import_module

# 0_reportgen.py
mod0 = import_module("0_reportgen")
wf0 = mod0.RegressionReportWorkflow(llm=MockLLM1(), timeout=120)
draw_all_possible_flows(wf0, filename="flow_0_reportgen.html")
print("Wrote flow_0_reportgen.html")

# 1_moreagentic.py
mod1 = import_module("1_moreagentic")
wf1 = mod1.SVCompileWorkflow(llm=MockLLM2(), timeout=120)
draw_all_possible_flows(wf1, filename="flow_1_moreagentic.html")
print("Wrote flow_1_moreagentic.html")
