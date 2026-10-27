
import json
import uuid
from pathlib import Path

import mock_llm
from agent import run_agent, load_checkpoint, resume_agent

ROOT = Path(__file__).resolve().parent
SCENARIOS_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"

import pandas as pd

scenarios = pd.read_csv(SCENARIOS_FILE)
s1 = scenarios[scenarios["scenario_id"] == "S1"].iloc[0]
message = str(s1["message"])

print("\n--- TEST 1: TOKEN LIMIT ---")
mock_llm.reset()

token_result = run_agent(
    user_message=message,
    scenario_id="TEST_TOKEN",
    token_limit=200,
    time_limit_seconds=10,
    run_id="test_token_" + uuid.uuid4().hex[:8],
)

print("Status:", token_result["status"])
print("Tokens:", token_result["tokens_used"])
print("Steps:", len(token_result["steps"]))
assert token_result["status"] == "FALLBACK_TOKEN_LIMIT"

print("\n--- TEST 2: TIMEOUT ---")
mock_llm.reset()

timeout_result = run_agent(
    user_message=message,
    scenario_id="TEST_TIMEOUT",
    time_limit_seconds=0,
    run_id="test_timeout_" + uuid.uuid4().hex[:8],
)

print("Status:", timeout_result["status"])
print("LLM calls:", timeout_result["llm_calls"])
assert timeout_result["status"] == "FALLBACK_TIMEOUT"

print("\n--- TEST 3: CHECKPOINT AND RESUME ---")
mock_llm.reset()

run_id = "test_resume_" + uuid.uuid4().hex[:8]

try:
    run_agent(
        user_message=message,
        scenario_id="TEST_RESUME",
        run_id=run_id,
        time_limit_seconds=30,
        crash_after_tool_executions=2,
    )
except RuntimeError as error:
    print("Simulated crash:", error)

checkpoint = load_checkpoint(run_id)

print("Checkpoint saved:", True)
print("Tool executions before resume:", checkpoint["tool_executions"])
assert checkpoint["step_number"] == 2
assert checkpoint["tools_executed"] == ["get_order", "search_policy"]
print(
    "Tools before resume:",
    [step["tool"] for step in checkpoint["steps"]],
)

resumed = resume_agent(run_id, time_limit_seconds=30)

print("Resumed status:", resumed["status"])
print("Resumed answer:", resumed["final_answer"])
print("Tools after resume:", [step["tool"] for step in resumed["steps"]])
assert resumed["status"] == "REFUND"
assert "INR 128.25" in resumed["final_answer"]
assert [step["tool"] for step in resumed["steps"]] == [
    "get_order",
    "search_policy",
    "calculate_refund",
    "issue_refund",
]

print("\n--- TESTS FINISHED ---")
