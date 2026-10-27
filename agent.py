
import json
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path

import mock_llm

from policy_assistant import (
    load_policy_documents,
    build_chunks,
    PolicyRetriever,
)

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

AGENT_RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"
REFUND_LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"


# =====================================================
# POLICY RETRIEVER
# =====================================================

_documents = load_policy_documents()
_chunks = build_chunks(_documents)
_retriever = PolicyRetriever(_chunks)


# =====================================================
# TOOL SPECIFICATIONS
# =====================================================

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order in the SQLite database.",
        "parameters": {
            "order_id": "string"
        },
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents for relevant rules.",
        "parameters": {
            "query": "string"
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate a refund for a late delivery.",
        "parameters": {
            "order_amount": "number",
            "delivery_minutes": "number"
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund.",
        "parameters": {
            "order_id": "string",
            "refund_inr": "number"
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund request for human review.",
        "parameters": {
            "order_id": "string",
            "refund_inr": "number"
        },
    },
]


# =====================================================
# TOOL IMPLEMENTATIONS
# =====================================================

def get_order(order_id):
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute(
            "SELECT * FROM orders WHERE order_id = ? LIMIT 1",
            (order_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return {"error": "order not found"}

        order = dict(row)

        return {
            "order_id": str(order.get("order_id", order_id)),
            "status": str(order.get("status", "")).lower(),
            "order_amount": float(
                order.get("order_amount", order.get("total_amount", 0)) or 0
            ),
            "delivery_minutes": float(
                order.get("delivery_minutes", 0) or 0
            ),
        }

    finally:
        connection.close()


def search_policy(query):
    results = _retriever.retrieve(query, k=3)

    return {
        "results": [
            {
                "source": item["source"],
                "score": item["score"],
                "text": item["text"],
            }
            for item in results
        ]
    }


def calculate_refund(order_amount, delivery_minutes):
    amount = round(float(order_amount), 2)
    minutes = float(delivery_minutes)

    if minutes > 45:
        refund = amount
    else:
        refund = 0.0

    return {
        "refund_inr": round(refund, 2),
        "reason": (
            "delivery exceeded 45 minutes"
            if refund > 0
            else "delivery was within 45 minutes"
        ),
    }


def _append_jsonl(path, data):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def _record_refund_action(tool_name, order_id, refund_inr):
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }

    _append_jsonl(REFUND_LEDGER_FILE, entry)

    return {
        "success": True,
        "order_id": order_id,
        "refund_inr": entry["refund_inr"],
        "tool_name": tool_name,
    }


def issue_refund(order_id, refund_inr):
    if float(refund_inr) <= 0:
        return {"error": "refund amount must be positive"}

    return _record_refund_action(
        "issue_refund",
        order_id,
        refund_inr,
    )


def escalate_to_human(order_id, refund_inr):
    return _record_refund_action(
        "escalate_to_human",
        order_id,
        refund_inr,
    )


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# =====================================================
# CHECKPOINT / RESUME
# =====================================================

def save_checkpoint(run_id, state):
    path = STATE_DIR / f"{run_id}.json"
    temporary_path = path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def load_checkpoint(run_id):
    path = STATE_DIR / f"{run_id}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return json.loads(path.read_text(encoding="utf-8"))


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    """Resume a saved run using its checkpointed conversation and tool history."""
    state = load_checkpoint(run_id)
    return run_agent(
        user_message=state["user_message"],
        scenario_id=state["scenario_id"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        run_id=run_id,
        resume=True,
    )


# =====================================================
# REACT AGENT
# =====================================================

SYSTEM_PROMPT = """
AGENT

You are an Instamart order and refund assistant.
Use the available tools to check orders and apply policy rules.
Only completed orders may be refunded.
A full refund is permitted when delivery takes more than 45 minutes.
Refunds above INR 1000 must be escalated to a human lead.
Never invent order details or refund amounts.
"""


def _fallback(status, message, steps, llm_calls, tokens_used):
    return {
        "status": status,
        "final_answer": message,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }


def run_agent(
    user_message,
    scenario_id="MANUAL",
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    resume=False,
    crash_after_tool_executions=None,
):
    run_id = run_id or str(uuid.uuid4())

    if resume:
        state = load_checkpoint(run_id)
        messages = state["messages"]
        steps = state["steps"]
        tokens_used = state["tokens_used"]
        llm_calls = state["llm_calls"]
        tool_executions = state["tool_executions"]
        tools_executed = state.get(
            "tools_executed",
            [step["tool"] for step in steps],
        )
        start_time = time.monotonic()
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        steps = []
        tokens_used = 0
        llm_calls = 0
        tool_executions = 0
        tools_executed = []
        start_time = time.monotonic()

        state = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "user_message": user_message,
            "messages": messages,
            "steps": steps,
            "tokens_used": tokens_used,
            "llm_calls": llm_calls,
            "tool_executions": tool_executions,
            "tools_executed": tools_executed,
            "step_number": len(steps),
        }

        save_checkpoint(run_id, state)

    while True:
        if len(steps) >= max_iterations:
            return _fallback(
                "FALLBACK_MAX_ITERATIONS",
                "FALLBACK_MAX_ITERATIONS: maximum agent iterations reached.",
                steps,
                llm_calls,
                tokens_used,
            )

        if time.monotonic() - start_time >= time_limit_seconds:
            return _fallback(
                "FALLBACK_TIMEOUT",
                "FALLBACK_TIMEOUT: time limit reached; human review may be needed.",
                steps,
                llm_calls,
                tokens_used,
            )

        try:
            response = mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            # Retry one simulated temporary failure.
            try:
                response = mock_llm.chat(messages)
            except Exception as error:
                return _fallback(
                    "FALLBACK_LLM_ERROR",
                    f"FALLBACK_LLM_ERROR: {error}",
                    steps,
                    llm_calls,
                    tokens_used,
                )

        llm_calls += 1
        tokens_used += response.usage.total_tokens

        if tokens_used > token_limit:
            return _fallback(
                "FALLBACK_TOKEN_LIMIT",
                "FALLBACK_TOKEN_LIMIT: token budget exceeded; requested tool was not executed.",
                steps,
                llm_calls,
                tokens_used,
            )

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", []) or []

        if not tool_calls:
            answer = (message.content or "").strip()
            status = answer.split(":", 1)[0].strip()

            result = {
                "status": status,
                "final_answer": answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

            state.update({
                "messages": messages,
                "steps": steps,
                "tokens_used": tokens_used,
                "llm_calls": llm_calls,
                "tool_executions": tool_executions,
                "tools_executed": tools_executed,
                "step_number": len(steps),
                "result": result,
            })

            save_checkpoint(run_id, state)

            return result

        # Add the assistant's tool-call message in the format
        # expected by the supplied mock LLM.
        serialised_calls = []

        for call in tool_calls:
            serialised_calls.append({
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            })

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": serialised_calls,
        })

        for call in tool_calls:
            name = call.name

            try:
                arguments = (
                    json.loads(call.arguments)
                    if isinstance(call.arguments, str)
                    else call.arguments
                )

                if name not in TOOL_MAP:
                    observation = {"error": f"unknown tool: {name}"}
                else:
                    observation = TOOL_MAP[name](**arguments)

            except Exception as error:
                observation = {"error": str(error)}

            tool_executions += 1
            tools_executed.append(name)

            steps.append({
                "tool": name,
                "arguments": arguments if "arguments" in locals() else {},
                "observation": observation,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(
                    observation,
                    ensure_ascii=False,
                ),
            })

            state.update({
                "messages": messages,
                "steps": steps,
                "tokens_used": tokens_used,
                "llm_calls": llm_calls,
                "tool_executions": tool_executions,
                "tools_executed": tools_executed,
                "step_number": len(steps),
            })

            # Save after every tool observation, so execution can resume.
            save_checkpoint(run_id, state)

            if (
                crash_after_tool_executions is not None
                and tool_executions >= crash_after_tool_executions
            ):
                raise RuntimeError(
                    "Simulated crash after tool execution."
                )

    # Unreachable, retained for readability.


# =====================================================
# SCENARIO BATCH
# =====================================================

def run_scenarios():
    import pandas as pd

    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(
            f"Scenario CSV not found: {SCENARIOS_PATH}"
        )

    scenarios = pd.read_csv(SCENARIOS_PATH)

    # A full batch run starts from clean ledgers so reproduction is repeatable.
    REFUND_LEDGER_FILE.write_text("", encoding="utf-8")
    with AGENT_RUNS_FILE.open("w", encoding="utf-8") as output:
        for _, row in scenarios.iterrows():
            scenario_id = str(row["scenario_id"])
            user_message = str(row["message"])

            mock_llm.reset()

            result = run_agent(
                user_message=user_message,
                scenario_id=scenario_id,
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10,
                run_id=f"{scenario_id}_{uuid.uuid4().hex[:8]}",
            )

            expected_status = str(row["expected_outcome"])
            if expected_status == "FALLBACK":
                expected_status = "FALLBACK_MAX_ITERATIONS"
            if result["status"] != expected_status:
                raise AssertionError(
                    f"{scenario_id} status {result['status']} "
                    f"does not match {expected_status}"
                )

            expected_refund = float(row["expected_refund_inr"])
            recorded_refunds = [
                float(step["arguments"]["refund_inr"])
                for step in result["steps"]
                if step["tool"] in {"issue_refund", "escalate_to_human"}
            ]
            actual_refund = recorded_refunds[-1] if recorded_refunds else 0.0
            if abs(actual_refund - expected_refund) > 0.01:
                raise AssertionError(
                    f"{scenario_id} refund INR {actual_refund:.2f} "
                    f"does not match INR {expected_refund:.2f}"
                )

            record = {
                "scenario_id": scenario_id,
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"],
            }

            output.write(json.dumps(record) + "\n")

            print(
                f"{scenario_id}: {result['status']} | "
                f"{result['final_answer']} | "
                f"calls={result['llm_calls']} | "
                f"tokens={result['tokens_used']}"
            )


def main():
    print("Loaded tools:")

    for tool in TOOLS_SPEC:
        print(f"- {tool['name']}: {tool['description']}")

    print("\nRunning agent scenarios...\n")
    run_scenarios()

    print("\nSaved agent runs:", AGENT_RUNS_FILE)
    print("Refund ledger:", REFUND_LEDGER_FILE)
    print("Checkpoints folder:", STATE_DIR)


if __name__ == "__main__":
    main()
