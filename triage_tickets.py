
import csv
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import mock_llm


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET = 100000

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"


# --------------------------------------------------
# Prompt definitions
# --------------------------------------------------

ZERO_SHOT_SYSTEM = """
CLASSIFY each customer support ticket into exactly one category:
billing, delivery, technical, or fraud.

Return JSON only in this format:
{"category": "one allowed category", "reason": "short explanation"}
Do not include markdown or additional text.
"""

FEW_SHOT_SYSTEM = """
CLASSIFY each customer support ticket into exactly one category:
billing, delivery, technical, or fraud.

EXAMPLES:
Ticket: Someone used my account without permission and stole my money.
{"category": "fraud", "reason": "unauthorized account activity"}

Ticket: My app crashes whenever I open it.
{"category": "technical", "reason": "app crash"}

Ticket: I was charged twice for the same purchase.
{"category": "billing", "reason": "duplicate charge"}

Ticket: My delivery arrived late and the rider did not call.
{"category": "delivery", "reason": "late delivery"}

Return JSON only in this format:
{"category": "one allowed category", "reason": "short explanation"}
Do not include markdown or additional text.
"""


# --------------------------------------------------
# PII redaction
# --------------------------------------------------

def redact_pii_with_counts(text):
    """Redact PII in the required order and return per-pattern counts."""
    text, emails = re.subn(EMAIL_PATTERN, "[EMAIL]", text)
    text, phones = re.subn(PHONE_PATTERN, "[PHONE]", text)
    text, order_ids = re.subn(ORDER_PATTERN, "[ORDER_ID]", text)
    return text, {
        "emails": emails,
        "phones": phones,
        "order_ids": order_ids,
    }


def redact_pii(text):
    """Redact email, phone number and order ID in the required order."""
    return redact_pii_with_counts(text)[0]


def count_pii(text):
    """Count remaining PII patterns."""
    return {
        "emails": len(re.findall(EMAIL_PATTERN, text)),
        "phones": len(re.findall(PHONE_PATTERN, text)),
        "order_ids": len(re.findall(ORDER_PATTERN, text)),
    }


# --------------------------------------------------
# Ticket validation
# --------------------------------------------------

def prepare_ticket(row):
    """Apply empty-ticket and long-ticket guardrails."""
    original_text = str(row.get("text", "") or "")

    if not original_text.strip():
        return {
            "status": "skipped_empty",
            "category": "",
            "reason": "",
            "tokens_used": 0,
            "redacted_text": "",
        }

    token_count = mock_llm.est_tokens(original_text)

    if token_count > 400:
        return {
            "status": "flagged_long",
            "category": "",
            "reason": "",
            "tokens_used": 0,
            "redacted_text": "",
        }

    safe_text = redact_pii(original_text)
    if any(count_pii(safe_text).values()):
        raise AssertionError(f"PII remains after redaction for {row.get('ticket_id')}")

    return {
        "status": "ready",
        "category": "",
        "reason": "",
        "tokens_used": 0,
        "redacted_text": safe_text,
    }


# --------------------------------------------------
# LLM output validation
# --------------------------------------------------

def validate_response(response_text):
    """
    Return:
        status, category, reason
    """
    try:
        parsed = json.loads(response_text)

        if isinstance(parsed, dict):
            category = str(parsed.get("category", "")).lower().strip()
            reason = str(parsed.get("reason", "")).strip()

            if category in ALLOWED_CATEGORIES:
                return "classified_json", category, reason

            # JSON was returned, but its category is invalid.
            return "flagged_invalid_output", "", ""

    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: accept only when exactly one allowed category
    # word appears in the response.
    words = re.findall(r"[a-z]+", str(response_text).lower())
    found = [word for word in words if word in ALLOWED_CATEGORIES]

    if len(found) == 1:
        return "classified_keyword", found[0], "Recovered from keyword response"

    return "flagged_invalid_output", "", ""


# --------------------------------------------------
# Classification with retry logic
# --------------------------------------------------

def classify_ticket(ticket_id, safe_text, system_prompt):
    if any(count_pii(safe_text).values()):
        raise AssertionError(f"Unredacted PII would be sent for ticket {ticket_id}")
    messages = [
        {"role": "system", "content": system_prompt},
        # The user prompt contains ONLY the redacted ticket text.
        {"role": "user", "content": safe_text},
    ]
    if messages[-1]["content"] != safe_text:
        raise AssertionError("The model user message must be exactly the redacted ticket text")

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)
            response_text = response.choices[0].message.content or ""

            status, category, reason = validate_response(response_text)

            return {
                "status": status,
                "category": category,
                "reason": reason,
                "tokens_used": response.usage.total_tokens,
            }

        except mock_llm.TransientLLMError as error:
            if attempt >= MAX_RETRIES:
                print(
                    f"Ticket {ticket_id}: retries exhausted "
                    f"after {MAX_RETRIES} retries: {error}"
                )
                return {
                    "status": "flagged_invalid_output",
                    "category": "",
                    "reason": "Transient model error after retries",
                    "tokens_used": 0,
                }

            wait_time = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)

            print(
                f"Ticket {ticket_id}: retry number {attempt} "
                f"(attempt {attempt + 1}/{MAX_RETRIES}), waiting {wait_time:.3f}s"
            )
            time.sleep(wait_time)

    return {
        "status": "flagged_invalid_output",
        "category": "",
        "reason": "Classification failed",
        "tokens_used": 0,
    }


# --------------------------------------------------
# Run one classification experiment
# --------------------------------------------------

def run_experiment(tickets, system_prompt, experiment_name,
                   token_budget=100000):
    mock_llm.reset()
    random.seed(42)

    results = []
    total_tokens = 0

    for _, row in tickets.iterrows():
        ticket_id = str(row["ticket_id"])
        prepared = prepare_ticket(row)

        result = {
            "ticket_id": ticket_id,
            "customer_id": row.get("customer_id", ""),
            "order_id": row.get("order_id", ""),
            "true_category": row.get("true_category", ""),
            "status": prepared["status"],
            "category": "",
            "reason": "",
            "tokens_used": 0,
            "redacted_text": prepared["redacted_text"],
        }

        if prepared["status"] != "ready":
            results.append(result)
            continue

        # Do not call the model if the budget is already exhausted.
        if total_tokens >= token_budget:
            result["status"] = "budget_exhausted"
            results.append(result)
            continue

        classified = classify_ticket(
            ticket_id,
            prepared["redacted_text"],
            system_prompt,
        )

        result["status"] = classified["status"]
        result["category"] = classified["category"]
        result["reason"] = classified["reason"]
        result["tokens_used"] = classified["tokens_used"]

        total_tokens += classified["tokens_used"]
        results.append(result)

    df = pd.DataFrame(results)

    print(f"\n--- {experiment_name} ---")
    print("Total tickets:", len(df))
    print("Statuses:")
    print(df["status"].value_counts().to_string())
    print("Total model tokens:", total_tokens)

    classified_df = df[
        df["status"].isin(["classified_json", "classified_keyword"])
    ]

    if len(classified_df):
        accuracy = (
            classified_df["category"].str.lower()
            == classified_df["true_category"].astype(str).str.lower()
        ).mean()

        print("Classified tickets:", len(classified_df))
        print(f"Accuracy among classified tickets: {accuracy:.1%}")
        print(
            "Tokens per classified ticket:",
            round(total_tokens / len(classified_df), 2),
        )
    else:
        accuracy = 0.0
        print("Classified tickets: 0")
        print("Accuracy among classified tickets: N/A")

    return df, total_tokens, accuracy


# --------------------------------------------------
# Redaction audit
# --------------------------------------------------

def run_redaction_audit(tickets):
    print("\n--- PII REDACTION AUDIT ---")

    substitutions = {"emails": 0, "phones": 0, "order_ids": 0}
    after = {"emails": 0, "phones": 0, "order_ids": 0}

    for text in tickets["text"].fillna("").astype(str):
        safe_text, found = redact_pii_with_counts(text)
        remaining = count_pii(safe_text)

        for key in substitutions:
            substitutions[key] += found[key]
            after[key] += remaining[key]

    if any(after.values()):
        raise AssertionError(f"PII remains after redaction: {after}")

    total = sum(substitutions.values())
    print("PII substitutions:", substitutions)
    print("Total PII substitutions:", total)
    print("PII matches after reapplying patterns:", after)


# --------------------------------------------------
# Customer segment lookup and routing
# --------------------------------------------------

def load_premium_customers():
    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(f"Customer segment file not found: {SEGMENTS_FILE}")

    segments = pd.read_csv(SEGMENTS_FILE)

    if "customer_id" not in segments.columns:
        raise ValueError("customer_segments.csv is missing customer_id")

    if "is_premium" not in segments.columns:
        raise ValueError("customer_segments.csv is missing is_premium")

    premium = segments.loc[
        segments["is_premium"].astype(str).str.lower().eq("true"),
        "customer_id",
    ]

    return set(premium.astype(str))


def choose_route(status, category, customer_id, premium_customers):
    """Apply the project brief's routing and priority rules."""
    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if str(customer_id) in premium_customers:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


def make_final_results(few_shot_df):
    premium_customers = load_premium_customers()
    final_rows = []

    for _, row in few_shot_df.iterrows():
        status = row["status"]
        category = row["category"] if pd.notna(row["category"]) else ""
        customer_id = str(row["customer_id"])

        route, priority = choose_route(
            status, category, customer_id, premium_customers
        )

        segment_name = "Premium High-Value" if customer_id in premium_customers else ""

        final_rows.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": row["tokens_used"],
        })

    final_df = pd.DataFrame(final_rows)

    final_path = OUTPUT_DIR / "triage_results.csv"
    final_df.to_csv(final_path, index=False)
    print("\nSaved final results:", final_path)
    print(final_df["route"].value_counts().to_string())

    return final_df


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {RAW_FILE}. Check that data/raw/tickets.csv exists."
        )

    tickets = pd.read_csv(RAW_FILE, keep_default_na=False)

    required_columns = {
        "ticket_id", "customer_id", "order_id", "text", "true_category"
    }
    missing = required_columns - set(tickets.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print("Total tickets:", len(tickets))
    print("Columns:", tickets.columns.tolist())

    run_redaction_audit(tickets)

    # Zero-shot experiment: no token budget limit for comparison.
    zero_df, zero_tokens, zero_accuracy = run_experiment(
        tickets,
        ZERO_SHOT_SYSTEM,
        "ZERO-SHOT CLASSIFICATION",
        token_budget=100000,
    )

    zero_path = OUTPUT_DIR / "triage_zero_shot.csv"
    zero_df.to_csv(zero_path, index=False)
    print("Saved:", zero_path)

    zero_classified = zero_df[
        zero_df["status"].isin(["classified_json", "classified_keyword"])
    ]

    if len(zero_classified):
        print("\nZero-shot confusion table:")
        print(pd.crosstab(
            zero_classified["true_category"],
            zero_classified["category"],
            rownames=["Actual"],
            colnames=["Predicted"],
        ).to_string())

    # Few-shot experiment: this becomes the final routing result.
    few_df, few_tokens, few_accuracy = run_experiment(
        tickets,
        FEW_SHOT_SYSTEM,
        "FEW-SHOT CLASSIFICATION",
        token_budget=100000,
    )

    few_path = OUTPUT_DIR / "triage_few_shot.csv"
    few_df.to_csv(few_path, index=False)
    print("Saved:", few_path)

    # Budget demonstration with a deliberately smaller budget.
    budget_df, budget_tokens, _ = run_experiment(
        tickets,
        ZERO_SHOT_SYSTEM,
        "ZERO-SHOT BUDGET DEMO",
        token_budget=6000,
    )

    budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
    budget_df.to_csv(budget_path, index=False)
    print("Saved:", budget_path)

    print("\n--- ZERO-SHOT VS FEW-SHOT ---")
    print(f"Zero-shot accuracy: {zero_accuracy:.1%}")
    print(f"Few-shot accuracy:  {few_accuracy:.1%}")
    print("Zero-shot total tokens:", zero_tokens)
    print("Few-shot total tokens:", few_tokens)

    make_final_results(few_df)

    print("\nPart C classification pipeline completed.")


if __name__ == "__main__":
    main()
