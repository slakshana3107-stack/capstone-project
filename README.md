# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements a four-part customer intelligence and support workflow over deterministic, synthetic data. It cleans and stores the exports, segments customers and predicts lapse, triages tickets with privacy guardrails, and provides policy retrieval plus a resumable refund agent. All amounts are in Indian Rupees (INR); all model calls use the supplied offline `mock_llm.py`.

## Reproduce

Run these numbered commands from the repository root on Python 3.11 or newer. The first command creates an isolated environment; the following commands regenerate data, rebuild all artifacts, validate the database and execute the demos in dependency order.

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python audit_data.py`
6. `python clean_products.py`
7. `python clean_data.py`
8. `python clean_orders.py`
9. `python create_database.py`
10. `python validate_data.py`
11. `python run_queries.py`
12. `python analysis.py`
13. `python segment_customers.py`
14. `python build_lapse_model.py`
15. `python triage_tickets.py`
16. `python policy_assistant.py`
17. `python agent.py`
18. `python test_agent.py`

`generate_data.py` and `mock_llm.py` are the supplied, unmodified files. `requirements.txt` records the installed package versions with `pip freeze`; `.gitignore` excludes `.env`, `.venv/` and `__pycache__/`. `validate_data.py` prints exactly `ALL CHECKPOINTS PASSED` after all of its reference assertions succeed. The pipeline writes `data/clean/{customers,products,orders}.csv`, `data/instamart.db`, `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`, `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`, both policy evaluation CSVs, the agent JSONL logs and the `state/` checkpoints.

## Part A — Data Foundation: Clean, Store, Query

### Raw data audit

The audit script reports each CSV's shape, inferred data types, and missing count and percentage for every column. Blank ticket text and blank ticket order IDs are preserved as empty strings on read and counted as missing for the audit.

| Raw file | Shape | Column data types and missing count (percentage) |
|---|---:|---|
| `customers.csv` | 1,200 × 5 | `customer_id` str 0 (0%); `city` str 0 (0%); `signup_date` str 0 (0%); `age` float64 61 (5.08%); `gender` str 30 (2.50%) |
| `products.csv` | 40 × 4 | `product_id` str 0 (0%); `product_name` str 0 (0%); `category` str 0 (0%); `price_inr` int64 0 (0%) |
| `orders.csv` | 13,084 × 9 | `order_id` str 0 (0%); `customer_id` str 0 (0%); `product_id` str 0 (0%); `quantity` int64 0 (0%); `order_ts` str 0 (0%); `status` str 0 (0%); `discount_pct` int64 0 (0%); `order_amount` str 53 (0.41%); `delivery_minutes` float64 2,541 (19.42%) |
| `tickets.csv` | 164 × 5 | `ticket_id` str 0 (0%); `customer_id` str 0 (0%); `order_id` str 8 (4.88%); `text` str 4 (2.44%); `true_category` str 0 (0%) |
| `policy_eval.csv` | 14 × 3 | `question` str 0 (0%); `expected_doc` str 0 (0%); `expected_fact` str 2 (14.29%) |
| `agent_scenarios.csv` | 7 × 5 | `scenario_id` str 0 (0%); `customer_id` str 0 (0%); `message` str 0 (0%); `expected_outcome` str 0 (0%); `expected_refund_inr` float64 0 (0%) |

| Raw audit measure | Count |
|---|---:|
| Order rows / exact duplicate rows / pending rows / distinct order IDs | 13,084 / 119 / 656 / 12,309 |
| Unparseable `order_amount` entries / negative `delivery_minutes` / blank `delivery_minutes` | 120 / 106 / 2,541 |
| Blank customer ages / ages outside 10–100 / distinct raw city spellings | 61 / 6 / 24 |
| Distinct non-blank raw gender spellings / blank genders | 10 / 30 |

### Cleaning and storage

| Cleaning rule | Recorded changes and resulting checkpoint |
|---|---|
| Customers: trim city whitespace and title-case city values | 476 values changed; six cities remain: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. |
| Customers: map gender spelling variants to `male`, `female` or `unknown` | 950 values changed; final counts are female 610, male 560 and unknown 30. |
| Customers: treat ages outside 10–100 as missing, then fill missing ages using each city's median with `groupby` and `transform` | 6 out-of-range values invalidated; 67 missing/invalid ages imputed; 0 missing remain and the resulting range is 18–60. |
| Customers: convert `signup_date` to datetime | 1,200 dates converted. |
| Products: remove exact duplicates, trim text fields and convert `price_inr` to numeric | 0 duplicate rows removed; 0 text values needed trimming; all 40 prices converted and 0 were unparseable; 40 products retained. |
| Orders: convert `order_ts`, remove exact duplicates, then keep the latest timestamp for each order ID | 13,084 timestamps converted; 119 exact duplicates removed, then 656 earlier/duplicate order-ID records discarded; 12,309 final orders, one per ID, with no pending rows. |
| Orders: convert amounts to numeric and fill missing amounts using rounded catalogue price × quantity × (1 − discount/100) | 112 missing/unparseable amounts after order deduplication filled; all 12,309 quantity and discount values parsed as numeric; no missing amount remains. |
| Orders: set negative delivery times missing, impute completed-order gaps using the completed-order median for that product category, and keep cancelled delivery times missing | 105 negative values invalidated; 774 missing completed delivery values imputed; all 11,110 completed orders have delivery times; all 1,199 cancelled orders retain missing delivery times. |

The final order table has 11,110 completed and 1,199 cancelled orders; completed revenue is **INR 2,998,668.80**. Clean CSVs are in `data/clean/`; `data/instamart.db` contains `customers`, `products` and `orders`. The SQL database stores timestamps as `YYYY-MM-DD HH:MM:SS`. The validation script asserts more than eight reference conditions, including row counts, city and gender counts, age bounds, order statuses, missing-value rules and completed revenue.

### SQL analysis

`run_queries.py` reads and runs all seven statements from `sql/queries.sql` with `pandas.read_sql_query`.

| Query | Result |
|---|---|
| Q1 — completed orders above INR 500, grouped and filtered by HAVING total above INR 8,000 | 30 customers |
| Q2 — completed revenue by city and product category | 30 cells; total INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3 — products priced above average catalogue price | 11 products |
| Q4 — customers without a completed order | 25 customers |
| Q5 — customers whose completed spend is above the average customer total | 550 customers; average INR 2,552.06 |
| Q6 — RANK, DENSE_RANK and ROW_NUMBER for every customer with completed spend, partitioned by city | 1,175 customers ranked. Top customers: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10). Eight rows have RANK different from ROW_NUMBER. |
| Q7 — daily completed revenue and cumulative total | 183 days; final cumulative revenue INR 2,998,668.80 |

Tied spend values receive the same RANK, while ROW_NUMBER assigns each row a distinct position, so ties make those values differ.

### Pandas cross-check and exploration

The Q2 pivot table uses `pd.pivot_table(..., margins=True)`; its grand total equals SQL at INR 2,998,668.80.

| City | Completed amount sum (INR) | Mean (INR) | Order count |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

For completed order amounts, mean = INR 269.91, median = INR 135.00, skewness = 4.61, minimum = INR 12.00 and maximum = INR 3,998.00. Q1 = INR 64.80, Q3 = INR 297.00, IQR = INR 232.20, lower fence = INR −283.50 and upper fence = INR 645.30; there are 0 observations below the lower fence and 897 above the upper fence.

Completed orders describe customer purchases across the six cities and five product categories. The amount distribution is strongly right-skewed: its INR 269.91 mean is above its INR 135.00 median, with a long upper tail and 897 upper-fence outliers. No negative-fence outliers occur; large baskets and higher-priced products can explain some extreme values, while dirty or implausible entries were handled during cleaning. The largest absolute pairwise correlation among amount, discount, quantity and delivery time is order amount with discount percentage (−0.095); this is weak, and a correlation of this size does not establish causation.

## Part B — Customer Segments and Lapse Prediction

### RFM segments

As of 2025-10-01, the RFM table contains 1,175 customers. Recency median/minimum/maximum are 19/0/180 days; frequency mean/maximum are 9.46/33; monetary mean/maximum are INR 2,552.06 / INR 14,766.55.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.76 | 0.465 |
| 3 | 1,268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

We use K=4: the elbow is pronounced at four clusters, and K=4 also has the highest silhouette score in the tested range, while four groups allow distinct campaign actions. Inertia necessarily decreases as K increases because adding centroids can only reduce within-cluster distances; K=8's lower inertia is not evidence that eight is the best business segmentation, since the score rewards more clusters and does not measure interpretability or useful separation by itself.

| Segment ID | Business name | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Regular Customers | 436 | 28.86 | 3.77 | 1,057.62 | Recommend relevant products and suitable bundles. |
| 1 | Premium High-Value (**Premium**) | 138 | 24.12 | 6.77 | 7,129.70 | Provide personalised service and exclusive offers. |
| 2 | Frequent Shoppers | 419 | 8.77 | 19.40 | 3,410.02 | Encourage repeat purchases with loyalty rewards. |
| 3 | At-Risk Occasional | 182 | 109.04 | 2.20 | 685.99 | Send a re-engagement offer to encourage another order. |

Cluster IDs are arbitrary labels, not an ordered customer score. The Premium label is assigned to the segment with the highest mean monetary value. PCA on scaled RFM explains 63.43% in component 1 and 21.80% in component 2, 85.23% total. `outputs/customer_segments.csv` contains the required seven columns.

### Lapse cohort, features and preprocessing

The cutoff is 2025-09-01. The cohort has 1,156 customers with a completed pre-cutoff order; 443 (38.32%) have no completed order in September and are labelled lapsed. The stratified split is 924 training rows and 232 untouched test rows (570 not-lapsed / 354 lapsed in training; 143 / 89 in test).

Features use orders strictly before 2025-09-01: recency, frequency, monetary spend, average discount, cancellation rate across final-status orders, average completed delivery time, favourite product category, city, age, gender and tenure in days from the cutoff to `signup_date`. No September-or-later order entered the feature set, and task 3 segment labels were not used as features. The feature matrix has 0 missing values, asserted in code. Numeric imputation, scaling and categorical imputation/one-hot encoding are inside the scikit-learn training pipeline; each CV fold fits its own preprocessing on its training fold, and final preprocessing is fitted only on training rows. No encoder, scaler, imputer or resampler saw test rows.

The always-not-lapsed baseline has 61.64% test accuracy and lapsed-class F1 of 0. The fitted scaler's training-only frequency mean is 8.1147, versus 8.0285 for the full cohort; the means differ because the scaler was fitted only on the training rows.

### Model tuning and test results

Both searches use `GridSearchCV(cv=5, scoring="f1")` on the training data. Logistic regression uses `max_iter=1000` and C ∈ {0.01, 0.1, 1, 10, 100}; the best C is 100 with CV F1 0.6506. The random forest uses 100 trees, `random_state=42`, max depth ∈ {3, 5, 8} and minimum leaf size ∈ {5, 10, 20} (45 fits); the best parameters are depth 8 and leaf size 5, with CV F1 0.6933.

| Model | Confusion matrix (rows actual 0/1; columns predicted 0/1) | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---:|---:|
| Logistic regression | `[[111, 32], [31, 58]]` | 0.6444 | 0.6517 | 0.6480 |
| Random forest | `[[105, 38], [35, 54]]` | 0.5870 | 0.6067 | 0.5967 |

We select the random forest because it has the strongest cross-validated F1 (0.6933), which is the model-selection criterion used before the held-out test evaluation. Missing a lapsing customer can forfeit future orders, while wrongly targeting a loyal customer wastes campaign spend and may annoy them. The forest identifies 54 of 89 lapsed customers and reaches a lapsed-class F1 of 0.5967, above the 0.55 minimum, although logistic regression scores higher on this test split. The test set was used only after tuning to report final metrics.

### Class balance, interpretation and persistence

Oversampling duplicates only lapsed training rows with replacement until both classes have 570 rows; the test set remains unchanged.

| Logistic regression | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training balance | 0.6444 | 0.6517 | 0.6480 |
| Balanced training data | 0.5891 | 0.8539 | 0.6972 |

Recall moves most, increasing by 0.2022, while precision falls slightly. The retention team can find more potential lapsers, but should expect more loyal customers to be targeted and should choose campaign intensity accordingly.

| Top random-forest feature | Importance | Business reading |
|---|---:|---|
| Frequency | 0.2685 | Ordering frequency is the strongest predictor in this model. |
| Monetary | 0.2423 | Higher historical spend materially informs lapse risk. |
| Recency | 0.1130 | How recently a customer ordered contributes to risk. |
| Average delivery time | 0.0651 | Delivery experience contributes predictive signal. |
| Cancellation rate | 0.0566 | A history of cancellations contributes to the prediction. |

| Logistic coefficient | Sign | Business reading |
|---|---|---|
| Frequency (−1.6982) | Negative | More completed orders are associated with lower lapse odds. |
| Favourite category: Electronics (−1.1367) | Negative | Relative to the dropped reference category, this category is associated with lower lapse odds. |
| Favourite category: Grocery (−0.4487) | Negative | Relative to the dropped reference category, this category is associated with lower lapse odds. |

The selected random forest is saved to `outputs/lapse_model.joblib`; reloading it reproduces all test predictions exactly. The top 10 Premium customers in the test set are written to `outputs/retention_list.csv`; 1 of these 10 actually lapsed. The Premium flag uses whole-window RFM and therefore includes the September period used for the label: **the retention list is illustrative and not a leakage-free evaluation**.

## Part C — Support Ticket Triage with Guardrails

The pipeline processes the 164 tickets in file order. It reads tickets with `keep_default_na=False`; 4 empty/whitespace tickets are skipped, and 4 long tickets (734 estimated tokens each) are flagged without model calls. The input limit is 400 tokens. The user message sent to the model is exactly the redacted ticket text. Before calls and in the full-ticket audit, assertions verify that reapplying the email, phone and order-ID patterns finds no matches in redacted text.

| PII pattern | Substitutions |
|---|---:|
| Email | 55 |
| Phone | 91 |
| Order ID | 132 |
| **Total** | **278** |

Zero-shot and few-shot system prompts both name the allowed categories and require JSON-only output; the few-shot prompt includes labelled examples, including fraud. The retry wrapper uses up to 3 retries with exponential backoff and jitter; one log line is printed per retry. In each full run, 19 simulated transient failures were recovered; each was retry number 0 and waited between 0.045 and 0.0575 seconds. Retry waits are BASE_DELAY × 2ⁿ × uniform(0.9, 1.15), with BASE_DELAY 0.05 seconds, and the jitter RNG is seeded for repeatability.

| Run | Validated classified tickets | Accuracy | Total response tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% (126/147) | 11,428 | 77.74 |
| Few-shot | 147 | 100% (147/147) | 23,752 | 161.58 |

The full comparison outputs are `outputs/triage_zero_shot.csv` and `outputs/triage_few_shot.csv`.

The zero-shot confusion table counts only JSON or keyword-validated categories:

| Actual \ Predicted | Billing | Delivery | Fraud | Technical |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Fraud | 11 | 0 | 23 | 0 |
| Technical | 10 | 0 | 0 | 29 |

The few-shot run costs 83.84 additional tokens per classified ticket (about 2.08× the zero-shot cost) and corrected all 21 zero-shot misclassifications; those errors included 11 fraud and 10 technical tickets predicted as billing. On this synthetic dataset the added cost is defensible where a missed fraud report or misrouted technical issue is costly, but the improvement needs validation on independent tickets before generalizing it.

The budget demonstration is saved to `outputs/triage_budget_demo.csv` and uses a 6,000-token budget. It produced 78 validated classifications and 4 invalid model responses, then marked 74 further eligible tickets `budget_exhausted`; the response that crossed the threshold brought measured usage to 6,021 tokens. Empty and long tickets remain independently skipped/flagged. Budget checks prevent subsequent calls once accumulated usage is at least the budget; a single in-flight response can take the measured total over it.

The final `outputs/triage_results.csv` is the few-shot run with budget 100,000 (164 rows). Outputs accepted as `classified_json` or `classified_keyword` have passed category validation. `flagged_invalid_output`, `skipped_empty`, `flagged_long` and `budget_exhausted` route to human review with NORMAL priority; valid fraud tickets route to the security team with HIGH priority; valid Premium-customer tickets route to the premium desk with HIGH priority; other valid tickets route to support with NORMAL priority. Missing segment rows are treated as not Premium.

| Final route | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |

## Part D — Policy RAG Assistant and Refund Agent

### Policy retrieval and evaluation

The five policy documents are split into 40-word chunks with 10-word overlap, producing 15 chunks: refund 4, delivery 3, escalation 3, payment 3 and privacy 2. `TfidfVectorizer()` measures term frequency within a chunk and inverse document frequency to give more weight to terms that are rarer across chunks. TF-IDF is lexical, so semantically equivalent phrasings with different vocabulary can retrieve poorly. For `refund policy late delivery`, the top three retrieved chunks all come from `refund_policy.txt`.

The assistant sends a system instruction containing `POLICY ASSISTANT` and a user message in the required `CONTEXT:` / `QUESTION:` format. It returns the mock model answer and the sources used.

| Chunking | Expected source among top 3 | Expected fact substring (case-sensitive) | Unanswerable fallback |
|---|---:|---:|---:|
| 40 words, 10-word overlap | 12/12 | 12/12 | 2/2 |
| 40 words, no overlap | 12/12 | 9/12 | 2/2 |

The overlapping-chunk evaluation is saved to `outputs/policy_eval_results.csv`. With zero overlap, “What delivery time makes a completed order eligible for a full refund?” failed its expected-fact match. The full refund-eligibility sentence is split at the 40-word boundary after “the customer”; the consequence begins in the next chunk, and top-three retrieval can omit that adjacent continuation. The zero-overlap evaluation is saved separately as `outputs/policy_eval_results_no_overlap.csv`.

### Refund tools and ReAct runs

The five tools are `get_order(order_id)`, `search_policy(query)`, `calculate_refund(order_amount, delivery_minutes)`, `issue_refund(order_id, refund_inr)` and `escalate_to_human(order_id, refund_inr)`. Their JSON-style parameter specifications are in `TOOLS_SPEC`, and `TOOL_MAP` maps each tool name to its implementation. The order lookup reads SQLite; policy search reuses the retriever; completed deliveries over 45 minutes get a full refund, deliveries at or below 45 minutes get zero, and amounts above INR 1,000 are escalated instead of automatically issued. Simulated requests are treated as raised inside the seven-day window. Each scenario is recorded in `outputs/agent_runs.jsonl`; issued and escalated actions are recorded in `outputs/refund_ledger.jsonl`.

| Scenario | Status | Refund (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1,219 |
| S2 | REFUND | 135.00 | 5 | 1,219 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,206 |
| S4 | DENIED | 0.00 | 4 | 853 |
| S5 | DENIED | 0.00 | 3 | 517 |
| S6 | NOT_FOUND | 0.00 | 2 | 192 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,360 |

`run_agent` enforces a six-iteration default, a 6,000-token limit and a 10-second wall-clock limit; each fallback returns completed steps and a human-handoff summary. S7 completed six model calls and ended with `FALLBACK_MAX_ITERATIONS`. The S1 demo with `token_limit=200` ended `FALLBACK_TOKEN_LIMIT` after 2 calls and 222 measured tokens (the limit is checked after a response and before executing its requested tool). The S1 demo with `time_limit_seconds=0` ended `FALLBACK_TIMEOUT` before any model call.

After each tool observation, the agent saves the conversation, run and scenario IDs, token count, step number and tools already executed to `state/`. `resume_agent(run_id)` continues from that checkpoint. The S1 crash-and-resume demonstration ended with REFUND INR 128.25; the tools executed exactly once, in order, were `get_order`, `search_policy`, `calculate_refund`, `issue_refund`.

In this workflow, ReAct used 5 calls / 1,219 tokens for S1, 3 / 517 for S5 and 2 / 192 for S6. A plan-and-execute agent could ask for a complete step list once and use fewer calls on successful paths, but its token total depends on the plan payload and was not measured here; it also needs validation and replanning when a tool returns an unexpected result. ReAct sees `NOT_FOUND` immediately and can stop without attempting a refund; similarly, a cancelled order is denied after its order lookup rather than blindly executing a planned payment action. I prefer ReAct here because each consequential action is gated by fresh order and policy observations, despite the added calls.

## Limitations and Next Steps

- The raw data and policy text are synthetic; the offline mock does not represent live model quality or latency.
- The lapse cohort and held-out test are modest in size; evaluate on additional time-based cohorts before campaign deployment.
- The Premium flag is calculated from whole-window RFM and includes the September label period, so **the retention list is illustrative and not a leakage-free evaluation** even though model features use only pre-cutoff orders.
- Ticket performance is measured on synthetic, templated examples; test privacy and triage with independent human-labelled data before production use.
- Policy retrieval is lexical TF-IDF; richer semantic retrieval and stricter grounded-answer checks could improve paraphrase handling.
- Refund actions are simulated ledger writes; a production system needs authorization, idempotency, audit controls and human approval integrations.
