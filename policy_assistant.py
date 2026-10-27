
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
TOP_K = 3

FALLBACK = (
    "I could not find that information in the policy documents."
)


# ==================================================
# LOAD POLICY DOCUMENTS
# ==================================================

def load_policy_documents():
    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy folder not found: {POLICY_DIR}"
        )

    documents = {}

    for path in sorted(POLICY_DIR.glob("*.txt")):
        documents[path.name] = path.read_text(encoding="utf-8")

    if not documents:
        raise FileNotFoundError(
            f"No policy .txt files found in {POLICY_DIR}"
        )

    print("Policy documents loaded:", len(documents))

    for filename in documents:
        print(" -", filename)

    return documents


# ==================================================
# CHUNKING
# ==================================================

def chunk_document(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunks.append({
            "start_word": start,
            "end_word": end,
            "text": " ".join(words[start:end]),
        })

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(documents, overlap=CHUNK_OVERLAP):
    chunks = []

    for filename, text in documents.items():
        for chunk in chunk_document(
            text,
            chunk_size=CHUNK_SIZE,
            overlap=overlap,
        ):
            chunks.append({
                "source": filename,
                "start_word": chunk["start_word"],
                "end_word": chunk["end_word"],
                "text": chunk["text"],
            })

    return chunks


# ==================================================
# TF-IDF RETRIEVER
# ==================================================

class PolicyRetriever:
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("No policy chunks available")

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
        )

        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )

    def retrieve(self, query, k=TOP_K):
        if not query or not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        )[0]

        ranked_indices = scores.argsort()[::-1]
        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            chunk = self.chunks[index]

            results.append({
                "source": chunk["source"],
                "score": round(score, 4),
                "text": chunk["text"],
                "start_word": chunk["start_word"],
                "end_word": chunk["end_word"],
            })

            if len(results) >= k:
                break

        return results


# ==================================================
# POLICY ANSWERING
# ==================================================

def answer_question(question, retriever):
    retrieved = retriever.retrieve(question, k=TOP_K)

    if not retrieved:
        return {
            "answer": FALLBACK,
            "sources": [],
            "retrieved": [],
            "tokens_used": 0,
        }

    context_parts = []

    for item in retrieved:
        context_parts.append(
            f"[{item['source']}] {item['text']}"
        )

    context = "\n".join(context_parts)

    system_prompt = """
POLICY ASSISTANT

Answer the question using only the supplied policy context.
Do not invent facts or policy rules.
If the answer cannot be found in the context, reply exactly:
I could not find that information in the policy documents.
Return a concise answer containing the relevant policy fact.
"""

    user_prompt = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = mock_llm.chat(messages)
    answer = (response.choices[0].message.content or "").strip()

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(
            item["source"] for item in retrieved
        )),
        "retrieved": retrieved,
        "tokens_used": response.usage.total_tokens,
    }


# ==================================================
# DEMO
# ==================================================

def show_retrieval_demo(retriever):
    query = "refund policy late delivery"

    print("\n--- RETRIEVAL DEMO ---")
    print("Query:", query)

    for rank, item in enumerate(
        retriever.retrieve(query, k=3), start=1
    ):
        print(f"\nRank {rank}")
        print("Source:", item["source"])
        print("Score:", item["score"])
        print("Text:", item["text"])


# ==================================================
# EVALUATION
# ==================================================

def evaluate_policy_assistant(
    retriever,
    output_path=OUTPUT_DIR / "policy_eval_results.csv",
    label="POLICY EVALUATION",
):
    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {EVAL_FILE}"
        )

    evaluation = pd.read_csv(
        EVAL_FILE,
        keep_default_na=False,
    )

    print("\nEvaluation file:", EVAL_FILE)
    print("Evaluation columns:", evaluation.columns.tolist())

    required = {"question", "expected_doc", "expected_fact"}
    missing = required - set(evaluation.columns)

    if missing:
        raise ValueError(
            f"Missing evaluation columns: {sorted(missing)}"
        )

    results = []

    for _, row in evaluation.iterrows():
        question = str(row["question"]).strip()
        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = str(row["expected_fact"]).strip()

        unanswerable = expected_doc.upper() == "NONE"

        response = answer_question(question, retriever)

        answer = response["answer"]
        sources = response["sources"]

        if unanswerable:
            source_match = None
            fact_match = None
        else:
            source_match = any(
                source.lower() == expected_doc.lower()
                for source in sources
            )

            # The brief defines expected facts as case-sensitive substrings.
            fact_match = (
                expected_fact in answer
                if expected_fact
                else None
            )

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "answer": answer,
            "retrieved_sources": ";".join(sources),
            "expected_source_found": source_match,
            "expected_fact_found": fact_match,
            "fallback_used": answer.startswith("I could not find"),
            "tokens_used": response["tokens_used"],
        })

    results_df = pd.DataFrame(results)

    print(f"\n--- {label} ---")
    print("Total evaluation questions:", len(results_df))

    answerable = results_df[
        results_df["expected_doc"].str.upper() != "NONE"
    ]

    if len(answerable):
        source_count = int(
            answerable["expected_source_found"].fillna(False).sum()
        )
        fact_count = int(
            answerable["expected_fact_found"].fillna(False).sum()
        )

        print(
            f"Expected source found: {source_count}/{len(answerable)} "
            f"({source_count / len(answerable):.1%})"
        )
        print(
            f"Expected fact matched: {fact_count}/{len(answerable)} "
            f"({fact_count / len(answerable):.1%})"
        )

    unanswerable_rows = results_df[
        results_df["expected_doc"].str.upper() == "NONE"
    ]

    if len(unanswerable_rows):
        fallback_count = int(
            unanswerable_rows["fallback_used"].sum()
        )

        print(
            "Unanswerable questions using fallback:",
            f"{fallback_count}/{len(unanswerable_rows)}",
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("Saved evaluation:", output_path)

    return results_df


# ==================================================
# CHUNK OVERLAP COMPARISON
# ==================================================

def compare_overlap(documents):
    print("\n--- CHUNK OVERLAP COMPARISON ---")

    chunks_overlap = build_chunks(
        documents,
        overlap=CHUNK_OVERLAP,
    )

    chunks_no_overlap = build_chunks(
        documents,
        overlap=0,
    )

    print("Chunks with overlap 10:", len(chunks_overlap))
    print("Chunks with overlap 0:", len(chunks_no_overlap))

    retriever_no_overlap = PolicyRetriever(chunks_no_overlap)
    results = evaluate_policy_assistant(
        retriever_no_overlap,
        output_path=OUTPUT_DIR / "policy_eval_results_no_overlap.csv",
        label="ZERO-OVERLAP POLICY EVALUATION",
    )
    broken = results.loc[
        (results["expected_doc"] != "NONE")
        & ~results["expected_fact_found"].fillna(False)
    ]
    if not broken.empty:
        row = broken.iloc[0]
        print("First answerable question that failed expected-fact check:")
        print("Question:", row["question"])
        print("Expected fact:", row["expected_fact"])
        print("Retrieved sources:", row["retrieved_sources"])
    else:
        print("No expected-fact failures occurred with zero-overlap chunks.")
    return results


# ==================================================
# MAIN
# ==================================================

def main():
    documents = load_policy_documents()

    chunks = build_chunks(
        documents,
        overlap=CHUNK_OVERLAP,
    )

    print("\nTotal chunks:", len(chunks))

    for filename in documents:
        count = sum(
            chunk["source"] == filename
            for chunk in chunks
        )
        print(f"{filename}: {count} chunks")

    retriever = PolicyRetriever(chunks)

    show_retrieval_demo(retriever)

    demo_question = "What is the refund policy for a late delivery?"

    print("\n--- ANSWER DEMO ---")
    print("Question:", demo_question)

    demo = answer_question(demo_question, retriever)

    print("Answer:", demo["answer"])
    print("Sources:", demo["sources"])
    print("Tokens used:", demo["tokens_used"])

    evaluate_policy_assistant(retriever)
    compare_overlap(documents)

    print("\nPart D Step 1 completed.")


if __name__ == "__main__":
    main()
