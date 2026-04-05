import os
import json
import sys
import time
import math
from pathlib import Path

# ── Allow imports from project root ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qa_chain import format_context, PROMPT_TEMPLATE
from hybrid_retriever import load_vectorstore, build_bm25_index, build_hybrid_retriever

GOLDEN_PATH            = Path(__file__).parent / "golden_dataset.json"
FAITHFULNESS_THRESHOLD = 0.7
COHERE_RATE_LIMIT_SECS = 7


def _require_env(name: str) -> str:
    """Fail fast with a clear message if a required env var is missing."""
    value = os.environ.get(name)
    if not value:
        print(f"❌ Missing required environment variable: {name}")
        sys.exit(1)
    return value


def run_evaluation():
    # ── Validate env vars up front (CI-safe: fail fast before any work) ───────
    groq_api_key = _require_env("GROQ_API_KEY")

    print("🔄 Loading vectorstore and BM25 index...")
    vectorstore = load_vectorstore()
    bm25_index  = build_bm25_index()

    print("📂 Loading golden dataset...")
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    questions     = []
    ground_truths = []
    answers       = []
    contexts      = []

    # ── Generation LLM ───────────────────────────────────────────────────────
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=groq_api_key,
    )

    # ── RAGAS judge LLM + embeddings ─────────────────────────────────────────
    # Use ragas.metrics (not ragas.metrics.collections) — these accept
    # LangchainLLMWrapper and LangchainEmbeddingsWrapper without issues.
    ragas_llm = LangchainLLMWrapper(groq_llm)
    ragas_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # ── Build metrics ─────────────────────────────────────────────────────────
    # strictness=1 on AnswerRelevancy is critical for Groq:
    # the default strictness=3 makes RAGAS send n=3 to the LLM in one call,
    # which Groq rejects with: "'n': number must be at most 1".
    # This was the root cause of all the BadRequestError exceptions and NaN scores.
    faithfulness_metric      = Faithfulness(llm=ragas_llm)
    answer_relevancy_metric  = AnswerRelevancy(
        llm=ragas_llm,
        embeddings=ragas_emb,
        strictness=1,           # ← forces n=1 per LLM call; required for Groq
    )
    context_precision_metric = ContextPrecision(llm=ragas_llm)

    print(f"🧪 Running {len(golden)} test cases...\n")

    for i, item in enumerate(golden):
        question    = item["question"]
        institution = item["institution"]
        gt          = item["ground_truth"]

        print(f"  [{i+1}/{len(golden)}] {question[:60]}...")

        if i > 0:
            print(f"   ⏳ Waiting {COHERE_RATE_LIMIT_SECS}s for Cohere rate limit...")
            time.sleep(COHERE_RATE_LIMIT_SECS)

        # ── Retrieve ──────────────────────────────────────────────────────────
        retriever = build_hybrid_retriever(
            vectorstore=vectorstore,
            bm25_index=bm25_index,
            institution_filter=institution,
            k=6,
        )
        docs          = retriever.invoke(question)
        context_texts = [doc.page_content for doc in docs]

        # ── Generate answer ───────────────────────────────────────────────────
        formatted   = format_context(docs)
        prompt_text = PROMPT_TEMPLATE.format(context=formatted, question=question)
        answer      = groq_llm.invoke(prompt_text).content

        questions.append(question)
        ground_truths.append(gt)
        answers.append(answer)
        contexts.append(context_texts)

    # ── Build RAGAS dataset ───────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts,
        "ground_truth": ground_truths,
    })

    print("\n📊 Running RAGAS evaluation...")
    print("⏳ Waiting 30s before RAGAS scoring to avoid rate limits...")
    time.sleep(30)

    # max_workers=1 prevents concurrent Groq calls that trigger rate limits.
    # timeout=300 + max_wait=120 gives enough headroom for slow Groq responses.
    run_config = RunConfig(timeout=300, max_retries=3, max_wait=120, max_workers=1)

    results = evaluate(
        dataset,
        metrics=[
            faithfulness_metric,
            answer_relevancy_metric,
            context_precision_metric,
        ],
        run_config=run_config,
        batch_size=1,
    )

    print("\n" + "=" * 50)
    print("📈 RAGAS Evaluation Results")
    print("=" * 50)
    df = results.to_pandas()
    print(df.to_string())

    mean_faithfulness  = df["faithfulness"].mean()
    valid_faithfulness = df["faithfulness"].dropna()

    # Guard: NaN silently passed the old threshold check (nan < 0.7 is False
    # in Python, so the PASSED branch was always taken). Catch it explicitly.
    if valid_faithfulness.empty or not math.isfinite(mean_faithfulness):
        print(
            "\n❌ FAILED — faithfulness score is NaN or missing. "
            "Check for BadRequestError / TimeoutError in the output above."
        )
        sys.exit(1)

    print(f"\n✅ Mean Faithfulness : {mean_faithfulness:.3f}")
    print(f"   Valid rows        : {len(valid_faithfulness)}/{len(df)}")
    print(f"   Threshold         : {FAITHFULNESS_THRESHOLD}")

    if mean_faithfulness < FAITHFULNESS_THRESHOLD:
        print(
            f"\n❌ FAILED — faithfulness {mean_faithfulness:.3f} "
            f"below threshold {FAITHFULNESS_THRESHOLD}"
        )
        sys.exit(1)
    else:
        print(f"\n✅ PASSED — faithfulness {mean_faithfulness:.3f} meets threshold")
        sys.exit(0)


if __name__ == "__main__":
    run_evaluation()