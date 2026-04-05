# Financial-and-Banking-Domain-RAG-System-with-Citation-Enforcement

> A **production-grade Retrieval-Augmented Generation (RAG)** system designed for **accurate, verifiable, and finance-specific question answering**, with **hybrid retrieval, re-ranking, and automated faithfulness evaluation**.

---

## 🚀 Overview

This project builds a **finance and banking domain RAG pipeline** that retrieves relevant information from a curated corpus of regulatory filings, central bank policy documents, and earnings reports — and generates **factually grounded answers with citations**.

Unlike naive LLM systems, this architecture ensures:

* **No hallucinations** — critical in financial contexts
* **Traceable answers** — every claim backed by a source
* **Production-ready evaluation + CI gating**

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:

* **Retriever** → Finds relevant financial documents
* **Generator** → Produces answer using retrieved context

This solves the core LLM problem in finance:

> ❌ Hallucinated figures / policies → ✅ Grounded, evidence-backed responses

---

## 🏦 Domain Corpus

This system is built on a curated corpus of real financial and regulatory documents:

### 📁 SEC EDGAR — 10-K Annual Filings
| Bank | Files |
|---|---|
| Bank of America (BAC) | `bac-20231231.pdf` |
| JPMorgan Chase (JPM) | `FORM 10-K: J.P. MORGAN CHASE & CO..pdf` |
| Goldman Sachs (GS) | `2025-10-k.pdf` |

### 📁 Reserve Bank of India (RBI)
| Document | Type |
|---|---|
| MPC Statement | PDF |
| Master Circular | PDF |

### 📁 Federal Reserve (FED)
| Document | Type |
|---|---|
| FOMC Policy Statement (Jan 2026) | PDF |

---

## 🗂️ Data Folder Structure

```
Data/
├── FED/
│   └── monetary20260128a1.pdf
├── RBI/
│   ├── PR6DFA5AD53D2D0414FAAB8D898975C40AA.PDF
│   └── PR19BD28196D176C4964A1C1E727002EF7AA.PDF
└── sec-edgar-filings/
    ├── BAC/
    │   └── bac-20231231.pdf
    ├── GS/
    │   └── 2025-10-k.pdf
    └── JPM/
        └── FORM 10-K: J.P. MORGAN CHASE & CO..pdf
```

---

## 🏗️ System Architecture

```
User Query (Finance / Banking)
    ↓
Hybrid Retriever (BM25 + Vector Search)
    ↓
Top-K Chunks (from SEC / RBI / FED corpus)
    ↓
Cross-Encoder Re-Ranker (Cohere)
    ↓
Filtered Context
    ↓
LLM Generator (Citation Enforced)
    ↓
Answer + Source Citations (e.g. "JPM 10-K 2023, p.45")
```

---

## 📦 Tech Stack

* **Orchestration**: LangChain
* **LLM**: Groq (`llama-3.3-70b-versatile`)
* **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector DB**: Chroma
* **Re-ranking**: Cohere (`rerank-english-v3.0`)
* **Evaluation**: RAGAS
* **Search**: BM25 (`rank-bm25`)
* **Chunking**: tiktoken (token-aware)
* **Frontend**: Streamlit
* **CI/CD**: GitHub Actions

---

## ⚙️ Phase 1 — Data Ingestion & Basic RAG

### 📄 Document Processing

* Load corpus from `data/` (`.txt` and `.pdf` supported)
* Chunk documents:
  * Size: **500–800 tokens**
  * Overlap: **~100 tokens**

👉 Why overlap matters in finance:

* Prevents **splitting financial statements** mid-context
* Preserves continuity across multi-paragraph regulatory clauses

---

### 🔍 Embedding & Storage

* Convert chunks → vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`
* Each chunk is tagged with paragraph-level citation metadata:
  * `source` → filename
  * `category` → SEC / RBI / FED
  * `institution` → BAC / JPM / GS / RBI / FED
  * `page` → page number (PDF)
  * `paragraph_index` → paragraph within page
  * `citation_label` → e.g. `bac-20231231.pdf, page 4, paragraph 2`
* Store in Chroma vector DB (`chroma_db/`)

---

### 🔎 Retrieval Pipeline

1. Embed user query
2. Retrieve **Top-K relevant chunks**
3. Pass to LLM with source metadata
4. Generate answer **with citations**

---

## ⚙️ Phase 2 — Production-Grade RAG

### 🔀 Hybrid Retrieval

Combine:

* **BM25 (keyword search)** → exact match for financial terms (e.g. "Tier 1 Capital", "Repo Rate", "EBITDA")
* **Vector search (semantic)** → meaning-based retrieval ("what is the bank's risk exposure?")

👉 Result: **Best of both worlds for financial text**

---

### 🧠 Cross-Encoder Re-Ranking

Using Cohere:

* Input: *(query, chunk)* pairs
* Output: relevance score

✔ Improves ranking accuracy on dense regulatory documents
✔ Filters noisy retrievals from long 10-K filings

---

### 📌 Citation Enforcement (Anti-Hallucination Layer)

* Model must **only answer using retrieved chunks**
* Every answer includes source attribution:
  > *"According to JPMorgan 10-K 2023..."*
  > *"Per RBI Master Circular on KYC..."*
  > *"As per FOMC Statement, March 2023..."*

* If insufficient evidence:
  > ❗ System declines instead of hallucinating financial figures

---

## ⚙️ Phase 3 — Evaluation & CI Integration

### 📊 Golden Dataset

* **10 Finance Q&A pairs** stored in `evaluation/golden_dataset.json`
* Examples:
  * *"What did the Fed decide on interest rates in January 2026?"*
  * *"What is the interest rate paid on reserve balances?"*
  * *"What are JPMorgan's total assets?"*
  * *"What is Goldman Sachs's net revenue?"*
  * *"What are the KYC norms mandated by RBI?"*

---

### 🧪 Faithfulness Evaluation

Using RAGAS:

Evaluate:

* **Faithfulness** → Is the financial answer supported by the retrieved document?
* **Answer correctness** → Does it match verified ground truth?
* **Context relevance** → Was the right regulatory/financial chunk retrieved?

---

### 🧠 Faithfulness Logic

For each generated answer:

1. Extract claims (e.g. revenue figures, policy rates, compliance rules)
2. Check if claims are supported by retrieved chunks
3. Penalize any unsupported financial outputs

---

### 🔁 CI/CD Integration

* Integrated with GitHub Actions
* On every PR:
  * Run evaluation pipeline against golden dataset
  * Compare faithfulness metrics vs threshold

✅ If quality drops → **Build fails**
❌ Prevents regression — no hallucinated financial data ships to production

---

## 📁 Project Structure

```
RAG-Domain/
│
├── Data/                        # Raw corpus (SEC, RBI, FED)
│   ├── FED/
│   ├── RBI/
│   └── sec-edgar-filings/
│       ├── BAC/
│       ├── GS/
│       └── JPM/
├── chroma_db/                   # Persisted Chroma vector DB
├── evaluation/
│   ├── golden_dataset.json      # 10 finance Q&A pairs
│   └── ragas_eval.py            # RAGAS faithfulness evaluation
├── .github/
│   └── workflows/
│       └── eval.yml             # GitHub Actions CI pipeline
├── datachunking.py              # Token-aware chunking + paragraph citation
├── dataimport.py                # SEC filing HTML cleaner + loader
├── embeddings.py                # Embedding + Chroma vectorstore builder
├── hybrid_retriever.py          # BM25 + vector search + Cohere rerank
├── qa_chain.py                  # Citation-enforced LLM QA chain
└── app.py                       # Streamlit frontend
```

---

## 🔬 Key Design Decisions

### 1. Why Hybrid Retrieval?

* BM25 → precise match for financial terms ("Net Interest Income", "CET1 Ratio")
* Vector → semantic retrieval for policy intent and regulatory meaning
* Combined → **robust retrieval across both structured filings and narrative policy docs**

---

### 2. Why Cross-Encoder Re-Ranking?

* 10-K filings are long and noisy — bi-encoders retrieve fast but coarse
* Cross-encoder provides much **more accurate ranking** for dense financial text

---

### 3. Why Citation Enforcement?

* Financial misinformation has real consequences
* Every answer must be traceable to a specific document and institution
* Ensures **regulatory and compliance trustworthiness**

---

### 4. Why RAGAS?

* Standardized evaluation for RAG systems
* Measures **faithfulness, not just accuracy** — critical for finance domain

---

### 5. Why Cross-Jurisdictional Corpus (US + India)?

* Enables cross-corpus queries: *"Compare Fed vs RBI inflation policy"*
* Demonstrates retrieval precision across different regulatory languages and formats
* Strong portfolio differentiator

---

## 📈 Future Improvements

* Expand golden dataset to 50–100 Q&A pairs
* Add multi-year 10-K filings for BAC, JPM, GS (2021–2024)
* Query rewriting for better retrieval on complex financial questions
* Multi-hop reasoning (e.g. linking RBI policy → bank capital impact)
* Caching frequent regulatory queries
* Fine-tuned domain embeddings on financial text (e.g. FinBERT)
* Feedback loop for continuous corpus updates (new filings, new MPC statements)

---

## 🎯 Key Takeaways

* RAG ≠ just retrieval + LLM
* In finance, real systems require:
  * Hybrid search for both exact financial terms and semantic meaning
  * Re-ranking for dense regulatory and filing documents
  * Strict citation grounding — no hallucinated figures
  * Continuous faithfulness evaluation

---

## 🤝 Contributing

Pull requests are evaluated automatically via CI.
Ensure your changes maintain **faithfulness thresholds** on the financial golden dataset.

---

## 📜 License

MIT License

---