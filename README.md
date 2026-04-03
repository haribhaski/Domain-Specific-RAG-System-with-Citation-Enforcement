# Domain-Specific-RAG-System-with-Citation-Enforcement

> A **production-grade Retrieval-Augmented Generation (RAG)** system designed for **accurate, verifiable, and domain-specific question answering**, with **hybrid retrieval, re-ranking, and automated faithfulness evaluation**.

---

## 🚀 Overview

This project builds a **domain-specific RAG pipeline** that retrieves relevant information from a curated corpus and generates **factually grounded answers with citations**.

Unlike naive LLM systems, this architecture ensures:

* **No hallucinations**
* **Traceable answers**
* **Production-ready evaluation + CI gating**

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:

* **Retriever** → Finds relevant documents
* **Generator** → Produces answer using retrieved context

This solves the core LLM problem:

> ❌ Hallucination → ✅ Grounded, evidence-backed responses

---

## 🏗️ System Architecture

```
User Query
    ↓
Hybrid Retriever (BM25 + Vector Search)
    ↓
Top-K Chunks
    ↓
Cross-Encoder Re-Ranker (Cohere)
    ↓
Filtered Context
    ↓
LLM Generator (Citation Enforced)
    ↓
Answer + Citations
```

---

## 📦 Tech Stack

* **Orchestration**: LangChain / LangGraph
* **Vector DB**: Chroma / Weaviate
* **Re-ranking**: Cohere (Cross-Encoder)
* **Evaluation**: RAGAS
* **Search**: BM25 (sparse retrieval)
* **CI/CD**: GitHub Actions

---

## ⚙️ Phase 1 — Data Ingestion & Basic RAG

### 📄 Document Processing

* Load domain-specific corpus
* Chunk documents:

  * Size: **500–800 tokens**
  * Overlap: **~100 tokens**

👉 Why overlap matters:

* Prevents **context fragmentation**
* Preserves semantic continuity across chunks

---

### 🔍 Embedding & Storage

* Convert chunks → vector embeddings
* Store in vector DB:

  * Chroma / Weaviate

---

### 🔎 Retrieval Pipeline

1. Embed user query
2. Retrieve **Top-K relevant chunks**
3. Pass to LLM
4. Generate answer **with citations**

---

## ⚙️ Phase 2 — Production-Grade RAG

### 🔀 Hybrid Retrieval

Combine:

* **BM25 (keyword search)** → exact match strength
* **Vector search (semantic)** → meaning understanding

👉 Result: **Best of both worlds**

---

### 🧠 Cross-Encoder Re-Ranking

Using Cohere:

* Input: *(query, chunk)* pairs
* Output: relevance score

✔ Improves ranking accuracy
✔ Filters noisy retrievals

---

### 📌 Citation Enforcement (Anti-Hallucination Layer)

* Model must **only answer using retrieved chunks**
* If insufficient evidence:

  > ❗ System declines instead of hallucinating

---

### 🧾 Prompt Versioning

* Store prompts in **config/versioned files**
* Enables:

  * Reproducibility
  * A/B testing
  * System-level control

---

## ⚙️ Phase 3 — Evaluation & CI Integration

### 📊 Golden Dataset

* Curate **50–100 Q&A pairs**
* Manually verified for correctness

---

### 🧪 Faithfulness Evaluation

Using RAGAS:

Evaluate:

* **Faithfulness** → Is answer supported by context?
* **Answer correctness**
* **Context relevance**

---

### 🧠 Faithfulness Logic

For each generated answer:

1. Extract claims
2. Check if claims are supported by retrieved chunks
3. Penalize unsupported outputs

---

### 🔁 CI/CD Integration

* Integrated with GitHub Actions
* On every PR:

  * Run evaluation pipeline
  * Compare metrics vs threshold

✅ If quality drops → **Build fails**
❌ Prevents regression in production

---

## 📁 Project Structure

```
rag-system/
│
├── data/                  # Raw corpus
├── embeddings/            # Stored vector DB
├── ingestion/             # Chunking + embedding pipeline
├── retrieval/             # Hybrid retriever
├── reranker/              # Cohere reranking logic
├── generation/            # LLM + prompt templates
├── evaluation/            # RAGAS scripts
├── config/                # Prompt/version configs
├── ci/                    # CI pipeline configs
└── app.py                 # Main entry point
```

---

## 🔬 Key Design Decisions

### 1. Why Hybrid Retrieval?

* BM25 → precise keyword match
* Vector → semantic understanding
  👉 Combined → **robust retrieval**

---

### 2. Why Cross-Encoder?

* Bi-encoders retrieve fast but coarse
* Cross-encoder:

  * Slower
  * Much **more accurate ranking**

---

### 3. Why Citation Enforcement?

* Prevents **hallucinated answers**
* Ensures **trustworthiness**

---

### 4. Why RAGAS?

* Standardized evaluation for RAG systems
* Measures **faithfulness, not just accuracy**

---

## 📈 Future Improvements

* Query rewriting for better retrieval
* Multi-hop reasoning RAG
* Caching frequent queries
* Fine-tuned domain embeddings
* Feedback loop for continuous learning

---

## 🎯 Key Takeaways

* RAG ≠ just retrieval + LLM
* Real systems require:

  * Hybrid search
  * Re-ranking
  * Strict grounding
  * Continuous evaluation

---

## 🤝 Contributing

Pull requests are evaluated automatically via CI.
Ensure your changes maintain **faithfulness thresholds**.

---

## 📜 License

MIT License

---
