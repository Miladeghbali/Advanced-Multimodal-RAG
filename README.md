# Professional Advanced Multimodal RAG

A portfolio/university-grade Retrieval-Augmented Generation system with a production-minded architecture.

The required retrieval sequence is implemented explicitly:

**Dense Vector Search + BM25 → weighted RRF Hybrid Search → MMR → Cross-Encoder Re-Ranker → bounded grounded context → LLM**

The project also includes conversational query rewriting, persistent lexical+dense state, source citations, multimodal image ingestion, evaluation, tracing, security checks, tests, and Docker packaging.

## Feature matrix

| Area | Capability | Status |
|---|---|---|
| Ingestion | PDF, TXT, Markdown, CSV, HTML, DOCX | ✅ |
| Chunking | Recursive chunking + overlap + stable chunk IDs | ✅ |
| Dense retrieval | HuggingFace multilingual embeddings + Chroma | ✅ |
| Lexical retrieval | BM25 with Persian/Arabic character normalization | ✅ |
| Hybrid | Weighted Reciprocal Rank Fusion (RRF) | ✅ |
| Diversity | Maximum Marginal Relevance (MMR) | ✅ |
| Re-ranking | Cross-Encoder second-stage re-ranker | ✅ |
| Conversation | Bounded memory + follow-up query rewriting | ✅ |
| Grounding | Bounded context + `[S1]` source labels + citation audit | ✅ |
| Persistence | Chroma + persisted BM25 corpus + index compatibility signature | ✅ |
| KB lifecycle | Add/update/replace/delete sources with stable deduplication | ✅ |
| Multimodal | KB image → vision caption → searchable RAG evidence | ✅ |
| Query image | Image + retrieved text can be answered together | ✅ |
| Evaluation | Precision@k, Recall@k, MRR, nDCG | ✅ |
| LangSmith | Tracing + dataset experiments + optional LLM judges | ✅ |
| Observability | Per-stage ranks/scores, latency timings, retrieval diagnostics | ✅ |
| Security | Upload limits, untrusted-context prompt, basic injection detection | ✅ |
| QA | Unit tests + GitHub Actions | ✅ |
| Deployment | Streamlit + Dockerfile + healthcheck | ✅ |
| UI | Persian / English / bilingual live language switcher + contextual `?` help | ✅ |

## Architecture

```text
Documents / Images
        │
        ├── PDF/TXT/MD/CSV/HTML/DOCX loader
        └── Vision captioning for KB images
        │
        ▼
Recursive chunking + stable chunk IDs
        │
        ├──────────────────────┐
        ▼                      ▼
Multilingual embeddings      BM25 index
        │                      │
        ▼                      ▼
Chroma dense retrieval     Lexical retrieval
        └──────────┬───────────┘
                   ▼
          Weighted RRF Hybrid
                   │
                   ▼
                  MMR
                   │
                   ▼
       Cross-Encoder Re-Ranker
                   │
                   ▼
       Bounded cited context
                   │
                   ▼
      OpenRouter vision/chat LLM
                   │
                   ▼
        Answer + [S1] Sources
```

For follow-up questions, recent conversation history is first used to rewrite the latest question into a standalone retrieval query. The original user question is still used for final answer generation.

## Why the persistence layer matters

A professional hybrid retriever must persist **both** sides of retrieval. Chroma persists dense vectors, while `rag_state/corpus.jsonl` persists the exact chunk corpus used to rebuild BM25 after an application restart.

`rag_state/index_meta.json` stores an index signature based on the embedding model and chunking configuration. If those settings change, the UI asks for re-indexing rather than silently mixing incompatible vectors and chunks.

## Multimodal behavior

There are two image paths:

1. **Knowledge-base image ingestion**: a vision-capable OpenRouter model captions the image; the caption is stored as a RAG document and embedded so later text queries can retrieve it. The original local image is retained under `rag_state/assets/` for source inspection.
2. **Query image**: an image can be attached to the current question and sent to the vision-capable LLM together with text retrieved from the knowledge base.

This is multimodal RAG through **vision-to-text indexing**, not a CLIP-style joint image/text vector space.

## Evaluation

### Local retrieval evaluation

```bash
python -m evaluation.run_local_eval
```

It evaluates the retrieval stages without modifying the app knowledge base and reports:

- Precision@k
- Recall@k
- MRR
- nDCG@k

### LangSmith evaluation

Configure:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=Advanced-Multimodal-RAG
```

Then:

```bash
python -m evaluation.run_langsmith_eval
```

The experiment uses a golden dataset and deterministic evaluators for citation and retrieval/answer keyword matching.

To additionally run LLM-as-judge checks for answer correctness, answer relevance, groundedness, and retrieval relevance:

```env
LANGSMITH_LLM_JUDGE=true
```

## Security controls

The application treats retrieved documents as **untrusted evidence**, not instructions. The generation system prompt explicitly tells the model not to follow instructions embedded inside retrieved content.

Additional controls include:

- file extension allow-list
- per-file size limit
- batch file-count limit
- filename sanitization for persisted image assets
- detection/visibility for common direct and indirect prompt-injection phrases
- `.env` excluded from Git
- optional OpenRouter provider setting `data_collection=deny`

These are defense-in-depth measures, not a guarantee against prompt injection.

## Models

Default embedding model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Default fast Cross-Encoder:

```text
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

The embedding model explicitly lists `fa`/Persian among its supported languages. The default fast mMARCO re-ranker is multilingual, but Persian is not explicitly part of its published training-language list, so Persian reranking quality should be measured on the included golden/evaluation workflow rather than assumed.

For a stronger but much heavier multilingual re-ranker, you can experiment with:

```text
BAAI/bge-reranker-v2-m3
```

Model choice should be validated on your own golden dataset rather than assumed to be universally best.

## Installation

Python **3.11** is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Create `.env`:

```powershell
copy .env.example .env
```

or:

```bash
cp .env.example .env
```

Set:

```env
OPENROUTER_API_KEY=...
```

Operational LLM controls are configurable in `.env`, including `LLM_TIMEOUT_MS` (milliseconds in the current `ChatOpenRouter` client), `LLM_MAX_RETRIES`, and `LLM_MAX_TOKENS`.

Run:

```bash
streamlit run app.py
```

## Demo

Upload:

```text
data/documents/sample_knowledge.txt
```

Then ask:

- What is hybrid search?
- Why is MMR useful?
- What does the re-ranker do?
- What is the difference between BM25 and vector search?
- What is LangSmith useful for in RAG?

Open **Retrieval / safety diagnostics** to show the executed sequence and the rewritten retrieval query.

## Tests

Install development dependencies and run the suite:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

GitHub Actions runs the unit tests on pushes and pull requests.

## Docker

```bash
docker build -t advanced-rag .
docker run --env-file .env -p 8501:8501 advanced-rag
```

## Repository structure

```text
Advanced-Multimodal-RAG/
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── config/
├── core/
│   ├── loaders.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── corpus_store.py
├── retrieval/
│   ├── vector.py
│   ├── bm25.py
│   ├── hybrid.py
│   ├── mmr.py
│   ├── reranker.py
│   └── pipeline.py
├── rag/
│   ├── llm_factory.py
│   ├── prompts.py
│   ├── context.py
│   ├── citations.py
│   ├── memory.py
│   ├── multimodal.py
│   └── chain.py
├── security/
│   └── guardrails.py
├── evaluation/
│   ├── metrics.py
│   ├── golden_dataset.json
│   ├── run_local_eval.py
│   ├── langsmith_eval.py
│   └── run_langsmith_eval.py
├── tests/
├── .github/workflows/tests.yml
├── Dockerfile
└── .streamlit/config.toml
```

## Audit

See `AUDIT.md` for the detailed before/after engineering review, resolved issues, and the boundary between a professional academic RAG and enterprise platform requirements.

## Scope / honest limitations

This is a strong **professional academic/portfolio RAG**. It intentionally does not pretend to include every enterprise platform concern. For a real multi-tenant production deployment you would still typically add authentication/authorization, tenant-level vector filtering, rate limiting, centralized secrets, a managed/distributed vector database, backups, malware scanning, full OCR for scanned PDFs, production logging/metrics, and load testing.
