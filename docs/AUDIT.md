# Professional RAG Audit

This repository was reviewed against the requirements of the course assignment and against common production-minded RAG engineering concerns.

## Required assignment capabilities

| Requirement | Implementation |
|---|---|
| Chunking | Recursive character chunking with overlap and stable chunk IDs |
| Embeddings | Local HuggingFace multilingual embeddings |
| Hybrid Search | Dense vector retrieval + BM25 fused with weighted Reciprocal Rank Fusion |
| MMR | Applied after hybrid candidate generation to reduce redundant evidence |
| Re-Ranker | Cross-Encoder second-stage re-ranking before generation |

All three required advanced retrieval techniques are executed in the live retrieval path, not only described in documentation.

## Issues found in the earlier version and fixes

| Earlier issue | Professional fix |
|---|---|
| LangSmith module only reported tracing status | Added real LangSmith dataset experiments and optional LLM judges |
| Chroma persisted but BM25 corpus lived only in Streamlit session | Added persisted `corpus.jsonl` + index metadata/signature |
| Memory affected answer generation but not retrieval | Added standalone follow-up query rewriting before retrieval |
| Context size could grow without a hard bound | Added `MAX_CONTEXT_CHARS` context budget |
| Source display existed but citation health was not checked | Added `[S#]` citation labels and citation audit diagnostics |
| English-centric retrieval defaults | Added multilingual embeddings and Persian/Arabic BM25 normalization |
| Only PDF/TXT ingestion | Added Markdown, CSV, HTML, DOCX and knowledge-base image caption indexing |
| Multimodal supported only a current image question | Added persistent KB image ingestion (vision-to-text indexing) plus query images |
| No retrieval quality harness | Added Precision@k, Recall@k, MRR and nDCG local evaluation |
| No explicit RAG quality experiment | Added LangSmith correctness/relevance/groundedness/retrieval-relevance judges |
| No persistence compatibility check | Added signature + vector/corpus count checks requiring safe re-index |
| Minimal security posture | Added upload limits, untrusted-context prompt, injection visibility and private-data provider policy |
| No deploy/CI packaging | Added Dockerfile, healthcheck, pytest suite and GitHub Actions |
| No latency visibility | Added per-stage Hybrid/MMR/Re-ranker and total retrieval timings |
| Vector clear removed the whole persistence directory | Clear now deletes only the configured Chroma collection |
| OpenRouter timeout treated like seconds | Fixed to `LLM_TIMEOUT_MS`; current ChatOpenRouter uses milliseconds |

## Current professional capabilities

### Ingestion and indexing
- PDF, TXT, Markdown, CSV, HTML and DOCX.
- Image knowledge sources through a vision model caption that becomes searchable evidence.
- Stable content-derived chunk IDs.
- Add/update, replace, and per-source delete knowledge-base lifecycle controls.
- Persistent Chroma dense index plus persistent lexical corpus.
- Index compatibility checks when embedding/chunk settings change.
- Persisted image assets with sanitized filenames.

### Retrieval
- Dense semantic search.
- BM25 lexical search with Persian/Arabic character normalization.
- Weighted RRF hybrid fusion.
- MMR diversity selection over hybrid candidates.
- Cross-Encoder re-ranking.
- Embedding caching for MMR candidate scoring within the process.
- Tunable Top-K, MMR lambda and final re-rank Top-K.
- Per-stage ranks, scores and latency diagnostics.

### Conversational RAG
- Bounded conversation memory.
- Follow-up query rewriting into a standalone retrieval query.
- Original user question retained for final generation.
- Memory reset when a knowledge base is replaced to avoid stale-context retrieval.

### Grounding and citations
- Retrieved context is explicitly marked as untrusted evidence.
- Bounded context window.
- Stable `[S1]`, `[S2]`, ... evidence labels.
- UI source cards show source/page/chunk and retrieval scores.
- Citation audit reports missing or impossible source labels.

### Multimodal
- Knowledge-base image -> vision caption -> chunk/embed/retrieve.
- Current-query image + retrieved text -> vision-capable LLM.
- This is intentionally described as **vision-to-text multimodal RAG**, not falsely presented as CLIP-style joint image/text embeddings.

### Evaluation
- Golden dataset included.
- Local retrieval metrics: Precision@k, Recall@k, MRR, nDCG.
- Retrieval stages can be compared: Hybrid vs MMR vs Re-ranked.
- LangSmith tracing.
- LangSmith dataset experiment.
- Deterministic evaluators.
- Optional LLM-as-judge for correctness, answer relevance, groundedness and retrieval relevance.

### Security / reliability
- `.env` excluded from version control.
- Upload extension allow-list, file-size limit and batch-count limit.
- Filename sanitization for stored image assets.
- English and Persian common prompt-injection phrase detection for visibility.
- System prompt explicitly prevents instructions inside retrieved documents from becoming executable instructions.
- OpenRouter provider `data_collection=deny` can be enabled by configuration and is enabled by default in the example config.
- Bounded LLM output, configurable timeout/retry policy.
- Corpus state is written using atomic file replacement.
- Dense/lexical mismatch is detected after interrupted or incompatible indexing.

### Engineering / delivery
- Modular packages rather than one monolithic script.
- Central environment-driven settings with validation.
- Streamlit UI.
- Unit tests for retrieval, persistence, context budget, citations and guardrails.
- GitHub Actions test workflow.
- Docker image and Streamlit healthcheck.
- README and Persian project report.

## Honest boundary

There is no finite checklist called "all professional RAG features". This repository is intentionally a **strong professional academic/portfolio implementation**, not a multi-tenant enterprise platform.

For an internet-facing enterprise deployment, the next layer would normally include: authentication/authorization, tenant/document ACL filtering at retrieval time, rate limiting, centralized secret management, managed/distributed vector infrastructure, backup/restore strategy, malware/content scanning, OCR for scanned documents, centralized structured logs/metrics, load/chaos testing, data-retention policies and a formal threat model.

Those concerns are documented rather than being faked as implemented.
