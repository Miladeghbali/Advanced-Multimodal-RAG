from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from config.settings import settings
from core.chunker import split_documents
from core.corpus_store import CorpusStore
from core.loaders import SUPPORTED_EXTENSIONS, load_file
from core.vector_store import VectorStoreManager
from evaluation.langsmith_eval import langsmith_status
from rag.chain import RAGChain
from rag.citations import audit_citations
from rag.memory import ConversationMemory
from rag.multimodal import create_image_content
from retrieval.pipeline import AdvancedRetrievalPipeline
from retrieval.reranker import Reranker
from security.guardrails import detect_prompt_injection, validate_upload
from ui.components import bi, bi_help, render_header, render_language_selector, render_pipeline, render_section_title, render_sources
from ui.styles import CSS

st.set_page_config(
    page_title="سامانه حرفه‌ای RAG | Professional Advanced RAG",
    page_icon="🧠",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)

DOCUMENT_TYPES = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
IMAGE_EXTENSIONS = {f".{x}" for x in IMAGE_TYPES}


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_reranker() -> Reranker:
    return Reranker()


if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store()
if "corpus_store" not in st.session_state:
    st.session_state.corpus_store = CorpusStore()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chunks" not in st.session_state:
    persisted = st.session_state.corpus_store.load()
    compatible = st.session_state.corpus_store.is_compatible()
    vector_count = st.session_state.vector_store.count()
    if compatible and persisted and vector_count == len(persisted):
        st.session_state.chunks = persisted
        st.session_state.index_warning = ""
    elif vector_count == 0 and not persisted:
        st.session_state.chunks = []
        st.session_state.index_warning = ""
    else:
        st.session_state.chunks = []
        st.session_state.index_warning = "incompatible_index"

with st.sidebar:
    render_language_selector()
    st.divider()

render_header()
render_pipeline()

with st.sidebar:
    st.header(bi("⚙️ تنظیمات بازیابی", "⚙️ Retrieval settings"))
    st.caption(bi("برای دیدن توضیح هر گزینه روی علامت ? کنار همان کنترل بروید.", "Use the ? next to each control for help."))

    top_k = st.slider(
        bi("تعداد کاندیدهای Hybrid", "Hybrid candidate count"),
        4,
        30,
        min(settings.top_k, 30),
        help=bi_help(
            "تعداد نتایج اولیه‌ای که از ترکیب Vector Search و BM25 با RRF وارد مرحله بعد می‌شوند. مقدار بزرگ‌تر پوشش بیشتری می‌دهد ولی پردازش را سنگین‌تر می‌کند.",
            "Number of first-stage candidates produced by Vector Search + BM25 + RRF. Higher values improve recall but increase downstream cost.",
        ),
    )
    mmr_k = st.slider(
        bi("تعداد انتخاب‌های MMR", "MMR selected candidates"),
        2,
        min(15, top_k),
        min(settings.mmr_k, top_k, 15),
        help=bi_help(
            "MMR از بین کاندیدهای Hybrid، نتایج مرتبط ولی غیرتکراری‌تر را انتخاب می‌کند. این مقدار تعداد خروجی‌های MMR است.",
            "MMR selects relevant but less redundant candidates from the Hybrid results. This value is the number of documents kept after MMR.",
        ),
    )
    mmr_lambda = st.slider(
        bi("ضریب λ در MMR (ارتباط ↔ تنوع)", "MMR λ (relevance ↔ diversity)"),
        0.0,
        1.0,
        settings.mmr_lambda,
        0.05,
        help=bi_help(
            "λ بزرگ‌تر یعنی تأکید بیشتر روی ارتباط با سؤال؛ λ کوچک‌تر یعنی تأکید بیشتر روی تنوع و حذف نتایج مشابه.",
            "A higher λ emphasizes query relevance; a lower λ emphasizes diversity and penalizes near-duplicate context.",
        ),
    )
    rerank_k = st.slider(
        bi("تعداد نهایی Re-Ranker", "Final Re-Ranker Top-K"),
        1,
        min(10, mmr_k),
        min(settings.rerank_top_k, mmr_k, 10),
        help=bi_help(
            "Cross-Encoder روی خروجی MMR اجرا می‌شود و این تعداد سند با بالاترین امتیاز را برای Context نهایی نگه می‌دارد.",
            "The Cross-Encoder re-ranks MMR candidates and keeps this many top-scoring documents for the final LLM context.",
        ),
    )

    st.divider()
    st.subheader(bi("📊 وضعیت سیستم", "📊 System status"))
    st.metric(
        bi("تعداد Chunkهای Vector DB", "Vector DB chunks"),
        st.session_state.vector_store.count(),
        help=bi_help(
            "تعداد قطعه‌هایی که در پایگاه برداری Chroma ذخیره شده‌اند.",
            "Number of chunks currently stored in the Chroma vector database.",
        ),
    )
    st.metric(
        bi("تعداد Chunkهای BM25", "BM25 corpus chunks"),
        len(st.session_state.chunks),
        help=bi_help(
            "تعداد قطعه‌هایی که برای جستجوی واژگانی BM25 نگه‌داری می‌شوند.",
            "Number of chunks available to the lexical BM25 retriever.",
        ),
    )
    source_count = len({doc.metadata.get("source") for doc in st.session_state.chunks})
    st.metric(
        bi("تعداد منابع دانشی", "Knowledge sources"),
        source_count,
        help=bi_help(
            "تعداد فایل‌ها یا تصاویر مستقل موجود در Knowledge Base.",
            "Number of distinct files or images currently represented in the knowledge base.",
        ),
    )

    sources = sorted(
        {str(doc.metadata.get("source")) for doc in st.session_state.chunks if doc.metadata.get("source")}
    )
    if sources:
        delete_source = st.selectbox(
            bi("منبع Knowledge Base", "Knowledge-base source"),
            sources,
            key="delete_source_select",
            help=bi_help(
                "یک منبع مشخص را برای حذف انتخاب کنید. پس از حذف، Vector DB و corpus مربوط به BM25 دوباره همگام می‌شوند.",
                "Select one indexed source to delete. The vector index and BM25 corpus are rebuilt to stay synchronized.",
            ),
        )
        if st.button(
            bi("🧾 حذف منبع انتخاب‌شده", "🧾 Delete selected source"),
            use_container_width=True,
            help=bi_help(
                "فقط منبع انتخاب‌شده را از Knowledge Base پاک می‌کند.",
                "Removes only the selected source from the knowledge base.",
            ),
        ):
            removed = [
                doc for doc in st.session_state.chunks
                if str(doc.metadata.get("source")) == delete_source
            ]
            remaining = [
                doc for doc in st.session_state.chunks
                if str(doc.metadata.get("source")) != delete_source
            ]
            for doc in removed:
                asset_path = doc.metadata.get("asset_path")
                if asset_path:
                    try:
                        Path(str(asset_path)).unlink(missing_ok=True)
                    except Exception:
                        pass
            st.session_state.vector_store.clear()
            st.session_state.vector_store.add_documents(remaining)
            st.session_state.corpus_store.save(remaining)
            st.session_state.chunks = remaining
            st.session_state.pipeline = None
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.session_state.index_warning = ""
            st.rerun()

    status = langsmith_status()
    st.caption(
        bi(
            f"LangSmith — ارزیابی و رهگیری: {'فعال' if status['enabled'] else 'اختیاری/خاموش'}",
            f"LangSmith — evaluation & tracing: {'enabled' if status['enabled'] else 'optional/off'}",
        )
    )
    st.caption(bi(f"مدل Embedding: {settings.embedding_model}", f"Embedding model: {settings.embedding_model}"))
    st.caption(bi(f"مدل Re-ranker: {settings.reranker_model}", f"Re-ranker model: {settings.reranker_model}"))

    if st.session_state.get("index_warning"):
        if st.session_state.index_warning == "incompatible_index":
            st.warning(bi(
                "ایندکس برداری و corpus واژگانی ذخیره‌شده ناقص، ناسازگار یا مربوط به تنظیمات قدیمی Chunk/Embedding هستند. پایگاه دانش را دوباره ایندکس کنید.",
                "The persisted vector index and lexical corpus are missing, inconsistent, or were built with different chunk/embedding settings. Re-index the knowledge base.",
            ))
        else:
            st.warning(st.session_state.index_warning)

    if st.button(
        bi("🗑️ پاک‌کردن Knowledge Base", "🗑️ Clear knowledge base"),
        use_container_width=True,
        help=bi_help(
            "همه اسناد، Chunkها و ایندکس‌های ذخیره‌شده را پاک می‌کند. تاریخچه گفتگو هم ریست می‌شود.",
            "Deletes all indexed documents/chunks and resets the conversation state.",
        ),
    ):
        st.session_state.vector_store.clear()
        st.session_state.corpus_store.clear()
        st.session_state.chunks = []
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.pipeline = None
        st.session_state.index_warning = ""
        st.rerun()

    if st.button(
        bi("🧹 پاک‌کردن گفتگو", "🧹 Clear conversation"),
        use_container_width=True,
        help=bi_help(
            "فقط Memory و پیام‌های گفتگو را پاک می‌کند و Knowledge Base را نگه می‌دارد.",
            "Clears only chat history and memory while preserving the knowledge base.",
        ),
    ):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

render_section_title(
    "1) ساخت پایگاه دانش",
    "1) Build the knowledge base",
    "در این بخش فایل‌های متنی و در صورت نیاز تصاویر را وارد می‌کنید. اسناد پس از استخراج متن، Chunking و Embedding وارد پایگاه دانش می‌شوند. تصاویر Knowledge Base ابتدا توسط مدل Vision توصیف و سپس متن آن‌ها ایندکس می‌شود.",
    "Upload text documents and optional images here. Documents are extracted, chunked, embedded, and indexed. Knowledge-base images are first captioned by the vision model and then indexed as searchable text evidence.",
    icon="📚",
)
st.caption(
    bi("دانش متنی: PDF، TXT، Markdown، CSV، HTML و DOCX. تصاویر اختیاری نیز با مسیر Vision → Text قابل جستجو می‌شوند.", "Text knowledge: PDF, TXT, Markdown, CSV, HTML, DOCX. Optional images use Vision → Text indexing.")
)

uploaded_files = st.file_uploader(
    bi("اسناد", "Documents"),
    type=DOCUMENT_TYPES,
    accept_multiple_files=True,
    key="kb_docs",
    help=bi_help(
        "فایل‌های متنی Knowledge Base را انتخاب کنید. چند فایل را می‌توانید همزمان بارگذاری کنید.",
        "Choose the text documents that should become part of the knowledge base. Multiple files can be uploaded together.",
    ),
)
kb_images = st.file_uploader(
    bi("تصاویر اختیاری پایگاه دانش (Vision → Text)", "Optional knowledge-base images (Vision → Text)"),
    type=IMAGE_TYPES,
    accept_multiple_files=True,
    key="kb_images",
    help=bi_help(
        "مدل Vision محتوای هر تصویر را به متن تبدیل می‌کند و همان متن مانند یک سند قابل بازیابی ایندکس می‌شود.",
        "A vision-capable model turns each image into a textual description, which is then indexed and retrieved like document evidence.",
    ),
)
index_mode = st.radio(
    bi("حالت ایندکس", "Index mode"),
    ["replace", "add_update"],
    format_func=lambda value: (
        bi("جایگزینی کامل", "Replace knowledge base")
        if value == "replace"
        else bi("افزودن/به‌روزرسانی با حفظ منابع قبلی", "Add/update while preserving existing sources")
    ),
    horizontal=True,
    help=bi_help(
        "در حالت Replace همه Knowledge Base با فایل‌های جدید جایگزین می‌شود. در Add/Update، منابع قبلی حفظ می‌شوند و فایل هم‌نام به‌روزرسانی می‌شود.",
        "Replace rebuilds the entire knowledge base from the new upload. Add/Update preserves existing sources and replaces incoming sources with the same name.",
    ),
)

if (uploaded_files or kb_images) and st.button(
    bi("🚀 پردازش پایگاه دانش", "🚀 Process knowledge base"),
    type="primary",
    help=bi_help(
        "استخراج متن، Chunking، Embedding و ساخت/به‌روزرسانی ایندکس‌های Vector و BM25 را شروع می‌کند.",
        "Starts extraction, chunking, embedding, and rebuilding/updating both the vector and BM25 indexes.",
    ),
):
    all_uploads = list(uploaded_files or []) + list(kb_images or [])
    if len(all_uploads) > settings.max_files_per_batch:
        st.error(bi(
            f"حداکثر {settings.max_files_per_batch} فایل در هر مرحله بارگذاری کنید.",
            f"Upload at most {settings.max_files_per_batch} files per batch.",
        ))
        st.stop()

    new_documents: list[Document] = []
    suspicious_documents = 0
    errors: list[str] = []

    with st.status(bi("در حال ایندکس پایگاه دانش...", "Indexing knowledge base..."), expanded=True) as status_box:
        for uploaded in uploaded_files or []:
            try:
                validate_upload(uploaded.name, uploaded.size, SUPPORTED_EXTENSIONS)
                suffix = Path(uploaded.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                    temp_path = Path(tmp.name)
                try:
                    loaded = load_file(temp_path)
                    if not loaded:
                        errors.append(
                            bi(
                                f"{uploaded.name}: متن قابل استخراج پیدا نشد. اگر PDF اسکن‌شده است، صفحات را به‌صورت تصویر وارد کنید یا OCR اضافه کنید.",
                                f"{uploaded.name}: no extractable text; for scanned PDFs, upload pages as images or add OCR.",
                            )
                        )
                    for doc in loaded:
                        doc.metadata["source"] = uploaded.name
                        doc.metadata["source_type"] = suffix.lstrip(".")
                        if detect_prompt_injection(doc.page_content):
                            suspicious_documents += 1
                    new_documents.extend(loaded)
                finally:
                    temp_path.unlink(missing_ok=True)
            except Exception as exc:
                errors.append(f"{uploaded.name}: {exc}")

        if kb_images:
            try:
                vision_chain = RAGChain()
            except Exception as exc:
                vision_chain = None
                errors.append(bi(f"ایندکس تصویر در دسترس نیست: {exc}", f"Image indexing unavailable: {exc}"))

            if vision_chain is not None:
                for uploaded in kb_images:
                    try:
                        validate_upload(uploaded.name, uploaded.size, IMAGE_EXTENSIONS)
                        suffix = Path(uploaded.name).suffix.lower()
                        image_bytes = bytes(uploaded.getbuffer())
                        asset_path = st.session_state.corpus_store.save_asset(uploaded.name, image_bytes)
                        caption = vision_chain.caption_image(asset_path)
                        if detect_prompt_injection(caption):
                            suspicious_documents += 1
                        new_documents.append(
                            Document(
                                page_content=caption,
                                metadata={
                                    "source": uploaded.name,
                                    "page": 1,
                                    "source_type": "image",
                                    "multimodal_caption": True,
                                    "asset_path": str(asset_path),
                                },
                            )
                        )
                    except Exception as exc:
                        errors.append(f"{uploaded.name}: {exc}")

        if not new_documents:
            status_box.update(label=bi("چیزی ایندکس نشد", "Nothing was indexed"), state="error")
        else:
            new_chunks = split_documents(new_documents)
            if index_mode == "add_update":
                incoming_sources = {str(doc.metadata.get("source")) for doc in new_chunks}
                combined = {
                    str(doc.metadata.get("chunk_id")): doc
                    for doc in st.session_state.chunks
                    if str(doc.metadata.get("source")) not in incoming_sources
                }
                for doc in new_chunks:
                    combined[str(doc.metadata.get("chunk_id"))] = doc
                final_chunks = list(combined.values())
            else:
                final_chunks = new_chunks

            # Rebuild dense and lexical stores from the exact same corpus.
            # On startup, count/signature checks detect any interrupted mismatch.
            st.session_state.vector_store.clear()
            st.session_state.vector_store.add_documents(final_chunks)
            st.session_state.corpus_store.save(final_chunks)
            st.session_state.chunks = final_chunks
            st.session_state.pipeline = None
            st.session_state.index_warning = ""
            if index_mode == "replace":
                # Prevent follow-up query rewriting from carrying stale context
                # from a knowledge base that has just been replaced.
                st.session_state.messages = []
                st.session_state.memory.clear()
            source_total = len({d.metadata.get("source") for d in final_chunks})
            status_box.update(
                label=bi(
                    f"{len(final_chunks)} Chunk از {source_total} منبع ایندکس شد",
                    f"Indexed {len(final_chunks)} chunks from {source_total} sources",
                ),
                state="complete",
            )

    if suspicious_documents:
        st.warning(bi(
            f"هشدار امنیتی: در {suspicious_documents} سند/صفحه عباراتی شبیه Prompt Injection دیده شد. این محتوا همچنان به‌عنوان evidence غیرقابل‌اعتماد ایندکس می‌شود؛ در صورت غیرمنتظره بودن منبع را بررسی کنید.",
            f"Security notice: {suspicious_documents} ingested document/page objects contain phrases resembling prompt-injection attempts; they remain indexed as untrusted evidence.",
        ))
    for error in errors:
        st.warning(error)

render_section_title(
    "2) تصویر اختیاری سؤال",
    "2) Optional query image",
    "اگر سؤال شما درباره یک تصویر است، آن را اینجا ضمیمه کنید. تصویر همراه با سؤال و Context بازیابی‌شده برای مدل Vision ارسال می‌شود.",
    "Attach an image here when the current question depends on visual content. The image is sent to the vision-capable model together with the question and retrieved context.",
    icon="🖼️",
)
query_image = st.file_uploader(
    bi("یک تصویر به سؤال فعلی ضمیمه کنید", "Attach one image to the current question"),
    type=IMAGE_TYPES,
    key="query_image",
    help=bi_help(
        "این تصویر فقط برای سؤال فعلی استفاده می‌شود و به‌طور خودکار وارد Knowledge Base نمی‌شود.",
        "This image is used only for the current question and is not automatically added to the knowledge base.",
    ),
)
if query_image:
    st.image(query_image, width=360, caption=bi("تصویر سؤال", "Query image"))

render_section_title(
    "3) پرسش از پایگاه دانش",
    "3) Ask questions",
    "سؤال خود را فارسی یا انگلیسی بنویسید. اگر Memory فعال باشد، سیستم برای سؤال‌های دنباله‌دار یک Query مستقل می‌سازد، سپس Hybrid Search → MMR → Re-Ranker اجرا می‌شود.",
    "Ask in Persian or English. With conversation memory, follow-up questions can be rewritten into standalone retrieval queries before Hybrid Search → MMR → Re-Ranking.",
    icon="💬",
)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input(bi("سؤال خود را درباره پایگاه دانش بنویسید...", "Ask about the indexed knowledge base..."))
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    query_flags = detect_prompt_injection(query)
    if query_flags:
        st.warning(bi(
            "متن سؤال شامل الگوهایی شبیه Prompt Injection است و به‌عنوان ورودی غیرقابل‌اعتماد پردازش می‌شود.",
            "The query contains wording commonly seen in prompt-injection attempts and will be handled as untrusted input.",
        ))

    final_docs: list[Document] = []
    retrieval_result = None
    rewritten_query = query
    retrieval_error = None

    try:
        chain = RAGChain()
    except Exception as exc:
        chain = None
        generation_error = str(exc)
    else:
        generation_error = None
        if st.session_state.chunks and st.session_state.memory.get():
            try:
                rewritten_query = chain.rewrite_query(query, st.session_state.memory.as_text())
            except Exception:
                rewritten_query = query

    if st.session_state.chunks:
        try:
            if st.session_state.pipeline is None:
                st.session_state.pipeline = AdvancedRetrievalPipeline(
                    st.session_state.vector_store,
                    st.session_state.chunks,
                    get_reranker(),
                )
            retrieval_result = st.session_state.pipeline.retrieve(
                rewritten_query,
                top_k=top_k,
                mmr_k=mmr_k,
                mmr_lambda=mmr_lambda,
                rerank_top_k=rerank_k,
            )
            final_docs = retrieval_result.reranked
        except Exception as exc:
            retrieval_error = str(exc)

    if not st.session_state.chunks and query_image is None:
        answer = bi("ابتدا Knowledge Base را بسازید یا برای سؤال فقط تصویری، یک تصویر ضمیمه کنید.", "Please build the knowledge base first, or attach an image for a vision-only question.")
    elif chain is None:
        answer = bi(f"خطای تولید پاسخ: `{generation_error}`", f"Answer-generation error: `{generation_error}`")
    elif retrieval_error and query_image is None:
        answer = bi(f"خطای بازیابی: `{retrieval_error}`", f"Retrieval error: `{retrieval_error}`")
    else:
        try:
            if query_image:
                validate_upload(query_image.name, query_image.size, IMAGE_EXTENSIONS)
                suffix = Path(query_image.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(query_image.getbuffer())
                    image_path = Path(tmp.name)
                try:
                    image_content = create_image_content(image_path, query)
                    answer = chain.answer_with_image(
                        query,
                        image_content,
                        final_docs,
                        history=st.session_state.memory.get(),
                    )
                finally:
                    image_path.unlink(missing_ok=True)
            else:
                answer = chain.answer(
                    query, final_docs, history=st.session_state.memory.get()
                )
        except Exception as exc:
            answer = bi(f"خطای تولید پاسخ: `{exc}`", f"Answer-generation error: `{exc}`")

    st.session_state.memory.add_user(query)
    st.session_state.memory.add_ai(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
        render_sources(final_docs)

    with st.expander(bi("🔬 جزئیات بازیابی و ایمنی", "🔬 Retrieval / safety diagnostics")):
        st.caption(bi(
            "❓ این بخش برای ارائه فنی مفید است: Query Rewrite، تعداد خروجی Hybrid/MMR/Re-Ranker، زمان اجرا و اعتبار Citationها را نشان می‌دهد.",
            "❓ Useful for technical demos: shows query rewriting, Hybrid/MMR/Re-Ranker counts, latency, and citation checks.",
        ))
        st.write(f"{bi('سؤال اصلی', 'Original query')}: `{query}`")
        st.write(f"{bi('Query مستقل برای بازیابی', 'Standalone retrieval query')}: `{rewritten_query}`")
        if retrieval_result is not None:
            st.write(f"{bi('کاندیدهای Hybrid/RRF', 'Hybrid/RRF candidates')}: {len(retrieval_result.hybrid)}")
            st.write(f"{bi('کاندیدهای MMR', 'MMR candidates')}: {len(retrieval_result.mmr)}")
            st.write(f"{bi('Context نهایی پس از Re-Ranker', 'Re-ranked final context')}: {len(retrieval_result.reranked)}")
            timings = retrieval_result.timings_ms
            st.write(
                f"{bi('زمان بازیابی', 'Retrieval latency')}: "
                f"Hybrid/RRF {timings.get('hybrid_rrf', 0):.1f} ms · "
                f"MMR {timings.get('mmr', 0):.1f} ms · "
                f"Re-ranker {timings.get('reranker', 0):.1f} ms · "
                f"Total {timings.get('total_retrieval', 0):.1f} ms"
            )
            st.code("Dense + BM25 → RRF Hybrid → MMR → Cross-Encoder Re-Ranker → Bounded Context → LLM")
        if retrieval_error:
            st.error(retrieval_error)
        citation_audit = audit_citations(answer, len(final_docs))
        if final_docs and not citation_audit.has_valid_citation:
            st.warning(bi(
                "با وجود evidence بازیابی‌شده، پاسخ Citation معتبر [S#] ندارد.",
                "The generated answer did not include a valid [S#] citation even though retrieved evidence was available.",
            ))
        if citation_audit.invalid_indices:
            st.warning(
                bi("پاسخ به برچسب منبع نامعتبر اشاره کرده است", "The answer referenced invalid source labels")
                + ": " + ", ".join(f"[S{i}]" for i in citation_audit.invalid_indices)
            )
        suspicious_retrieved = [
            doc.metadata.get("source", "Unknown")
            for doc in final_docs
            if detect_prompt_injection(doc.page_content)
        ]
        if suspicious_retrieved:
            st.warning(
                bi("متن مشکوک به Prompt Injection غیرمستقیم از این منابع بازیابی شد", "Potential indirect prompt-injection text was retrieved from")
                + ": " + ", ".join(sorted(set(map(str, suspicious_retrieved))))
            )

with st.expander(bi("📊 دستورات ارزیابی", "📊 Evaluation commands")):
    st.caption(bi(
        "❓ ارزیابی محلی معیارهای Retrieval را محاسبه می‌کند و LangSmith برای experiment/tracing استفاده می‌شود.",
        "❓ Local evaluation computes retrieval metrics; LangSmith is used for experiments and tracing.",
    ))
    st.code("python -m evaluation.run_local_eval", language="bash")
    st.caption(bi("معیارهای قطعی بازیابی را اجرا می‌کند: Precision@k، Recall@k، MRR و nDCG.", "Runs deterministic retrieval metrics: Precision@k, Recall@k, MRR, and nDCG."))
    st.code("python -m evaluation.run_langsmith_eval", language="bash")
    st.caption(bi("اگر اطلاعات LangSmith تنظیم شده باشد، یک Dataset Experiment اجرا می‌کند.", "Runs a LangSmith dataset experiment when credentials are configured."))
