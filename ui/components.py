from __future__ import annotations

import html
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document


LANGUAGE_OPTIONS = {
    "fa": "🇮🇷 فارسی",
    "en": "🇬🇧 English",
    "bi": "🌐 فارسی + English",
}


def get_ui_language() -> str:
    return str(st.session_state.get("ui_language", "bi"))


def bi(fa: str, en: str) -> str:
    """Return UI text in Persian, English, or bilingual mode."""
    lang = get_ui_language()
    if lang == "fa":
        return fa
    if lang == "en":
        return en
    return f"{fa} | {en}"


def bi_help(fa: str, en: str) -> str:
    """Return tooltip/help text in the currently selected UI language."""
    lang = get_ui_language()
    if lang == "fa":
        return fa
    if lang == "en":
        return en
    return f"🇮🇷 فارسی: {fa}\n\n🇬🇧 English: {en}"


def render_language_selector() -> str:
    """Render the persistent UI language selector and return the selected mode."""
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = "bi"

    selected = st.radio(
        "🌐 زبان رابط | Interface language",
        options=list(LANGUAGE_OPTIONS),
        format_func=lambda key: LANGUAGE_OPTIONS[key],
        key="ui_language",
        horizontal=False,
        help=(
            "زبان رابط کاربری را انتخاب کنید. این تنظیم فقط متن‌های رابط را تغییر می‌دهد و روی Pipeline بازیابی اثر ندارد.\n\n"
            "Choose the interface language. This only changes UI text and does not affect the retrieval pipeline."
        ),
    )
    return str(selected)


def render_header() -> None:
    lang = get_ui_language()
    if lang == "fa":
        kicker = "🇮🇷 رابط فارسی"
        title = "🧠 سامانه حرفه‌ای RAG"
        subtitle = "جستجوی ترکیبی (Dense + BM25 + RRF) → MMR → بازرتبه‌بندی Cross-Encoder → پاسخ مستند"
    elif lang == "en":
        kicker = "🇬🇧 English interface"
        title = "🧠 Professional Advanced RAG"
        subtitle = "Hybrid Search (Dense + BM25 + RRF) → MMR → Cross-Encoder Re-Ranker → Grounded LLM"
    else:
        kicker = "🇮🇷 فارسی &nbsp; • &nbsp; 🇬🇧 English"
        title = "🧠 سامانه حرفه‌ای RAG | Professional Advanced RAG"
        subtitle = (
            "جستجوی ترکیبی (Dense + BM25 + RRF) → MMR → بازرتبه‌بندی Cross-Encoder → پاسخ مستند"
            "<br><span class='hero-en'>Hybrid Search (Dense + BM25 + RRF) → MMR → Cross-Encoder Re-Ranker → Grounded LLM</span>"
        )

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">{kicker}</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline() -> None:
    labels = [
        bi("📄 ورود چندفرمتی", "📄 Multiformat ingestion"),
        bi("✂️ قطعه‌بندی", "✂️ Chunking"),
        bi("🔢 بردارسازی", "🔢 Embeddings"),
        bi("🔎 بازیابی معنایی", "🔎 Dense retrieval"),
        bi("📝 جستجوی واژگانی", "📝 BM25"),
        bi("🔀 جستجوی ترکیبی", "🔀 RRF Hybrid"),
        bi("🎯 تنوع نتایج", "🎯 MMR"),
        bi("🏆 بازرتبه‌بندی", "🏆 Re-Ranker"),
        bi("🧠 بازنویسی سؤال", "🧠 Query rewrite"),
        bi("🤖 پاسخ مستند", "🤖 Grounded LLM"),
        bi("📊 ارزیابی", "📊 Evaluation"),
    ]
    st.markdown(
        '<div class="pipeline-wrap">'
        + "".join(f'<span class="badge">{html.escape(x)}</span>' for x in labels)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_section_title(
    title_fa: str,
    title_en: str,
    help_fa: str,
    help_en: str,
    *,
    icon: str = "",
) -> None:
    """Localized section heading with an explicit ? help popover."""
    title_col, help_col = st.columns([0.94, 0.06], vertical_alignment="center")
    with title_col:
        st.subheader(f"{icon} {bi(title_fa, title_en)}".strip())
    with help_col:
        with st.popover("❓", help=bi_help(help_fa, help_en)):
            lang = get_ui_language()
            if lang in {"fa", "bi"}:
                st.markdown(f"**🇮🇷 راهنما**\n\n{help_fa}")
            if lang in {"en", "bi"}:
                st.markdown(f"**🇬🇧 Help**\n\n{help_en}")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value) if value is not None else "-"


def render_sources(documents: list[Document]) -> None:
    if not documents:
        return

    render_section_title(
        "منابع نهایی بازیابی‌شده",
        "Final retrieved sources",
        "این بخش دقیقاً نشان می‌دهد کدام قطعه‌ها بعد از Hybrid Search، MMR و Re-Ranker وارد Context نهایی شده‌اند. برچسب‌های [S1]، [S2] و ... در پاسخ به همین منابع اشاره می‌کنند.",
        "This section shows the exact chunks that survived Hybrid Search, MMR, and the Re-Ranker and were sent to the final context. Answer citations such as [S1] and [S2] point to these sources.",
        icon="📚",
    )

    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        preview = html.escape(doc.page_content[:500])
        st.markdown(
            f"""
            <div class="source-card">
              <b>[S{i}] {html.escape(str(meta.get('source','Unknown')))}</b><br>
              {bi('نوع', 'Type')}: {_fmt(meta.get('source_type'))} | {bi('صفحه', 'Page')}: {_fmt(meta.get('page'))} | {bi('قطعه', 'Chunk')}: {_fmt(meta.get('chunk_id'))}<br>
              {bi('رتبه/امتیاز برداری', 'Vector rank/score')}: {_fmt(meta.get('vector_rank'))} / {_fmt(meta.get('vector_score'))}<br>
              {bi('رتبه/امتیاز BM25', 'BM25 rank/score')}: {_fmt(meta.get('bm25_rank'))} / {_fmt(meta.get('bm25_score'))}<br>
              {bi('رتبه/امتیاز Hybrid', 'Hybrid rank/score')}: {_fmt(meta.get('hybrid_rank'))} / {_fmt(meta.get('hybrid_score'))}<br>
              {bi('رتبه MMR', 'MMR rank')}: {_fmt(meta.get('mmr_rank'))} | {bi('رتبه/امتیاز Re-ranker', 'Re-ranker rank/score')}: {_fmt(meta.get('reranker_rank'))} / {_fmt(meta.get('reranker_score'))}<br>
              <details><summary>{bi('پیش‌نمایش', 'Preview')}</summary><div style="margin-top:8px">{preview}</div></details>
            </div>
            """,
            unsafe_allow_html=True,
        )
        asset_path = meta.get("asset_path")
        if meta.get("source_type") == "image" and asset_path and Path(str(asset_path)).exists():
            st.image(
                str(asset_path),
                width=320,
                caption=f"[S{i}] {meta.get('source','image')}",
            )
