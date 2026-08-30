# گزارش پروژه RAG پیشرفته و حرفه‌ای

## 1. هدف
این پروژه یک سامانه Retrieval-Augmented Generation است که سه روش پیشرفته‌ی خواسته‌شده در صورت پروژه را واقعاً در مسیر بازیابی اجرا می‌کند:

1. **Hybrid Search** با ترکیب Dense Vector Search و BM25 و ادغام رتبه‌ها به روش RRF.
2. **MMR** برای کاهش افزونگی و افزایش تنوع context.
3. **Cross-Encoder Re-Ranker** برای رتبه‌بندی دقیق‌تر candidateها پیش از ارسال به مدل زبانی.

علاوه بر این سه الزام، قابلیت‌های حرفه‌ای شامل Memory، Query Rewrite، Multimodal، Persistence، Evaluation، LangSmith، Citation، Security، Testing و Docker نیز اضافه شده‌اند.

## 2. ورود داده
سامانه از PDF، TXT، Markdown، CSV، HTML و DOCX پشتیبانی می‌کند. برای هر داده metadata شامل نام منبع، صفحه و نوع فایل نگهداری می‌شود.

برای تصاویر knowledge base، یک مدل vision تصویر را به توضیح متنی قابل جست‌وجو تبدیل می‌کند. این توضیح مانند سایر اسناد chunk و embed می‌شود و خود تصویر نیز برای نمایش منبع در state محلی نگهداری می‌شود.

## 3. Chunking و شناسه پایدار
متن با `RecursiveCharacterTextSplitter` و overlap قابل تنظیم chunk می‌شود. هر chunk یک `chunk_id` پایدار بر اساس source، page و content دارد. این شناسه برای deduplication و نمایش مسیر retrieval استفاده می‌شود.

## 4. Embedding و Vector Store
Embedding با Sentence Transformers ساخته و در Chroma ذخیره می‌شود. مدل پیش‌فرض چندزبانه است تا retrieval محدود به متن انگلیسی نباشد. تنظیمات مدل و chunking در index signature ثبت می‌شوند تا اگر مدل embedding یا اندازه chunk عوض شد، index قدیمی به اشتباه استفاده نشود.

## 5. BM25 و Persistence
BM25 بازیابی واژگانی را انجام می‌دهد و برای exact term، نام‌ها و عبارت‌های فنی مفید است. برای بهبود جست‌وجوی فارسی، برخی حروف عربی/فارسی مانند `ي/ی` و `ك/ک` normalize می‌شوند.

از آنجا که Chroma فقط بخش dense را persist می‌کند، corpus اصلی chunkها نیز در `rag_state/corpus.jsonl` ذخیره می‌شود. بنابراین بعد از restart برنامه، BM25 نیز قابل بازسازی است و Hybrid Search از بین نمی‌رود.

## 6. Hybrid Search با RRF
Dense و BM25 مستقل اجرا می‌شوند. سپس رتبه‌ها با Weighted Reciprocal Rank Fusion ترکیب می‌شوند. استفاده از RRF باعث می‌شود نیازی به هم‌مقیاس کردن مستقیم scoreهای BM25 و vector similarity نباشد.

وزن Dense و BM25 و مقدار ثابت RRF از `.env` قابل تنظیم‌اند.

## 7. MMR
روی candidateهای حاصل از Hybrid Search، Maximum Marginal Relevance اجرا می‌شود. MMR همزمان relevance به query و similarity بین candidateها را در نظر می‌گیرد؛ در نتیجه context نهایی کمتر تکراری می‌شود.

## 8. Re-Ranking
پس از MMR، یک Cross-Encoder زوج query-passage را مستقیماً score می‌کند. چون Cross-Encoder سنگین‌تر از retrieval اولیه است، فقط روی candidateهای محدود اجرا می‌شود. نتیجه نهایی Top-K به LLM می‌رود.

## 9. Memory و Query Rewrite
حافظه مکالمه تعداد محدودی از پیام‌های اخیر را نگه می‌دارد. برای سؤال follow-up مانند «مزیتش چیست؟»، پیش از retrieval سؤال با استفاده از history به یک query مستقل تبدیل می‌شود. این موضوع باعث می‌شود Memory فقط در تولید پاسخ استفاده نشود، بلکه کیفیت retrieval هم بهتر شود.

## 10. Context Management و Citation
Context نهایی سقف حجمی دارد تا تعداد اسناد زیاد باعث رشد کنترل‌نشده‌ی prompt نشود. هر passage با شناسه‌هایی مثل `[S1]` و `[S2]` به مدل داده می‌شود و prompt از مدل می‌خواهد ادعاهای factual را با این شناسه‌ها ارجاع دهد.

## 11. Multimodal
دو مسیر تصویر وجود دارد:

- Image ingestion: تصویر به caption قابل retrieval تبدیل و در knowledge base ایندکس می‌شود.
- Query image: کاربر می‌تواند همراه سؤال یک تصویر بدهد و مدل vision آن را در کنار context متنی retrieved بررسی کند.

این طراحی multimodal از نوع vision-to-text RAG است، نه joint CLIP embedding.

## 12. Evaluation
پروژه دو لایه ارزیابی دارد.

### ارزیابی محلی retrieval
متریک‌های زیر محاسبه می‌شوند:

- Precision@k
- Recall@k
- MRR
- nDCG@k

نتایج برای مراحل Hybrid، MMR و Re-Ranked قابل مقایسه‌اند.

### LangSmith
LangSmith برای tracing و experiment استفاده می‌شود. پروژه golden dataset دارد و امکان اجرای evaluatorهای deterministic و در صورت فعال‌سازی، LLM-as-judge برای موارد زیر را فراهم می‌کند:

- Answer Correctness
- Answer Relevance
- Groundedness
- Retrieval Relevance

## 13. Security
Retrieved context صراحتاً به عنوان **untrusted evidence** در system prompt مشخص می‌شود و مدل نباید دستورهای موجود داخل اسناد را اجرا کند. علاوه بر آن، محدودیت حجم فایل، allow-list پسوند، محدودیت تعداد فایل و شناسایی الگوهای رایج prompt injection وجود دارد.

این کنترل‌ها risk را کاهش می‌دهند اما به تنهایی تضمین امنیت کامل نیستند.

## 14. Testing و Deployment
برای اجزای مهم مانند RRF، BM25 normalization، context budget، persistence، guardrails و retrieval metrics تست واحد وجود دارد. GitHub Actions اجرای تست‌ها را خودکار می‌کند. Dockerfile و healthcheck نیز برای deploy اضافه شده‌اند.

## 15. Pipeline نهایی

```text
Documents / Images
      ↓
Load / Vision Caption
      ↓
Chunking + Stable IDs
      ↓
Embeddings + Persistent Corpus
      ↓
Dense Search + BM25
      ↓
Weighted RRF Hybrid Search
      ↓
MMR
      ↓
Cross-Encoder Re-Ranker
      ↓
Bounded Context + Source IDs
      ↓
LLM
      ↓
Answer + Citations + Diagnostics
```

## 16. نتیجه‌گیری
نسخه نهایی فقط سه روش خواسته‌شده را نمایش نمی‌دهد، بلکه آن‌ها را در یک pipeline واقعی و قابل تست قرار می‌دهد. قابلیت‌های persistence، conversational retrieval، multimodal indexing، evaluation، tracing، security و deployment باعث شده پروژه برای ارائه دانشگاهی و portfolio سطح بالاتری از یک RAG ساده داشته باشد.
