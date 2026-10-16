"""Translations (English / Persian), RTL helpers and text-direction detection.
"""

import streamlit as st


# --------------------------------------------------------------------------
# Internationalization / RTL support
# --------------------------------------------------------------------------
# Persian (fa) is fully RTL. Rather than bolting translated strings on as an
# afterthought, every user-facing label used in the UI functions below is
# looked up through t(key) so the whole app (nav, buttons, empty states,
# hero banners) switches language together with the text direction.
RTL_LANGS = {"fa"}

TRANSLATIONS = {
    "en": {
        "app_title": "Document Q&A",
        "app_tagline": "Chat with your local files, privately.",
        "offline_badge": "Offline mode — no internet access, no downloads",
        "online_badge": "Online mode — internet access and model downloads allowed",
        "network_label": "Network",
        "net_offline": "Offline",
        "net_online": "Online",
        "network_help": "Offline blocks all internet access and model downloads (local/LAN servers still work). Online allows HuggingFace model downloads and remote OpenAI-compatible APIs. Your choice is remembered.",
        "nav_chat": "💬 Chat",
        "nav_documents": "📄 Documents",
        "nav_settings": "⚙️ Settings",
        "nav_status": "📊 Status",
        "overview": "Overview",
        "indexed_chunks": "Indexed Chunks",
        "documents_size": "Documents Size",
        "llm_status": "LLM Status",
        "api_status_label": "API Status",
        "api_not_used": "Not used",
        "api_no_key": "No key",
        "api_blocked": "Blocked (offline)",
        "api_not_ready": "Not initialized",
        "api_ready": "Key set (untested)",
        "api_connected": "Connected",
        "api_failed": "Failed",
        "loaded": "✅ Loaded",
        "not_loaded": "❌ Not Loaded",
        "language_label": "Language",
        "chat_hero_title": "💬 Chat with Your Documents",
        "chat_hero_sub": "Ask questions and get grounded answers with sources, all running locally.",
        "no_model_title": "No model loaded yet",
        "no_model_sub": "Load your local LLM to start chatting with your documents.",
        "load_default_model": "🔄 Load Default Model",
        "llm_loaded_success": "LLM loaded successfully!",
        "llm_load_failed": "Failed to load LLM. Check the GGUF model path or the OpenAI API key in Settings.",
        "no_docs_title": "No documents indexed",
        "no_docs_sub": "Head to the Documents tab to upload files before asking questions.",
        "ask_first_title": "Ask your first question",
        "ask_first_sub": 'Try something like "Summarize the key points of my documents."',
        "chat_placeholder": "Ask a question about your documents...",
        "thinking": "Thinking...",
        "sources": "📖 Sources",
        "retrieval_time_label": "Retrieval",
        "generation_time_label": "Generation",
        "cached_answer_label": "Instant — answered from cache",
        "cloud_answer_notice": "Answered via a remote OpenAI-compatible API — your question and the retrieved excerpts were sent to it.",
        "local_endpoint_notice": "Answered via a local/LAN OpenAI-compatible server — nothing left your network.",
        "clear_chat": "🗑️ Clear Chat History",
        "export_chat": "⬇️ Export Chat (Markdown)",
        "scope_title": "Documents used for answers",
        "scope_all": "All documents",
        "scope_selected": "Selected documents",
        "scope_pick": "Choose documents",
        "scope_none_selected": "Select at least one document to ask questions.",
        "scope_active_all": "Using all {n} documents",
        "scope_active_selected": "Using {m} of {n} documents",
        "docs_hero_title": "📄 Document Management",
        "docs_hero_sub": "Upload, browse, and manage the files powering your knowledge base.",
        "tab_upload": "📤 Upload",
        "tab_library": "📚 Library",
        "tab_bulk": "🗂️ Bulk Folder Import",
        "tab_danger": "⚠️ Danger Zone",
        "bulk_intro": "For large corpora (multi-GB, tens of GB), point the app at a folder on disk instead of uploading through the browser. Indexing runs in the background, one file at a time, and is safe to stop and resume — already-indexed files are skipped on restart.",
        "bulk_folder_label": "Folder path (on the machine running this app)",
        "bulk_start": "▶️ Start / Resume Indexing",
        "bulk_stop": "⏹️ Stop After Current File",
        "bulk_refresh": "🔄 Refresh Status",
        "bulk_status_idle": "Not started yet.",
        "bulk_status_running": "Indexing in progress…",
        "bulk_status_stopped": "Stopped — resume anytime, already-indexed files will be skipped.",
        "bulk_status_done": "Indexing complete.",
        "bulk_status_error": "Indexing hit an error.",
        "bulk_files_done": "Files Done",
        "bulk_files_total": "Files Found",
        "bulk_chunks": "Chunks Indexed",
        "bulk_errors": "Errors",
        "bulk_current_file": "Currently processing",
        "bulk_view_errors": "Show failed files",
        "bulk_folder_missing": "That folder doesn't exist or isn't accessible to the app.",
        "supported": "Supported",
        "no_loaders": "No document loaders are available — check the import errors at the top of the page.",
        "upload_drop": "Drag and drop files here, or browse",
        "files_selected": "Files Selected",
        "total_size": "Total Size",
        "add_to_kb": "➕ Add to Knowledge Base",
        "processing_docs": "Processing documents...",
        "no_docs_yet_title": "No documents yet",
        "no_docs_yet_sub": "Switch to the Upload tab to add your first file.",
        "filter_filename": "🔍 Filter by filename",
        "no_match_filter": "No documents match that filter.",
        "danger_warning": "These actions permanently delete data and cannot be undone.",
        "clear_all": "🗑️ Clear All Documents and Index",
        "confirm_delete_all_msg": "This will permanently delete every document and the entire vector index.",
        "confirm_delete_all_btn": "⚠️ Confirm Delete All",
        "cancel": "Cancel",
        "kb_cleared": "Knowledge base cleared!",
        "kb_clear_failed": "Failed to clear knowledge base.",
        "settings_hero_title": "⚙️ Settings",
        "settings_hero_sub": "Configure the local model, embeddings, and chunking behavior.",
        "tab_llm": "🧠 LLM",
        "tab_retrieval": "🔍 Retrieval",
        "tab_embed": "🔢 Embeddings & Chunking",
        "tab_storage": "💾 Storage",
        "tab_language": "🌐 Language",
        "tab_chat": "💬 Chat & Cache",
        "tab_appearance": "🎨 Appearance",
        "theme_label": "Theme",
        "theme_dark": "🌙 Dark",
        "theme_light": "☀️ Light",
        "status_hero_title": "📊 System Status",
        "status_hero_sub": "A quick health check of your knowledge base and machine.",
        "doc_stats": "📈 Document Statistics",
        "no_stats_title": "No statistics yet",
        "no_stats_sub": "Add documents to see a breakdown by file type.",
        "system_info": "🖥️ System Information",
        "no_psutil_title": "psutil not installed",
        "no_psutil_sub": "Install 'psutil' to see live RAM usage.",
        "total_documents_size": "Total Documents Size",
    },
    "fa": {
        "app_title": "پرسش و پاسخ اسناد",
        "app_tagline": "به‌صورت خصوصی با فایل‌های محلی خود گفتگو کنید.",
        "offline_badge": "حالت آفلاین — بدون اینترنت و بدون دانلود",
        "online_badge": "حالت آنلاین — دسترسی به اینترنت و دانلود مدل مجاز است",
        "network_label": "شبکه",
        "net_offline": "آفلاین",
        "net_online": "آنلاین",
        "network_help": "حالت آفلاین هر دسترسی به اینترنت و دانلود مدل را مسدود می‌کند (سرورهای محلی/شبکهٔ داخلی همچنان کار می‌کنند). حالت آنلاین دانلود مدل از HuggingFace و APIهای راه‌دور سازگار با OpenAI را مجاز می‌کند. انتخاب شما به خاطر سپرده می‌شود.",
        "nav_chat": "💬 گفتگو",
        "nav_documents": "📄 اسناد",
        "nav_settings": "⚙️ تنظیمات",
        "nav_status": "📊 وضعیت",
        "overview": "نمای کلی",
        "indexed_chunks": "قطعات نمایه‌شده",
        "documents_size": "حجم اسناد",
        "llm_status": "وضعیت مدل",
        "api_status_label": "وضعیت API",
        "api_not_used": "استفاده نمی‌شود",
        "api_no_key": "بدون کلید",
        "api_blocked": "مسدود (آفلاین)",
        "api_not_ready": "راه‌اندازی نشده",
        "api_ready": "کلید تنظیم شده (آزمایش نشده)",
        "api_connected": "متصل",
        "api_failed": "ناموفق",
        "loaded": "✅ بارگذاری شد",
        "not_loaded": "❌ بارگذاری نشده",
        "language_label": "زبان",
        "chat_hero_title": "💬 گفتگو با اسناد شما",
        "chat_hero_sub": "پرسش بپرسید و پاسخ‌های مستند با منبع دریافت کنید، همه به‌صورت محلی.",
        "no_model_title": "هنوز مدلی بارگذاری نشده",
        "no_model_sub": "برای شروع گفتگو با اسناد خود، مدل زبانی محلی را بارگذاری کنید.",
        "load_default_model": "🔄 بارگذاری مدل پیش‌فرض",
        "llm_loaded_success": "مدل با موفقیت بارگذاری شد!",
        "llm_load_failed": "بارگذاری مدل ناموفق بود. مسیر مدل GGUF یا کلید OpenAI را در تنظیمات بررسی کنید.",
        "no_docs_title": "هیچ سندی نمایه نشده است",
        "no_docs_sub": "پیش از پرسیدن سؤال، از بخش اسناد فایل‌ها را بارگذاری کنید.",
        "ask_first_title": "اولین پرسش خود را بپرسید",
        "ask_first_sub": 'مثلاً بنویسید: «نکات کلیدی اسناد من را خلاصه کن.»',
        "chat_placeholder": "دربارهٔ اسناد خود سؤال بپرسید...",
        "thinking": "در حال فکر کردن...",
        "sources": "📖 منابع",
        "retrieval_time_label": "بازیابی",
        "generation_time_label": "تولید پاسخ",
        "cached_answer_label": "فوری — پاسخ از حافظهٔ نهان",
        "cloud_answer_notice": "پاسخ از طریق یک API راه‌دور سازگار با OpenAI تولید شد — پرسش شما و بخش‌های بازیابی‌شده به آن ارسال شدند.",
        "local_endpoint_notice": "پاسخ از طریق یک سرور محلی/شبکهٔ داخلی سازگار با OpenAI تولید شد — چیزی از شبکهٔ شما خارج نشد.",
        "clear_chat": "🗑️ پاک‌کردن تاریخچهٔ گفتگو",
        "export_chat": "⬇️ خروجی گفتگو (Markdown)",
        "scope_title": "اسناد مورد استفاده برای پاسخ",
        "scope_all": "همهٔ اسناد",
        "scope_selected": "اسناد انتخاب‌شده",
        "scope_pick": "انتخاب اسناد",
        "scope_none_selected": "برای پرسیدن سؤال، حداقل یک سند انتخاب کنید.",
        "scope_active_all": "استفاده از همهٔ {n} سند",
        "scope_active_selected": "استفاده از {m} سند از {n} سند",
        "docs_hero_title": "📄 مدیریت اسناد",
        "docs_hero_sub": "فایل‌های تشکیل‌دهندهٔ پایگاه دانش خود را بارگذاری، مرور و مدیریت کنید.",
        "tab_upload": "📤 بارگذاری",
        "tab_library": "📚 کتابخانه",
        "tab_bulk": "🗂️ ورود انبوه از پوشه",
        "tab_danger": "⚠️ منطقهٔ خطر",
        "bulk_intro": "برای مجموعه‌اسناد بزرگ (چند گیگابایت تا چند ده گیگابایت)، به‌جای بارگذاری از مرورگر، مسیر یک پوشه روی دیسک را وارد کنید. نمایه‌سازی در پس‌زمینه و فایل‌به‌فایل انجام می‌شود و می‌توانید هر زمان آن را متوقف و دوباره ازسرگیری کنید — فایل‌های قبلاً نمایه‌شده دوباره پردازش نمی‌شوند.",
        "bulk_folder_label": "مسیر پوشه (روی همان دستگاهی که این برنامه در حال اجراست)",
        "bulk_start": "▶️ شروع / ازسرگیری نمایه‌سازی",
        "bulk_stop": "⏹️ توقف پس از فایل جاری",
        "bulk_refresh": "🔄 به‌روزرسانی وضعیت",
        "bulk_status_idle": "هنوز شروع نشده است.",
        "bulk_status_running": "نمایه‌سازی در حال انجام است…",
        "bulk_status_stopped": "متوقف شد — هر زمان می‌توانید ازسرگیری کنید؛ فایل‌های نمایه‌شده دوباره پردازش نمی‌شوند.",
        "bulk_status_done": "نمایه‌سازی کامل شد.",
        "bulk_status_error": "نمایه‌سازی با خطا مواجه شد.",
        "bulk_files_done": "فایل‌های انجام‌شده",
        "bulk_files_total": "فایل‌های یافت‌شده",
        "bulk_chunks": "قطعات نمایه‌شده",
        "bulk_errors": "خطاها",
        "bulk_current_file": "در حال پردازش",
        "bulk_view_errors": "نمایش فایل‌های ناموفق",
        "bulk_folder_missing": "این پوشه وجود ندارد یا برای برنامه در دسترس نیست.",
        "supported": "پشتیبانی‌شده",
        "no_loaders": "هیچ بارگذار سندی در دسترس نیست — خطاهای وارد کردن را در بالای صفحه بررسی کنید.",
        "upload_drop": "فایل‌ها را اینجا بکشید و رها کنید، یا انتخاب کنید",
        "files_selected": "فایل‌های انتخاب‌شده",
        "total_size": "حجم کل",
        "add_to_kb": "➕ افزودن به پایگاه دانش",
        "processing_docs": "در حال پردازش اسناد...",
        "no_docs_yet_title": "هنوز سندی وجود ندارد",
        "no_docs_yet_sub": "برای افزودن اولین فایل خود به تب بارگذاری بروید.",
        "filter_filename": "🔍 فیلتر بر اساس نام فایل",
        "no_match_filter": "هیچ سندی با این فیلتر مطابقت ندارد.",
        "danger_warning": "این اقدامات داده‌ها را برای همیشه حذف می‌کنند و غیرقابل بازگشت هستند.",
        "clear_all": "🗑️ پاک‌کردن همهٔ اسناد و نمایه",
        "confirm_delete_all_msg": "این کار همهٔ اسناد و کل نمایهٔ برداری را برای همیشه حذف می‌کند.",
        "confirm_delete_all_btn": "⚠️ تأیید حذف همه",
        "cancel": "لغو",
        "kb_cleared": "پایگاه دانش پاک شد!",
        "kb_clear_failed": "پاک‌کردن پایگاه دانش ناموفق بود.",
        "settings_hero_title": "⚙️ تنظیمات",
        "settings_hero_sub": "مدل محلی، امبدینگ‌ها و رفتار تکه‌بندی را پیکربندی کنید.",
        "tab_llm": "🧠 مدل زبانی",
        "tab_retrieval": "🔍 بازیابی",
        "tab_embed": "🔢 امبدینگ و تکه‌بندی",
        "tab_storage": "💾 ذخیره‌سازی",
        "tab_language": "🌐 زبان",
        "tab_chat": "💬 گفتگو و حافظهٔ نهان",
        "tab_appearance": "🎨 ظاهر",
        "theme_label": "پوسته",
        "theme_dark": "🌙 تیره",
        "theme_light": "☀️ روشن",
        "status_hero_title": "📊 وضعیت سیستم",
        "status_hero_sub": "بررسی سریع سلامت پایگاه دانش و دستگاه شما.",
        "doc_stats": "📈 آمار اسناد",
        "no_stats_title": "هنوز آماری وجود ندارد",
        "no_stats_sub": "برای مشاهدهٔ تفکیک بر اساس نوع فایل، سند اضافه کنید.",
        "system_info": "🖥️ اطلاعات سیستم",
        "no_psutil_title": "psutil نصب نشده است",
        "no_psutil_sub": "برای مشاهدهٔ مصرف زندهٔ RAM، «psutil» را نصب کنید.",
        "total_documents_size": "حجم کل اسناد",
    },
}


def t(key: str) -> str:
    """Translate a UI string key into the currently active language,
    falling back to English if the key or language is missing."""
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))


def current_lang() -> str:
    return st.session_state.get("language", "en")


def is_rtl_lang(lang: str) -> bool:
    return lang in RTL_LANGS


_PERSIAN_ARABIC_RANGES = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


def detect_text_direction(text: str) -> str:
    """Heuristically detect whether a piece of text (e.g. a chat message or
    a document chunk) is predominantly RTL (Persian/Arabic script) or LTR,
    so mixed-language chat history can render each bubble with the correct
    alignment instead of forcing one direction on everything."""
    if not text:
        return "ltr"
    rtl_count = 0
    ltr_count = 0
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in _PERSIAN_ARABIC_RANGES):
            rtl_count += 1
        elif ch.isalpha():
            ltr_count += 1
    if rtl_count == 0 and ltr_count == 0:
        return "ltr"
    return "rtl" if rtl_count >= ltr_count else "ltr"
