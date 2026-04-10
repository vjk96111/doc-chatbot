# DocChat — AI Document Assistant

Upload PDF or Word documents and chat with them using Google Gemini AI.  
**100% free** to run locally and to host online via Streamlit Community Cloud.

---

## Features

| Feature | Details |
|---|---|
| 📄 Document support | PDF and Word (.docx) files |
| 🤖 AI engine | Google Gemini 1.5 Flash (free tier) |
| 🔍 Semantic search | FAISS vector index (NumPy fallback) |
| 🖼️ Diagram display | Images extracted from documents shown inline |
| 📚 Source citations | Every answer cites document + page/section |
| 📋 Summarization | One-click summary of any loaded document |
| 💬 Multi-doc Q&A | Ask across multiple uploaded documents at once |
| 💾 Export chat | Download the full conversation as a .txt file |

---

## Quick start (local)

### 1  Prerequisites
- Python 3.9 or newer
- A free Google Gemini API key → [Get one here](https://aistudio.google.com/app/apikey)

### 2  Install dependencies

```bash
cd doc-chatbot
pip install -r requirements.txt
```

> **Note:** `faiss-cpu` occasionally has build issues on some platforms.  
> If installation fails, remove it from `requirements.txt` — the app will fall  
> back to a pure-NumPy cosine similarity implementation automatically.

### 3  (Optional) Set API key via environment variable

```bash
# Copy template and fill in your key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=AIza...
```

The app also accepts the key typed directly in the sidebar, so this step is optional.

### 4  Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deploy for free on Streamlit Community Cloud

**Total time: ~5 minutes.**

1. **Push this folder to a public GitHub repository.**

2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.

3. Click **"New app"** → select your repo → set **Main file path** to `doc-chatbot/app.py`  
   *(or `app.py` if the repo root is the `doc-chatbot` folder).*

4. Under **Advanced settings → Secrets**, add your API key:

   ```toml
   GEMINI_API_KEY = "AIza..."
   ```

5. Click **Deploy**. Your app gets a permanent public URL like  
   `https://your-app-name.streamlit.app` — shareable with anyone.

> **Free tier limits (Streamlit Community Cloud):**  
> 1 GB RAM · 1 CPU · public apps only. Sufficient for most documents.

---

## Google Gemini free tier limits

| Resource | Free limit |
|---|---|
| `gemini-1.5-flash` requests | 15 RPM · 1 000 000 TPD |
| `text-embedding-004` requests | 1 500 requests / day · 15 RPM |
| Context window | 1 M tokens |

For a typical document (50 pages ≈ 100 text chunks), processing uses ~100 embedding requests.  
Each chat question uses 1 embedding + 1 generation request.

---

## Project structure

```
doc-chatbot/
├── app.py                      Main Streamlit application
├── requirements.txt            Python dependencies
├── .env.example                Environment variable template
├── .gitignore
├── README.md
└── utils/
    ├── __init__.py
    ├── document_processor.py   PDF/DOCX text + image extraction
    ├── vector_store.py         FAISS / NumPy vector store
    └── llm_handler.py          Gemini Q&A + summarization
```

---

## How it works

```
Upload document
      │
      ▼
Extract text (per page/section) + images (per page)
      │
      ▼
Chunk text (400-word chunks, 50-word overlap)
      │
      ▼
Embed each chunk → store in FAISS index (in RAM)
      │
 User asks question
      │
      ▼
Embed question → FAISS top-5 similarity search
      │
      ▼
Build prompt: retrieved chunks + question
      │
      ▼
Gemini generates answer with source citations
      │
      ▼
Display answer + images from matching pages
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `faiss-cpu` install fails | Remove it from requirements.txt; NumPy fallback activates automatically |
| Rate limit error (429) | Wait 60 s; free tier = 15 RPM. Large documents embed slowly. |
| No text extracted from PDF | The PDF may be scanned/image-only. OCR (e.g. Tesseract) is a planned future feature. |
| Images not showing | Images smaller than 5 KB are filtered out as decorative. Only embedded raster images are extracted. |
| `ModuleNotFoundError: fitz` | Run `pip install PyMuPDF` |

---

## Roadmap (suggested future features)

- [ ] Table extraction — render document tables as markdown
- [ ] OCR for scanned PDFs via Tesseract
- [ ] Persistent knowledge base (disk-backed FAISS index)
- [ ] Document comparison mode
- [ ] Highlighted source excerpts in answers
- [ ] Voice input via `st.audio_input`
- [ ] Multi-language answer support
- [ ] PDF page image preview panel
- [ ] Custom system prompt / persona field

---

## License

MIT — free to use, modify, and deploy.
