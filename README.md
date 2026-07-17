# 🎬 YouTube RAG Chatbot

> **Ask any question about any YouTube video — powered by LangChain, Google Gemini & FAISS**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-latest-1C3C3C?style=flat)
![Gemini](https://img.shields.io/badge/Google_Gemini-API-4285F4?style=flat&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-009688?style=flat)

---

## 📌 Project Overview

**YouTube RAG Chatbot** is an AI-powered web application that lets you have a conversation with any YouTube video. Instead of watching a 1-hour video to find one answer, you paste the URL, and the chatbot answers your questions instantly — using only the content from that video.

It uses a technique called **Retrieval-Augmented Generation (RAG)**, which combines:
- **Document retrieval** — finding the most relevant parts of the transcript
- **LLM generation** — using Gemini AI to generate accurate, context-aware answers

---

<img width="1918" height="865" alt="image" src="https://github.com/user-attachments/assets/a2a1a10e-b804-4792-a5d4-086ded94fb5a" />



## ❓ Problem It Solves

| Problem | Solution |
|---|---|
| Videos are long and hard to search | Instantly answer specific questions from any video |
| No way to "ctrl+F" a YouTube video | Semantic search over the full transcript |
| AI might hallucinate facts | RAG anchors answers **only** to the video transcript |
| Requires watching the whole video | Chat interface extracts what you need in seconds |

**Real-world use cases:**
- 📚 Students summarizing lecture videos
- 🔬 Researchers extracting key findings from conference talks
- 💼 Professionals reviewing long webinars or podcasts
- 🎓 Quick revision before exams from educational YouTube content

---

## 🏗️ How the Project Works

The application follows a **4-step RAG pipeline**:

```
Step 1a: INGEST          Step 1b: SPLIT           Step 1c/d: EMBED & STORE
┌─────────────────┐     ┌─────────────────────┐   ┌──────────────────────┐
│  YouTube URL    │────▶│  YouTube Transcript  │──▶│  Text Chunks         │
│                 │     │  API fetches full    │   │  (1000 chars,        │
│  e.g. youtu.be/│     │  transcript as text  │   │   200 overlap)       │
│  LPZh9BOjkQs   │     └─────────────────────┘   └──────────┬───────────┘
└─────────────────┘                                           │
                                                              ▼
                                                   ┌──────────────────────┐
                                                   │  Gemini Embeddings   │
                                                   │  (vector numbers)    │
                                                   └──────────┬───────────┘
                                                              │
                                                              ▼
                                                   ┌──────────────────────┐
                                                   │  FAISS Vector Store  │
                                                   │  (in-memory index)   │
                                                   └──────────────────────┘

Step 2: RETRIEVE          Step 3: AUGMENT          Step 4: GENERATE
┌─────────────────┐     ┌─────────────────────┐   ┌──────────────────────┐
│  User Question  │────▶│  Top-4 Relevant      │──▶│  Gemini 2.5 Flash   │
│                 │     │  Chunks retrieved    │   │  generates answer    │
│  "What is an   │     │  by similarity       │   │  from context only   │
│   LLM?"        │     │  search in FAISS     │   │                      │
└─────────────────┘     └─────────────────────┘   └──────────────────────┘
```

### LangChain Chain Architecture

```
User Question
     │
     ▼
┌────────────────────────────────────────┐
│         RunnableParallel               │
│  ┌──────────────┐  ┌────────────────┐  │
│  │  Retriever   │  │ RunnablePass   │  │
│  │  (FAISS k=4) │  │ through        │  │
│  └──────┬───────┘  └───────┬────────┘  │
│         │ context          │ question   │
└─────────┼──────────────────┼────────────┘
          │                  │
          ▼                  ▼
     ┌────────────────────────────┐
     │      PromptTemplate        │
     │  "Answer from context..."  │
     └────────────┬───────────────┘
                  │
                  ▼
     ┌────────────────────────────┐
     │   ChatGoogleGenerativeAI   │
     │   (gemini-2.5-flash)       │
     └────────────┬───────────────┘
                  │
                  ▼
     ┌────────────────────────────┐
     │      StrOutputParser       │
     │   (plain text answer)      │
     └────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | [Streamlit](https://streamlit.io/) | Web interface with dark theme |
| **Orchestration** | [LangChain](https://www.langchain.com/) | RAG pipeline & chain management |
| **LLM** | [Google Gemini 2.5 Flash](https://ai.google.dev/) | Answer generation |
| **Embeddings** | Google Gemini Embedding | Convert text to vectors |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) | Fast similarity search |
| **Transcript** | [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) | Fetch YouTube subtitles |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` | Chunk transcript into pieces |
| **Environment** | `python-dotenv` | Secure API key management |

---

## 📁 Project Structure

```
YT_Chat/
│
├── app.py                    # Streamlit UI (main entry point)
├── rag_engine.py             # RAG pipeline logic (extracted from notebook)
├── rag_using_langchain.ipynb # Original Jupyter notebook (learning reference)
│
├── .streamlit/
│   └── config.toml           # Streamlit dark theme config
│
├── .env                      # API keys (NOT committed to GitHub)
├── .gitignore                # Git ignore rules
├── requirements.txt          # All Python dependencies
│
└── venv/                     # Virtual environment (NOT committed)
```

---

## ⚙️ Setup Guide (For Students)

### Prerequisites
- Python 3.10 or higher installed
- A Google AI Studio API key (free at [aistudio.google.com](https://aistudio.google.com/))
- Git installed

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/sanjit507/Youtube-RAG-Chatbot.git
cd Youtube-RAG-Chatbot
```

---

### Step 2 — Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal line.

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ This may take 2–3 minutes. FAISS and LangChain have several sub-dependencies.

---

### Step 4 — Set Up Your API Key

Create a file called `.env` in the project root:

```bash
# Windows
echo GOOGLE_API_KEY=your_api_key_here > .env

# macOS/Linux
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

Or create it manually — create a new file named `.env` and add:

```
GOOGLE_API_KEY=AIzaSy...your_key_here
```

> 🔑 Get your free API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

### Step 5 — Run the App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

### Step 6 — Use the App

1. Paste any YouTube URL in the input box  
   e.g. `https://www.youtube.com/watch?v=LPZh9BOjkQs`
2. Click **🚀 Load** and wait ~10-30 seconds for indexing
3. Type your question and click **💬 Ask**
4. Get AI-powered answers based only on the video content!

---

## 🧪 Example Questions to Try

Use this video: `https://www.youtube.com/watch?v=LPZh9BOjkQs` (3Blue1Brown — How LLMs work)

| Question | Expected Answer Type |
|---|---|
| What is a large language model? | Definition from the video |
| How does training work? | Step-by-step explanation |
| What is backpropagation? | Concept from the video |
| Summarize this video | Full summary |
| What is the transformer architecture? | Technical explanation |

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `TranscriptsDisabled` | Video has no subtitles — try another video |
| `Invalid API Key` | Check your `.env` file and API key |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| White input boxes | Make sure `.streamlit/config.toml` exists |
| Port already in use | Run `streamlit run app.py --server.port 8502` |

---

## 📖 Learning Resources

- [LangChain RAG Documentation](https://python.langchain.com/docs/tutorials/rag/)
- [Google Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [FAISS Documentation](https://faiss.ai/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)

---

## 👤 Author

**Sanjit** — [GitHub: sanjit507](https://github.com/sanjit507)

---

> **Note:** This project is for educational purposes. Ensure you comply with YouTube's Terms of Service when using transcript data.
