"""
RAG Engine – YouTube transcript Q&A using LangChain + Gemini
Extracted from rag_using_langchain.ipynb
"""

import re
import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&?#]|$)",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"embed\/([0-9A-Za-z_-]{11})",
        r"shorts\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # If it looks like a raw video ID (11 chars)
    if re.match(r"^[0-9A-Za-z_-]{11}$", url.strip()):
        return url.strip()
    raise ValueError(f"Could not extract a valid YouTube video ID from: {url!r}")


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = PromptTemplate(
    template="""
You are a helpful assistant that answers questions based strictly on a YouTube video transcript.
Answer ONLY from the provided transcript context below.
If the context does not contain enough information to answer the question, say "I don't know based on the video content."

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"],
)


def build_rag_chain(video_id: str):
    """
    Fetch transcript for *video_id*, build FAISS vector store and return
    (chain, transcript_text, num_chunks).
    """
    # Step 1a – Transcript ingestion
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    text = " ".join(item.text for item in transcript)

    # Step 1b – Text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    # Step 1c/d – Embeddings + Vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Step 2 – Retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Step 3 – LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    # Step 4 – Chain
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    parser = StrOutputParser()
    chain = parallel_chain | PROMPT_TEMPLATE | llm | parser

    return chain, text, len(chunks)
