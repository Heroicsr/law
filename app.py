# app.py — working version using gemini-1.0-pro (compatible with all free-tier API keys)
import streamlit as st
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from docx import Document as DocxDocument
import fitz  # PyMuPDF
import io

# --------------------------------------------------------------------
# SAFE RERUN (NO DEPRECATION ERRORS)
# --------------------------------------------------------------------
def safe_rerun():
    try:
        st.rerun()
    except Exception:
        pass

# --------------------------------------------------------------------
# LOAD ENV + CONFIGURE API KEY
# --------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    st.error("❌ API_KEY not found in .env file.")
    st.stop()

genai.configure(api_key=api_key)

# --------------------------------------------------------------------
# SAFE TEXT EXTRACTION FROM GEMINI RESPONSE
# --------------------------------------------------------------------
def safe_extract_text(response):
    try:
        if hasattr(response, "text") and response.text:
            return response.text

        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text
    except:
        pass
    return "(No content returned)"

# --------------------------------------------------------------------
# LLM CALL — USING GEMINI-1.0-PRO (THE ONLY FULLY SUPPORTED FREE MODEL)
# --------------------------------------------------------------------
def robust_generate(prompt, retries=2):
    model_name = "gemini-1.0-pro"

    model = genai.GenerativeModel(model_name)

    for attempt in range(retries + 1):
        try:
            resp = model.generate_content(prompt)
            txt = safe_extract_text(resp)
            if txt.strip():
                return txt
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
                continue
            return f"(Error: {str(e)})"

    return "(No valid response)"

# --------------------------------------------------------------------
# DOCUMENT READING (PDF / DOCX)
# --------------------------------------------------------------------
def get_document_text(uploaded_files):
    text = ""

    for file in uploaded_files:
        name = file.name.lower()

        try:
            # PDF
            if name.endswith(".pdf"):
                data = file.read()
                pdf = fitz.open(stream=data, filetype="pdf")
                for page in pdf:
                    text += page.get_text()

            # DOCX
            elif name.endswith(".docx"):
                data = io.BytesIO(file.read())
                doc = DocxDocument(data)
                for p in doc.paragraphs:
                    text += p.text + "\n"

            else:
                text += file.read().decode("utf-8", errors="ignore")

        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")

    return text

# --------------------------------------------------------------------
# TEXT SPLITTING + EMBEDDINGS + VECTOR STORE
# --------------------------------------------------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=800
    )
    return splitter.split_text(text)

def get_vector_store(chunks):
    try:
        embed = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        store = FAISS.from_texts(chunks, embed)
        st.session_state.vector_store = store
        st.session_state.raw_text = "\n".join(chunks)
        st.sidebar.success("✅ Document processed.")
        return store

    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

# --------------------------------------------------------------------
# CONVERT RETRIEVED DOCS INTO CONTEXT
# --------------------------------------------------------------------
def docs_to_context(docs, max_chars=12000):
    parts = []
    for d in docs:
        content = getattr(d, "page_content", None) or str(d)
        parts.append(content)

    ctx = "\n\n---\n\n".join(parts)
    return ctx[:max_chars]

# --------------------------------------------------------------------
# DOCUMENT Q&A
# --------------------------------------------------------------------
def handle_document_qna(question):
    store = st.session_state.get("vector_store")

    if not store:
        return "(No documents uploaded.)"

    docs = store.similarity_search(question, k=5)
    context = docs_to_context(docs)

    prompt = f"""
You are ClearClause, a legal assistant.
Use ONLY the context below to answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (accurate, legal tone):
"""
    return robust_generate(prompt)

# --------------------------------------------------------------------
# GENERAL Q&A
# --------------------------------------------------------------------
def handle_general_qna(question):
    return robust_generate(
        f"You are ClearClause, a legal AI system. Answer: {question}"
    )

# --------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------
def generate_summary(instruction):
    raw = st.session_state.get("raw_text")
    if not raw:
        st.error("No text to summarize.")
        return

    chunks = get_text_chunks(raw)
    summaries = []

    for idx, ch in enumerate(chunks):
        prompt = f"Summarize clearly:\n\n{ch}\n\nSummary:"
        summaries.append(robust_generate(prompt))

    combined = "\n\n".join(summaries)

    final = robust_generate(
        f"Refine this summary with instruction '{instruction}':\n\n{combined}"
    )

    st.session_state.chat_history.append(("ClearClause", final))

# --------------------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------------------
st.set_page_config(page_title="ClearClause", layout="wide")
st.title("⚖️ ClearClause – Legal AI Assistant")

# Init session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("📄 Upload Documents")
    files = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)

    if st.button("Process Documents"):
        if files:
            text = get_document_text(files)
            if text.strip():
                chunks = get_text_chunks(text)
                get_vector_store(chunks)
            else:
                st.error("No readable text found.")

    if st.session_state.get("vector_store"):
        st.header("📝 Summary")
        instr = st.text_input("Instruction", "One-paragraph summary")
        if st.button("Generate Summary"):
            generate_summary(instr)

    st.header("⚙️ Chat Settings")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        safe_rerun()

# Chat UI
tab_chat, tab_translate = st.tabs(["💬 Chat", "🌐 Translate"])

with tab_chat:
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    user = st.chat_input("Ask a legal question...")
    if user:
        st.session_state.chat_history.append(("You", user))

        if st.session_state.get("vector_store"):
            ans = handle_document_qna(user)
        else:
            ans = handle_general_qna(user)

        st.session_state.chat_history.append(("ClearClause", ans))
        safe_rerun()

with tab_translate:
    st.subheader("Translate Text")
    txt = st.text_area("Enter English text")
    lang = st.selectbox("Language", ["Hindi", "Marathi", "Telugu"])

    if st.button("Translate"):
        if txt.strip():
            out = robust_generate(
                f"Translate to {lang}: {txt}"
            )
            st.success("Translation:")
            st.text_area("", value=out, height=150)

