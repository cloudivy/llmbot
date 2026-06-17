import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader
from groq import Groq
import os

st.set_page_config(page_title="Ultra-Fast Groq PDF AI", layout="wide")

# Connect to Groq Engine via Environment Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("⚡ Blazing-Fast Groq PDF Chatbot")
st.caption("Powered by Groq LPU (Llama-3.3-70b-versatile) & FAISS Vector Search")

# Initialize state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar PDF manager
with st.sidebar:
    st.header("📚 Document Repository")
    st.write("Upload PDF files below. They will be indexed locally for retrieval.")

    uploaded_files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Process & Embed Documents", use_container_width=True):
        with st.spinner("Analyzing documents..."):
            all_text_chunks = []
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            for file in uploaded_files:
                pdf_reader = PdfReader(file)
                raw_text = ""
                for page in pdf_reader.pages:
                    text_content = page.extract_text()
                    if text_content:
                        raw_text += text_content + "\n"

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(raw_text)
                all_text_chunks.extend(chunks)

            if all_text_chunks:
                st.session_state.vector_store = FAISS.from_texts(all_text_chunks, embeddings)
                st.success(f"Successfully processed {len(all_text_chunks)} text fragments!")
            else:
                st.error("Could not parse readable text from the provided documents.")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask any question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY missing. Set it as an environment variable before launching.")
        elif st.session_state.vector_store is None:
            st.warning("Please upload and index a PDF using the sidebar before asking questions.")
        else:
            with st.spinner("Streaming response..."):
                try:
                    docs = st.session_state.vector_store.similarity_search(user_input, k=4)
                    context_block = "\n\n---\n\n".join([doc.page_content for doc in docs])

                    system_prompt = (
                        "You are a strict technical assistant. Use ONLY the following provided document "
                        "context to answer the user query.\n"
                        "If the answer cannot be confidently deduced from the context, state that clearly.\n\n"
                        f"Document Context:\n{context_block}"
                    )

                    client = Groq(api_key=GROQ_API_KEY)
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_input},
                        ],
                        temperature=0.2,
                        max_tokens=1024,
                    )

                    ai_response = completion.choices[0].message.content
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})

                except Exception as e:
                    st.error(f"Inference pipeline failure: {str(e)}")
