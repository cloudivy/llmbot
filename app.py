import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pypdf import PyPDFReader
from groq import Groq
import os

st.set_page_config(page_title="Ultra-Fast Groq PDF AI", layout="wide")

# Connect to Groq Engine via Environment Secrets 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("⚡ Blazing-Fast Groq PDF Chatbot")
st.caption("Powered by Groq LPU Hardware (Llama-3.3-70b-versatile) & FAISS Vector Layers")

# Initialize state memory structures
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar PDF Control Manager
with st.sidebar:
    st.header("📚 Document Repository")
    st.write("Upload PDF files below. They will be indexed locally for prompt injection.")
    
    uploaded_files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.button("Process & Embed Documents", use_container_width=True):
        with st.spinner("Analyzing text tracks..."):
            all_text_chunks = []
            
            # Use an efficient open-source embedding model running on your free web node CPU
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            for file in uploaded_files:
                pdf_reader = PyPDFReader(file)
                raw_text = ""
                for page in pdf_reader.pages:
                    text_content = page.extract_text()
                    if text_content:
                        raw_text += text_content + "\n"
                
                # Split raw texts into meaningful passages
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(raw_text)
                all_text_chunks.extend(chunks)
            
            if all_text_chunks:
                # Commit processed fragments into our runtime search matrix
                st.session_state.vector_store = FAISS.from_texts(all_text_chunks, embeddings)
                st.success(f"Successfully processed {len(all_text_chunks)} text fragments!")
            else:
                st.error("Could not parse readable text tracks from the provided documents.")

# Print conversational historical tracks inside client view template
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input Prompt
if user_input := st.chat_input("Ask any analytical question about your document layers..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        if not GROQ_API_KEY:
            st.error("Groq API token signature missing. Please configure GROQ_API_KEY inside workspace settings.")
        elif st.session_state.vector_store is None:
            st.warning("Please upload and index a PDF using the sidebar panel before asking questions.")
        else:
            with st.spinner("Streaming response..."):
                try:
                    # Look up relevant facts out of our local database matrix
                    docs = st.session_state.vector_store.similarity_search(user_input, k=4)
                    context_block = "\n\n---\n\n".join([doc.page_content for doc in docs])
                    
                    # Formulate structured payload parameters
                    system_prompt = (
                        "You are a strict technical assistant. Use ONLY the following provided document context blocks to answer the user query.\n"
                        "If the answer cannot be confidently deduced from the provided context, state that clearly.\n\n"
                        f"Document Context:\n{context_block}"
                    )
                    
                    client = Groq(api_key=GROQ_API_KEY)
                    
                    # Execute high-throughput model call with Llama 3.3 70B
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_input}
                        ],
                        temperature=0.2,
                        max_tokens=1024,
                    )
                    
                    ai_response = completion.choices[0].message.content
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                except Exception as e:
                    st.error(f"Inference pipeline failure execution log: {str(e)}")