import streamlit as st
import requests
import PyPDF2
from io import BytesIO
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss

# === Load and preprocess PDFs ===

@st.cache_data
def load_pdf_text(url):
    response = requests.get(url)
    pdf_file = BytesIO(response.content)
    reader = PyPDF2.PdfReader(pdf_file)
    text = " ".join([page.extract_text() or "" for page in reader.pages])
    return text

litigation_url = "https://www.cyrilshroff.com/wp-content/uploads/2020/09/Guide-to-Litigation-in-India.pdf"
compliance_url = "https://kb.icai.org/pdfs/PDFFile5b28c9ce64e524.54675199.pdf"

litigation_text = load_pdf_text(litigation_url)
compliance_text = load_pdf_text(compliance_url)

# === Load embedding model ===

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# === Chunk documents ===

def chunk_text(text, max_len=500):
    words = text.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

litigation_chunks = chunk_text(litigation_text)
compliance_chunks = chunk_text(compliance_text)

all_chunks = litigation_chunks + compliance_chunks
chunk_embeddings = np.array([get_embedding(chunk) for chunk in all_chunks])

# === Build FAISS index ===

index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(chunk_embeddings)

# === Summarizer ===

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

# === Streamlit UI ===

st.title("🧑‍⚖️ Indian Legal AI Chatbot")
st.write("Ask legal questions from two official documents:")

user_query = st.text_input("Enter your legal question:")

if user_query:
    # Step 1: Embed user query
    query_embedding = get_embedding(user_query).reshape(1, -1)

    # Step 2: Search FAISS index
    D, I = index.search(query_embedding, k=3)
    top_chunks = [all_chunks[i] for i in I[0]]

    # Step 3: Summarize top chunks
    summary_input = " ".join(top_chunks)[:3000]  # summarizer input limit
    summary = summarizer(summary_input, max_length=200, min_length=60, do_sample=False)[0]['summary_text']

    # Step 4: Respond
    st.subheader("📌 Answer")
    st.write(summary)

    with st.expander("🔎 Sources"):
        for i, chunk in enumerate(top_chunks):
            st.markdown(f"**Chunk {i+1}:** {chunk[:500]}...")
