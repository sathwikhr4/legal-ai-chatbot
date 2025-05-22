import streamlit as st
import requests
import PyPDF2
from io import BytesIO
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss

st.set_page_config(page_title="Legal AI Chatbot", layout="centered")

@st.cache_data
def load_pdf_text(url):
    response = requests.get(url)
    pdf_file = BytesIO(response.content)
    reader = PyPDF2.PdfReader(pdf_file)
    return " ".join([page.extract_text() or "" for page in reader.pages])

@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def chunk_text(text, max_len=500):
    words = text.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

# Load documents
litigation_url = "https://www.cyrilshroff.com/wp-content/uploads/2020/09/Guide-to-Litigation-in-India.pdf"
compliance_url = "https://kb.icai.org/pdfs/PDFFile5b28c9ce64e524.54675199.pdf"

litigation_text = load_pdf_text(litigation_url)
compliance_text = load_pdf_text(compliance_url)
all_text = litigation_text + "\n\n" + compliance_text

# Chunking and Embeddings
st.info("🔍 Processing documents... please wait.")
chunks = chunk_text(all_text)
tokenizer, model = load_embedding_model()
embeddings = np.array([get_embedding(c, tokenizer, model) for c in chunks])

# FAISS Search
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

# UI
st.title("🧑‍⚖️ Legal AI Chatbot")
query = st.text_input("Ask your legal question")

if query:
    query_embed = get_embedding(query, tokenizer, model).reshape(1, -1)
    _, I = index.search(query_embed, 3)
    top_chunks = [chunks[i] for i in I[0]]

    joined = " ".join(top_chunks)[:3000]
    summary = summarizer(joined, max_length=200, min_length=60, do_sample=False)[0]["summary_text"]

    st.subheader("📌 Answer")
    st.write(summary)

    with st.expander("📚 Source Snippets"):
        for i, chunk in enumerate(top_chunks):
            st.markdown(f"**Snippet {i+1}**\n\n{chunk[:400]}...")
