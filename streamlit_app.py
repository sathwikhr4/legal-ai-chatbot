import streamlit as st
import requests
import pdfplumber
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import os

PDF_URLS = [
    "https://www.cyrilshroff.com/wp-content/uploads/2020/09/Guide-to-Litigation-in-India.pdf",
    "https://kb.icai.org/pdfs/PDFFile5b28c9ce64e524.54675199.pdf"
]

def download_pdfs():
    pdf_texts = []
    for url in PDF_URLS:
        response = requests.get(url)
        filename = url.split("/")[-1]
        with open(filename, "wb") as f:
            f.write(response.content)
        pdf_texts.append(extract_text_from_pdf(filename))
        os.remove(filename)
    return "\n".join(pdf_texts)

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

from sentence_transformers import SentenceTransformer
import torch

# Force CPU device
device = "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def build_index(text_chunks):
    embeddings = model.encode(text_chunks, convert_to_tensor=False)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings, text_chunks

def search_index(index, chunks, embeddings, query, top_k=3):
    query_embedding = model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)
    return [chunks[i] for i in I[0]]

summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY", device=-1)

def summarize_text(texts):
    summaries = []
    for text in texts:
        input_text = text[:1000] if len(text) > 1000 else text
        summary = summarizer(input_text)[0]['summary_text']
        summaries.append(summary)
    return summaries

# Streamlit Interface
st.title("🇮🇳 Legal Assistant Chatbot (Free & Online)")
st.markdown("Ask any question from:\n- Guide to Litigation in India\n- ICAI Corporate Law Guide")

if st.button("Load PDFs & Initialize Chatbot"):
    with st.spinner("Downloading and processing legal documents..."):
        full_text = download_pdfs()
        chunks = full_text.split("\n\n")
        index, embeddings, stored_chunks = build_index(chunks)
        st.session_state.index = index
        st.session_state.embeddings = embeddings
        st.session_state.chunks = stored_chunks
    st.success("Chatbot ready! You can now ask legal questions.")

query = st.text_input("Ask a legal question here:")
if query and "index" in st.session_state:
    results = search_index(st.session_state.index, st.session_state.chunks, st.session_state.embeddings, query)
    summary = summarize_text(results)
    st.markdown("**Answer:**")
    for s in summary:
        st.success(s)
elif query:
    st.warning("Please click 'Load PDFs' first.")
