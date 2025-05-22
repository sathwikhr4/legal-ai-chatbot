# 🧑‍⚖️ AI-Powered Legal Chatbot (India)

A multi-agent chatbot that answers legal questions using Indian law PDFs.

## 📚 Data Sources
- [Guide to Litigation in India (Cyril Amarchand Mangaldas)](https://www.cyrilshroff.com/wp-content/uploads/2020/09/Guide-to-Litigation-in-India.pdf)
- [ICAI Corporate Law Guide](https://kb.icai.org/pdfs/PDFFile5b28c9ce64e524.54675199.pdf)

## 🚀 Features
- Extracts and indexes PDFs
- Uses Sentence-BERT for semantic search
- Summarizes complex legal info using free Hugging Face model
- Streamlit UI for querying

## 🧪 Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py