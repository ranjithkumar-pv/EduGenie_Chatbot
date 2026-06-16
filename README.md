# 📘 EduGenie – AI-Powered Educational Assistant
*A Document-aware RAG Chatbot using Groq, LangChain, ChromaDB & Flask*

## 🚀 Overview
EduGenie is an intelligent education assistant for students and learners. It accepts PDF, DOCX, or TXT study materials, builds a local vector store, and answers questions using a document-aware AI workflow.

## 🧠 Features
- Upload PDF / DOCX / TXT
- Document retrieval with ChromaDB
- SentenceTransformer embeddings
- LangChain text chunking and prompt handling
- Groq LLaMA-3.3-70B inference
- Flask-based web UI with custom HTML/CSS frontend
- Local vector store cleanup between uploads

## 📂 Project Structure
```
EduGenie_Chatbot/
│── app.py
│── llm.py
│── .env
│── db/
│── requirements.txt
│── templates/
│   └── index.html
│── static/
│   └── styles.css
│── .gitignore
```

## 🚀 Run Locally
1. Create and activate your virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with `GROQ_API_KEY`.
4. Start the app:
   ```bash
   python app.py
   ```
5. Open the URL shown in the console.

## ⚠️ Notes
- Do not commit `.env` or `db/` to source control.
- The app uses a local ChromaDB persistent client in `db/`.
- For deployment, ensure `GROQ_API_KEY` is set in the environment.
