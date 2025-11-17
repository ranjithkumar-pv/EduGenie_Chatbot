# 📘 EduGenie – AI-Powered Educational Assistant  
*A Document-Aware RAG Chatbot using Groq, LangChain, ChromaDB & Streamlit*

## 🚀 Overview
EduGenie is an intelligent educational assistant that helps students understand study materials quickly.  
It allows users to upload PDF, DOCX, or TXT study materials and ask questions about them.

EduGenie works in two modes:

### 🔹 1. Document-Aware RAG Mode  
Uses Retrieval-Augmented Generation (RAG) to answer strictly from the uploaded document.

### 🔹 2. Educational General Knowledge Mode  
If no document is uploaded, EduGenie answers education-related questions using Groq’s LLaMA-3.3-70B model.

---

## 🧠 Features
- Upload PDF / DOCX / TXT  
- NLTK preprocessing  
- LangChain text chunking  
- SentenceTransformer embeddings  
- ChromaDB vector storage  
- Fast inference using Groq LLaMA-3.3-70B  
- Streamlit interface  
- Auto-clears old vectors  
- Educational-only assistant  

---

## 📂 Project Structure
```
EduGenie_Chatbot/
│── app.py
│── llm.py
│── .env
│── db/
│── requirements.txt
