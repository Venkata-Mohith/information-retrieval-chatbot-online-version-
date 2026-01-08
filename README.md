📌 Project Title

 Online based information retrieval system 

 

🧠 Project Overview

This project is an online intelligent document assistant designed to answer user queries based on a curated collection of missile system–related PDFs and general technical documents.

The project is an online adaptation of a desktop-based application originally developed during a DRDO internship using PyQt. Since the offline version could not be shared publicly, this version was rebuilt as a web-accessible application, preserving the core idea while extending accessibility and scalability.

Users can upload or select documents, ask natural language questions, and receive context-aware answers extracted from the document content.

🏷️ Tags

Document Intelligence PDF Question Answering NLP
Python Streamlit LLM Information Retrieval AI Assistant

🛠️ Technologies Used
Programming Language

Python

Frameworks & Libraries

Streamlit – Web UI

LangChain – Document processing & retrieval

PyMuPDF – PDF loading and parsing

ChromaDB – Vector storage & similarity search

Groq API (Free Tier) – Language model inference

Concepts

Document chunking

Vector embeddings

Semantic search

Context-aware question answering

✨ Key Features

📄 PDF-based question answering

🔍 Contextual semantic search

🌐 Web-based interface (online access)

⚡ Fast responses using optimized embeddings

📚 Supports both domain-specific and general documents

🔐 No document modification — read-only processing

🔄 Project Workflow (How It Works)

PDF Ingestion
Uploaded or predefined PDFs are loaded and parsed.

Text Chunking
Documents are split into manageable chunks for efficient processing.

Embedding Generation
Text chunks are converted into vector embeddings.

Vector Storage
Embeddings are stored in a vector database for similarity search.

Query Processing
User queries are embedded and matched with relevant document chunks.

Answer Generation
The most relevant context is passed to the language model to generate responses.

 
🚀 How I Built It

Started from an offline PyQt-based desktop application

Refactored logic into reusable backend components

Migrated UI to Streamlit for online accessibility

Integrated vector databases and LLM APIs

Optimized document handling for real-time querying

🔧 How to Run the Project Locally
Prerequisites

Python 3.9+

Virtual environment (recommended)

Installation
git clone https://github.com/Venkata-Mohith/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt

Run the Application
streamlit run app.py

📈 Future Improvements

User authentication & session management

Support for more document formats (DOCX, TXT)

Improved UI/UX and document visualization

Fine-tuned domain-specific language models

Deployment on scalable cloud infrastructure

Multi-user document management

🎥 Output / Demo

 

https://github.com/user-attachments/assets/b81d9ec8-0a17-451e-b94d-e24a75d38b3f



📌 Notes

This project uses a free-tier API for demonstration purposes

Intended for educational and research use

No sensitive or restricted documents are included
