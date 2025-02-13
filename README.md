Welcome to the RAG Pipeline repository! This project implements a Retrieval-Augmented Generation (RAG) system designed to process, index, and query various document types (PDF, Excel, PPT, DOCX) using advanced indexing techniques with FAISS and Whoosh. Enhanced by Large Language Models (LLMs), this pipeline facilitates efficient information retrieval and intelligent response generation tailored for specific domains.

Features
Multi-Format Document Processing: Supports PDF, Excel, PowerPoint, and Word documents.
Advanced Indexing: Utilizes FAISS for vector-based similarity search and Whoosh for BM25 ranking.
Chunking Mechanism: Splits large documents into manageable chunks based on token counts and headings.
Hybrid Search Strategy: Combines vector search and BM25 ranking to enhance retrieval accuracy.
Cross-Encoder Reranking: Refines search results using a cross-encoder model for improved relevance.
Interactive Chatbot Interface: Provides a user-friendly Gradio-based chatbot for querying indexed data.
Logging with Rotation: Implements robust logging mechanisms to monitor pipeline activities.
Secure Configuration Management: Handles sensitive information securely using configuration files and environment variables.
Scalable Deployment: Designed for easy deployment to platforms like Google Vertex AI.

Project Structure
RAG_Pipeline/
├── data_preparation/
│   ├── data_preparation.py
│   └── requirements.txt
├── inference/
│   ├── inference.py
│   └── requirements.txt
├── config/
│   └── config.json
├── logs/
│   ├── data_preparation.log
│   └── inference_pipeline.log
├── .gitignore
└──  README.md

Directory Breakdown
data_preparation/: Contains scripts and dependencies for data ingestion, preprocessing, and indexing.
inference/: Houses scripts and dependencies for querying the indexed data and generating responses.
config/: Stores configuration files, including API keys and directory paths.
logs/: Maintains log files for both data preparation and inference pipelines.
.gitignore: Specifies files and directories to exclude from version control.
README.md: Provides an overview and instructions for the project.
LICENSE: (Optional) Specifies the licensing for the project.
