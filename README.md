# RAG Pipeline

A RAG (Retrieval-Augmented Generation) pipeline with separate data preparation and inference components.

## Project Structure
```
rag_pipeline/
├── data_preparation/     # Data processing pipeline
├── inference/           # Inference/chatbot pipeline
├── config/             # Configuration files
└── logs/              # Log files
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

## Usage

### Data Preparation Pipeline
Process new documents and create/update indices:
```bash
python run_data_preparation.py
```
This will:
- Process documents from `data/to_process/`
- Create/update search indices
- Move processed files to `data/processed/`

### Inference Pipeline
Run the chatbot/inference service:
```bash
python run_inference.py
```
This will:
- Start FastAPI server on port 8080
- Launch Gradio interface on port 7861
- Enable document querying

## Interfaces

### Data Preparation
- FastAPI: http://localhost:8080
- Gradio: http://localhost:7860

### Inference
- FastAPI: http://localhost:8080
- Gradio: http://localhost:7861

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
rag_pipeline/
├── data_preparation/
│ ├── init.py
│ ├── main.py
│ ├── processors/
│ │ ├── init.py
│ │ ├── pdf_processor.py
│ │ ├── excel_processor.py
│ │ ├── ppt_processor.py
│ │ └── word_processor.py
│ └── utils/
│ ├── init.py
│ ├── logging_utils.py
│ ├── file_utils.py
│ └── index_utils.py
├── inference/
│ ├── init.py
│ ├── main.py
│ ├── services/
│ │ ├── init.py
│ │ ├── search_service.py
│ │ ├── reranking_service.py
│ │ └── llm_service.py
│ └── utils/
│ ├── init.py
│ ├── logging_utils.py
│ └── response_formatter.py
├── config/
│ └── config.json
└── common/
├── init.py
└── model_loader.py

### Directory Structure Explanation
- `data_preparation/`: Contains all code related to document processing and indexing
  - `processors/`: Individual document type processors (PDF, Excel, PPT, Word)
  - `utils/`: Utility functions for file handling, logging, and indexing
- `inference/`: Contains code for the query and response pipeline
  - `services/`: Core services for search, reranking, and LLM interaction
  - `utils/`: Utility functions for logging and response formatting
- `common/`: Shared components used by both pipelines
- `config/`: Configuration files for the application
