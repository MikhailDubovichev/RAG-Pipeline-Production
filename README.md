# RAG Pipeline

A RAG (Retrieval-Augmented Generation) pipeline with separate data preparation and inference components.

## Architecture Overview

### Data Preparation Pipeline
The data preparation pipeline is responsible for processing documents and creating search indices. It follows a modular architecture:

1. **Entry Point** (`run_data_preparation.py`):
   - Validates environment setup
   - Checks for required directories and files
   - Initializes the pipeline components

2. **Core Components** (`data_preparation/`):
   - `main.py`: Orchestrates the document processing workflow
   - `services/`:
     - `document_processor.py`: Handles document parsing and chunking
     - `indexing_service.py`: Manages FAISS and Whoosh index creation
     - `file_service.py`: Handles file operations and tracking
   - `utils/`:
     - `config_utils.py`: Configuration management
     - `logging_utils.py`: Logging setup and management

3. **Document Processing Flow**:
   ```
   Documents → DocumentProcessor → Chunks → IndexingService → Search Indices
   └── PDF, Excel, PPT, Word    └── Text extraction    └── FAISS (Vector)
                                └── Chunking           └── Whoosh (Keyword)
   ```

### Inference Pipeline
The inference pipeline handles user queries and generates responses. It uses a layered architecture:

1. **Entry Point** (`run_inference.py`):
   - Environment validation
   - Index availability checks
   - Pipeline initialization

2. **Core Components** (`inference/`):
   - `main.py`: Main application logic and service orchestration
   - `services/`:
     - `search_service.py`: Hybrid search implementation (FAISS + Whoosh)
     - `llm_service.py`: LLM interaction and response generation
   - `api/`:
     - `routes.py`: FastAPI endpoints
   - `utils/`:
     - Configuration and logging utilities

3. **Query Processing Flow**:
   ```
   User Query → Search Service → LLM Service → Formatted Response
                └── Vector Search (FAISS)   └── Context Integration
                └── Keyword Search (Whoosh) └── Response Generation
                └── Result Reranking        └── Reference Addition
   ```

4. **Interfaces**:
   - FastAPI (port 8082): RESTful API for programmatic access
   - Gradio (port 7862): Web-based chat interface

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
- Start FastAPI server on port 8082
- Launch Gradio interface on port 7862
- Enable document querying

## Interfaces

### Data Preparation
- FastAPI: http://localhost:8080
- Gradio: http://localhost:7860

### Inference
- FastAPI: http://localhost:8082
- Gradio: http://localhost:7862

## Technical Details

### Search Implementation
The pipeline implements a hybrid search strategy:

1. **Vector Search (FAISS)**:
   - Semantic similarity using embeddings
   - Efficient nearest neighbor search
   - Optimized for understanding meaning

2. **Keyword Search (Whoosh)**:
   - BM25 ranking algorithm
   - Exact keyword matching
   - Good for specific terms/phrases

3. **Result Combination**:
   - Weighted combination of both search types
   - Configurable importance weights
   - Cross-encoder reranking for better relevance

### Document Processing
Documents are processed in several stages:

1. **Text Extraction**:
   - PDF: Page-by-page extraction
   - Excel: Row-by-row with column headers
   - PowerPoint: Slide-by-slide
   - Word: Section-based with heading structure

2. **Chunking**:
   - Token-based chunking with overlap
   - Heading-aware splitting for Word docs
   - Metadata preservation for context

3. **Index Updates**:
   - Incremental updates supported
   - Processed file tracking
   - Atomic index operations

### Security and Configuration
The pipeline implements several security measures:

1. **API Security**:
   - API keys stored in .env
   - Configurable access controls
   - Secure configuration loading

2. **Logging**:
   - Rotating log files
   - Configurable log levels
   - Separate logs for each component

3. **Error Handling**:
   - Graceful failure recovery
   - Detailed error reporting
   - User-friendly error messages

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

## File Structure Details

### Root Directory Files
- `.env`: Contains actual environment variables and API keys (not tracked in Git)
  - Stores sensitive configuration like API keys and custom settings
  - Should never be committed to version control
  - Created by copying and modifying `.env.example`

- `.env.example`: Template for environment variables
  - Provides example configuration without sensitive data
  - Documents required environment variables:
    - `API_KEY`: Your Nebius API key
    - `API_BASE`: Nebius API endpoint URL
    - `LOG_LEVEL`: Logging verbosity (default: INFO)
    - `MAX_TOKENS`: Maximum tokens per chunk
    - `OVERLAP_TOKENS`: Token overlap between chunks

- `.gitignore`: Specifies which files Git should ignore
  - Excludes sensitive files (.env)
  - Ignores virtual environment folder (.venv)
  - Skips cache directories and compiled files
  - Excludes data and log directories

- `requirements.txt`: Lists all Python dependencies
  - FastAPI for REST API (v0.100.0)
  - Uvicorn for ASGI server (v0.22.0)
  - Gradio for web interface (v3.50.2)
  - Document processing libraries (pdfplumber, python-docx, etc.)
  - Vector search components (FAISS, Whoosh)
  - LLM integration packages

- `setup.py`: Package configuration for installation
  - Defines project metadata
  - Lists dependencies with version constraints
  - Enables installation with pip
  - Configures package discovery

- `run_data_preparation.py`: Entry point for data processing
  - Validates environment setup
  - Initializes document processors
  - Manages document ingestion workflow
  - Creates and updates search indices

- `run_inference.py`: Entry point for inference service
  - Launches FastAPI and Gradio servers
  - Initializes search and LLM services
  - Manages query processing pipeline
  - Handles API endpoints

### Directory Contents

#### `data_preparation/`
- `processors/`: Document type-specific processors
  - `pdf_processor.py`: PDF document handling
  - `excel_processor.py`: Excel spreadsheet processing
  - `ppt_processor.py`: PowerPoint presentation parsing
  - `word_processor.py`: Word document processing
  
- `utils/`: Utility functions
  - `logging_utils.py`: Logging configuration
  - `file_utils.py`: File operations
  - `index_utils.py`: Index management

#### `inference/`
- `services/`: Core service implementations
  - `search_service.py`: Hybrid search (FAISS + Whoosh)
  - `reranking_service.py`: Result reranking
  - `llm_service.py`: LLM interaction
  
- `utils/`: Support utilities
  - `logging_utils.py`: Logging setup
  - `response_formatter.py`: Response formatting

#### `config/`
- `config.json`: Application configuration
  - Search parameters
  - Model settings
  - Processing options

#### `data/`
- `to_process/`: Input documents awaiting processing
- `processed/`: Successfully processed documents
- `faiss_index/`: FAISS vector indices
- `whoosh_index/`: Whoosh text indices

#### `logs/`
- `data_preparation/`: Data processing logs
- `inference/`: Inference service logs
- Log files use rotation to manage size

#### `.github/`
- GitHub-specific configuration
- Workflow definitions
- Issue templates

#### `.venv/`
- Python virtual environment
- Isolated dependency installation
- Not tracked in version control

### Generated Directories
- `rag_pipeline.egg-info/`: Package metadata
  - Generated during installation
  - Contains package information
  - Not tracked in version control
