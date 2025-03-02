# RAG Pipeline

A RAG (Retrieval-Augmented Generation) pipeline with separate data preparation and inference components. This project is built based on the MVP for the "RAG-pipeline-MVP" (Jupiter Notebook), that also could be found in my GitHub repository. The pipeline uses LLM and embedding models from Nebius AI Studio platform.

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
     - `document_processor.py`: Handles document parsing and chunking using llama_index's PDFReader
     - `indexing_service.py`: Manages VectorStoreIndex and Whoosh index creation
     - `file_service.py`: Handles file operations and tracking
   - `utils/`:
     - `config_utils.py`: Configuration management
     - `logging_utils.py`: Centralized logging setup and management for all pipeline components

3. **Document Processing Flow**:
   ```
   Documents → DocumentProcessor → Chunks → IndexingService → Search Indices
   └── PDF, Excel, PPT, Word    └── Text extraction    └── VectorStoreIndex (Vector)
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
     - `search_service.py`: Hybrid search implementation (Vector + Whoosh)
     - `llm_service.py`: LLM interaction and response generation using NebiusLLM
   - `api/`:
     - `routes.py`: FastAPI endpoints
   - `utils/`:
     - Configuration and logging utilities

3. **Query Processing Flow**:
   ```
   User Query → Search Service → LLM Service → Formatted Response
                └── Vector Search          └── Context Integration
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

1. **Vector Search**:
   - Semantic similarity using embeddings
   - Efficient nearest neighbor search via VectorStoreIndex
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
   - PDF: Page-by-page extraction using llama_index's PDFReader
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

### LLM Integration
The pipeline uses NebiusLLM for response generation:

1. **Context Integration**:
   - Combines search results with user query
   - Optimized prompt engineering
   - Relevance checking

2. **Response Formatting**:
   - Source attribution
   - Metadata inclusion
   - Structured output

3. **Error Handling**:
   - Graceful fallbacks
   - User-friendly error messages
   - Detailed logging

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

## Testing
The project includes comprehensive test coverage:

1. **Unit Tests**:
   - Individual component testing
   - Mock-based isolation
   - Function-level validation

2. **Integration Tests**:
   - Component interaction testing
   - End-to-end pipeline validation
   - Mock services for external dependencies

3. **Test Configuration**:
   - Temporary directories and files
   - Isolated test environment
   - Cleanup after test execution

## Project Structure
```
rag_pipeline/
├── data_preparation/
│ ├── __init__.py
│ ├── main.py
│ ├── services/
│ │ ├── __init__.py
│ │ ├── document_processor.py
│ │ ├── indexing_service.py
│ │ └── file_service.py
│ └── utils/
│   ├── __init__.py
│   ├── logging_utils.py
│   └── config_utils.py
├── inference/
│ ├── __init__.py
│ ├── main.py
│ ├── services/
│ │ ├── __init__.py
│ │ ├── search_service.py
│ │ └── llm_service.py
│ ├── api/
│ │ ├── __init__.py
│ │ └── routes.py
│ └── utils/
│   ├── __init__.py
│   ├── logging_utils.py
│   └── response_formatter.py
├── common/
│ ├── __init__.py
│ ├── config_utils.py
│ └── logging_utils.py
├── tests/
│ ├── data_preparation/
│ ├── inference/
│ └── test_integration.py
├── config/
│ └── config.json
├── run_data_preparation.py
├── run_inference.py
├── requirements.txt
└── .env.example
```

## Key Dependencies

- **llama_index**: Core framework for RAG pipeline components
  - PDFReader for document processing
  - VectorStoreIndex for vector search
  - NebiusLLM for language model integration

- **Whoosh**: Full-text search engine for keyword-based retrieval

- **FastAPI & Gradio**: Web interfaces for API and chat interaction

- **pytest**: Testing framework for unit and integration tests

## Configuration

The pipeline uses a JSON configuration file with the following sections:

```json
{
    "directories": {
        "to_process_dir": "path/to/input",
        "processed_dir": "path/to/processed",
        "log_folder": "path/to/logs",
        "whoosh_index_path": "path/to/whoosh",
        "faiss_index_path": "path/to/faiss",
        "data_directory": "path/to/data"
    },
    "chunking": {
        "max_tokens": 1000,
        "overlap_tokens": 200
    },
    "embedding_model": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "llm": {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.1
    },
    "reranking": {
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "threshold": 0.2
    },
    "logging": {
        "data_preparation_log": "data_prep.log",
        "inference_log": "inference.log",
        "max_bytes": 10485760,
        "backup_count": 3,
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
```

## Environment Variables

The `.env` file should include:

- `API_KEY`: Your Nebius API key
- `API_BASE`: Nebius API endpoint URL
- `LOG_LEVEL`: Logging verbosity (default: INFO)
- `MAX_TOKENS`: Maximum tokens per chunk
- `OVERLAP_TOKENS`: Token overlap between chunks

Welcome to the RAG Pipeline repository! This project implements a Retrieval-Augmented Generation (RAG) system designed to process, index, and query various document types (PDF, Excel, PPT, DOCX) using advanced indexing techniques with VectorStoreIndex and Whoosh. Enhanced by Nebius LLM, this pipeline facilitates efficient information retrieval and intelligent response generation tailored for specific domains.

Features:
- Multi-Format Document Processing: Supports PDF, Excel, PowerPoint, and Word documents
- Advanced Indexing: Utilizes VectorStoreIndex for vector-based similarity search and Whoosh for BM25 ranking
- Chunking Mechanism: Splits large documents into manageable chunks based on token counts and headings
- Hybrid Search Strategy: Combines vector search and BM25 ranking to enhance retrieval accuracy
- Cross-Encoder Reranking: Refines search results using a cross-encoder model for improved relevance
- Interactive Chatbot Interface: Provides a user-friendly Gradio-based chatbot for querying indexed data
- Logging with Rotation: Implements robust logging mechanisms to monitor pipeline activities
- Secure Configuration Management: Handles sensitive information securely using configuration files and environment variables
- Comprehensive Testing: Includes unit and integration tests to ensure reliability and correctness
