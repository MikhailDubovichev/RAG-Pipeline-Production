"""
Main entry point for the inference (question-answering) pipeline.

This module implements a RAG (Retrieval-Augmented Generation) chatbot that:
1. Accepts user questions about documents in the knowledge base
2. Retrieves relevant context using both semantic (FAISS) and keyword (Whoosh) search
3. Uses an LLM (Large Language Model) to generate accurate answers
4. Provides source references for transparency

The system combines:
- Vector search (FAISS) for semantic similarity
- Keyword search (Whoosh) for exact matches
- Cross-encoder for result reranking
- LLM for natural language generation

Available interfaces:
1. REST API (via FastAPI) for programmatic access
2. Web UI (via Gradio) for interactive chat
"""

import logging  # Python's built-in logging facility
from pathlib import Path  # Object-oriented filesystem paths
import os  # Operating system interface
import threading  # Thread-based parallelism

# External dependencies
import httpx  # Modern HTTP client with timeout support
import uvicorn  # ASGI server implementation
import gradio as gr  # Framework for creating web UI
from fastapi import FastAPI  # Framework for building REST APIs
from sentence_transformers import CrossEncoder  # Model for reranking search results

# LlamaIndex components
from llama_index.core import (
    Settings,  # Global settings
    StorageContext,  # Manages index storage
    load_index_from_storage  # Loads saved indices
)
from llama_index.embeddings.nebius import NebiusEmbedding  # Text to vector conversion
from llama_index.llms.nebius import NebiusLLM  # Large Language Model interface
from llama_index.vector_stores.faiss import FaissVectorStore  # Vector similarity search

from whoosh.index import open_dir  # Keyword search engine

# Try package-level import first (for installed package)
# Fall back to local import (for development)
try:
    from inference.services import SearchService, LLMService  # Core services
    from inference.api.routes import router, initialize_routes  # API endpoints
    from inference.utils import load_config, load_environment, setup_directories  # Utilities
except ImportError:
    from services import SearchService, LLMService
    from api.routes import router, initialize_routes
    from utils import load_config, load_environment, setup_directories

from common.logging_utils import setup_logging  # Logging configuration

# Initialize FastAPI app and include routes
app = FastAPI(title="Inference Pipeline API")
app.include_router(router)

def initialize_services(config, api_key, api_base):
    """
    Initialize all required services for the inference pipeline.
    
    This function sets up four main components:
    1. Embedding Model: Converts text to numerical vectors for semantic search
    2. Language Model (LLM): Generates natural language responses
    3. Search Indices: Both FAISS (vector) and Whoosh (keyword) indices
    4. Cross-Encoder: Reranks search results for better accuracy
    
    The function expects pre-existing search indices created by the
    data preparation pipeline. If indices are not found, it raises
    an error.
    
    Process Flow:
    1. Setup embedding model and LLM with API credentials
    2. Load and verify search indices
    3. Initialize cross-encoder for result reranking
    4. Create service instances
    
    Args:
        config (dict): Configuration settings including model parameters
            and directory paths
        api_key (str): API key for Nebius services (embedding and LLM)
        api_base (str): Base URL for Nebius API
        
    Returns:
        tuple: (SearchService, LLMService) - Initialized service instances
        
    Raises:
        FileNotFoundError: If required indices are not found
        Exception: For any other initialization errors
    """
    try:
        # Setup embedding model with extended timeout for large texts
        custom_http_client = httpx.Client(timeout=60.0)  # 60-second timeout
        embedding_model = NebiusEmbedding(
            api_key=api_key,
            model_name=config['embedding_model']['model_name'],
            http_client=custom_http_client,
            api_base=api_base,
        )
        Settings.embed_model = embedding_model  # Set as global embedding model

        # Setup Language Model with temperature control
        # Temperature controls randomness in generation (0.0 = deterministic)
        llm = NebiusLLM(
            api_key=api_key,
            model=config['llm']['model_name'],
            temperature=config['llm']['temperature'],
        )
        Settings.llm = llm  # Set as global LLM

        # Get paths to pre-built search indices
        whoosh_index_path = Path(config['directories']['whoosh_index_path'])
        faiss_index_path = Path(config['directories']['faiss_index_path'])

        # Verify that required indices exist
        if not whoosh_index_path.exists():
            raise FileNotFoundError(f"Whoosh index not found at {whoosh_index_path}")
        if not (faiss_index_path / "default__vector_store.faiss").exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")

        # Load Whoosh index for keyword search
        whoosh_index = open_dir(str(whoosh_index_path))
        logging.info("Whoosh index loaded successfully")

        # Load FAISS index for semantic search
        vector_store = FaissVectorStore.from_persist_path(
            str(faiss_index_path / "default__vector_store.faiss")
        )
        storage_context = StorageContext.from_defaults(
            persist_dir=str(faiss_index_path),
            vector_store=vector_store,
        )
        faiss_index = load_index_from_storage(storage_context, embedding=embedding_model)
        logging.info("FAISS index loaded successfully")

        # Initialize cross-encoder model for reranking search results
        cross_encoder = CrossEncoder(config['reranking']['cross_encoder_model'])

        # Create service instances that will handle business logic
        search_service = SearchService(
            faiss_index=faiss_index,  # For semantic search
            whoosh_index=whoosh_index,  # For keyword search
            cross_encoder=cross_encoder  # For result reranking
        )
        
        llm_service = LLMService(llm)  # For answer generation

        return search_service, llm_service

    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        raise

def build_gradio_interface(search_service, llm_service):
    """
    Build an interactive web interface for the RAG chatbot using Gradio.
    
    This function creates a user-friendly chat interface with:
    1. Chat history display
    2. Input box for questions
    3. Send button for submission
    4. Clear button for resetting chat
    
    The interface is styled with custom CSS for better user experience:
    - Larger, more readable font for input
    - Comfortable padding and spacing
    - Distinctive button styling
    - Clear visual hierarchy
    
    Args:
        search_service (SearchService): Service for retrieving relevant context
        llm_service (LLMService): Service for generating answers
        
    Returns:
        gr.Blocks: Configured Gradio interface ready for launch
    """
    
    def chatbot_interface(message, history):
        """
        Core chatbot logic that processes user messages and generates responses.
        
        Process Flow:
        1. Validate user input
        2. Search for relevant context
        3. Generate response using LLM
        4. Format response with references
        5. Update chat history
        
        Args:
            message (str): User's question or input
            history (list): List of previous (question, answer) tuples
            
        Returns:
            tuple: (updated_history, display_history)
                - updated_history: Internal history state
                - display_history: Formatted history for display
                
        Error Handling:
        - Empty messages are rejected
        - All exceptions are caught and return user-friendly errors
        """
        try:
            # Validate user input
            if not message.strip():
                return history, "Please enter a valid question."

            # Search for relevant context in the knowledge base
            search_results = search_service.search(message)

            # Combine all relevant text passages into context
            context = "\n\n".join([text for text, score, metadata in search_results])

            # Generate response using the LLM with retrieved context
            response, _ = llm_service.generate_response(message, context)

            # Add source references to the response
            formatted_response = llm_service.format_response_with_references(
                response, search_results
            )

            # Update chat history
            history.append((message, formatted_response))
            return history, history

        except Exception as e:
            logging.error(f"Error in chatbot interface: {e}")
            return history, "An error occurred. Please try again."

    # Build the Gradio interface with custom styling
    with gr.Blocks(css="""
        #user-input {
            padding-top: 2px !important;
            font-size: 16px !important;
            line-height: 1.2 !important;
            background-color: #f9f9f9 !important;
        }
        #submit-button {
            margin-top: 10px !important;
            background-color: #2196F3 !important;
            color: white !important;
        }
    """) as demo:
        # Header and description
        gr.Markdown("### RAG Chatbot")
        gr.Markdown("Ask any question about the documents in the knowledge base.")

        # Chat history display
        chatbot = gr.Chatbot(
            label="Chat History",
            show_label=True,
            height=400
        )
        
        # Input area with submit button
        with gr.Row():
            msg = gr.Textbox(
                lines=2,
                placeholder="Enter your question here...",
                label="Your Question",
                elem_id="user-input"
            )
            submit_btn = gr.Button("Send", elem_id="submit-button")
        
        # Clear history button
        clear_btn = gr.Button("Clear Chat History")
        state = gr.State([])  # Internal state for chat history

        # Connect interface components to chatbot logic
        submit_btn.click(  # Handle button clicks
            fn=chatbot_interface,
            inputs=[msg, state],
            outputs=[chatbot, state]
        )
        msg.submit(  # Handle Enter key in textbox
            fn=chatbot_interface,
            inputs=[msg, state],
            outputs=[chatbot, state]
        )
        
        # Clear chat history when requested
        clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

    return demo

def main():
    """
    Main entry point for the inference pipeline application.
    
    This function:
    1. Loads configuration and environment settings
    2. Sets up logging infrastructure
    3. Initializes required directories
    4. Creates service instances
    5. Launches API and UI interfaces
    
    The application provides two interfaces:
    - FastAPI REST API on port 8082
    - Gradio web UI on port 7862
    
    Special handling:
    - In CI/CD environments (detected via CI env var), interfaces are not launched
    - FastAPI runs in a separate thread to not block the Gradio interface
    - All errors are logged before being re-raised
    
    Configuration:
    - Loads from config/config.json
    - Environment variables from .env file
    - Logging settings from config
    
    Error Handling:
    - All exceptions are caught, logged, and re-raised
    - Detailed error information is saved to log files
    """
    try:
        # Load configuration and environment settings
        config = load_config(Path("config/config.json"))
        api_key, api_base = load_environment()
        
        # Setup logging infrastructure with rotation
        logger = setup_logging(
            log_folder=Path(config['directories']['log_folder']),
            log_file=config['logging']['inference_log'],
            max_bytes=config['logging']['max_bytes'],
            backup_count=config['logging']['backup_count'],
            log_level=config['logging']['level'],
            log_format=config['logging']['format'],
            logger_name='inference'
        )
        
        # Create necessary directories if they don't exist
        setup_directories(config)

        # Initialize core services
        search_service, llm_service = initialize_services(config, api_key, api_base)

        # Setup FastAPI routes with service instances
        initialize_routes(search_service, llm_service)

        # Check if we're in a CI environment
        if os.getenv("CI") is None:
            # Launch FastAPI in a separate thread to not block
            uvicorn_thread = threading.Thread(
                target=lambda: uvicorn.run(app, host="0.0.0.0", port=8082)
            )
            uvicorn_thread.start()
            logging.info("FastAPI interface launched")

            # Create and launch Gradio interface
            demo = build_gradio_interface(search_service, llm_service)
            demo.launch(
                server_name="0.0.0.0",
                server_port=7862,
                share=False  # Don't expose publicly
            )
        else:
            logging.info("CI/CD environment detected. Skipping interface launch.")

    except Exception as e:
        logging.error(f"Application startup failed: {e}")
        logging.exception("Detailed error information:")
        raise

if __name__ == "__main__":
    main() 