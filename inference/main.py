import logging
from pathlib import Path
import os
import threading

import httpx
import uvicorn
import gradio as gr
from fastapi import FastAPI
from sentence_transformers import CrossEncoder

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.nebius import NebiusEmbedding
from llama_index.llms.nebius import NebiusLLM
from llama_index.vector_stores.faiss import FaissVectorStore
from whoosh.index import open_dir

# Try package-level import first, fall back to local import
try:
    from inference.services import SearchService, LLMService
    from inference.api.routes import router, initialize_routes
    from inference.utils import load_config, load_environment, setup_directories, setup_logging
except ImportError:
    from services import SearchService, LLMService
    from api.routes import router, initialize_routes
    from utils import load_config, load_environment, setup_directories, setup_logging

# Initialize FastAPI app
app = FastAPI(title="Inference Pipeline API")
app.include_router(router)

def initialize_services(config, api_key, api_base):
    """Initialize all required services."""
    try:
        # Setup embedding model
        custom_http_client = httpx.Client(timeout=60.0)
        embedding_model = NebiusEmbedding(
            api_key=api_key,
            model_name=config['embedding_model']['model_name'],
            http_client=custom_http_client,
            api_base=api_base,
        )
        Settings.embed_model = embedding_model

        # Setup LLM
        llm = NebiusLLM(
            api_key=api_key,
            model=config['llm']['model_name'],
            temperature=config['llm']['temperature'],
        )
        Settings.llm = llm

        # Load indices
        whoosh_index_path = Path(config['directories']['whoosh_index_path'])
        faiss_index_path = Path(config['directories']['faiss_index_path'])

        if not whoosh_index_path.exists():
            raise FileNotFoundError(f"Whoosh index not found at {whoosh_index_path}")
        if not (faiss_index_path / "default__vector_store.faiss").exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")

        # Load Whoosh index
        whoosh_index = open_dir(str(whoosh_index_path))
        logging.info("Whoosh index loaded successfully")

        # Load FAISS index
        vector_store = FaissVectorStore.from_persist_path(
            str(faiss_index_path / "default__vector_store.faiss")
        )
        storage_context = StorageContext.from_defaults(
            persist_dir=str(faiss_index_path),
            vector_store=vector_store,
        )
        faiss_index = load_index_from_storage(storage_context, embedding=embedding_model)
        logging.info("FAISS index loaded successfully")

        # Initialize cross-encoder for reranking
        cross_encoder = CrossEncoder(config['reranking']['cross_encoder_model'])

        # Initialize services
        search_service = SearchService(
            faiss_index=faiss_index,
            whoosh_index=whoosh_index,
            cross_encoder=cross_encoder
        )
        
        llm_service = LLMService(llm)

        return search_service, llm_service

    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        raise

def build_gradio_interface(search_service, llm_service):
    """Build Gradio interface for the chatbot."""
    def chatbot_interface(message, history):
        try:
            if not message.strip():
                return history, "Please enter a valid question."

            # Perform search
            search_results = search_service.search(message)

            # Prepare context
            context = "\n\n".join([text for text, score, metadata in search_results])

            # Generate response
            response, _ = llm_service.generate_response(message, context)

            # Format response with references
            formatted_response = llm_service.format_response_with_references(
                response, search_results
            )

            history.append((message, formatted_response))
            return history, history

        except Exception as e:
            logging.error(f"Error in chatbot interface: {e}")
            return history, "An error occurred. Please try again."

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
        gr.Markdown("### RAG Chatbot")
        gr.Markdown("Ask any question about the documents in the knowledge base.")

        chatbot = gr.Chatbot(
            label="Chat History",
            show_label=True,
            height=400
        )
        
        with gr.Row():
            msg = gr.Textbox(
                lines=2,
                placeholder="Enter your question here...",
                label="Your Question",
                elem_id="user-input"
            )
            submit_btn = gr.Button("Send", elem_id="submit-button")
        
        clear_btn = gr.Button("Clear Chat History")
        state = gr.State([])

        # Handle message submission
        submit_btn.click(
            fn=chatbot_interface,
            inputs=[msg, state],
            outputs=[chatbot, state]
        )
        msg.submit(
            fn=chatbot_interface,
            inputs=[msg, state],
            outputs=[chatbot, state]
        )
        
        # Clear chat history
        clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

    return demo

def main():
    try:
        # Load configuration
        config = load_config(Path("config/config.sample.json"))
        api_key, api_base = load_environment()

        # Setup logging
        setup_logging(
            log_folder=Path(config['directories']['inference_log_folder']),
            log_file=config['logging']['log_file'],
            max_bytes=config['logging']['max_bytes'],
            backup_count=config['logging']['backup_count'],
            log_level=config['logging']['level'],
            log_format=config['logging']['format']
        )

        # Setup directories
        setup_directories(config)

        # Initialize services
        search_service, llm_service = initialize_services(config, api_key, api_base)

        # Initialize FastAPI routes
        initialize_routes(search_service, llm_service)

        if os.getenv("CI") is None:
            # Launch FastAPI with Uvicorn in a separate thread
            uvicorn_thread = threading.Thread(
                target=lambda: uvicorn.run(app, host="0.0.0.0", port=8082)
            )
            uvicorn_thread.start()
            logging.info("FastAPI interface launched")

            # Launch Gradio interface
            demo = build_gradio_interface(search_service, llm_service)
            demo.launch(
                server_name="0.0.0.0",
                server_port=7862,
                share=False
            )
        else:
            logging.info("CI/CD environment detected. Skipping interface launch.")

    except Exception as e:
        logging.error(f"Application startup failed: {e}")
        raise

if __name__ == "__main__":
    main() 