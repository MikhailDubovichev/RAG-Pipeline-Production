# inference/inference.py

########################################
# IMPORT DEPENDENCIES
########################################

import os
import json
import logging
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv

import faiss
import gradio as gr
import httpx
import openai
from sentence_transformers import CrossEncoder
from transformers import GPT2TokenizerFast
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from logging.handlers import RotatingFileHandler

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.nebius import NebiusEmbedding
from llama_index.llms.nebius import NebiusLLM
from llama_index.vector_stores.faiss import FaissVectorStore

import threading

########################################
# INITIALIZE FASTAPI APP
########################################

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Inference Pipeline API")

########################################
# LOAD ENVIRONMENT VARIABLES
########################################

load_dotenv(dotenv_path=Path(".env"))

########################################
# CONFIG & INITIAL SETUP
########################################

# Load configuration
CONFIG_PATH = Path("config/config.sample.json")  # Use sample for reference
with open(CONFIG_PATH, encoding='utf-8-sig') as config_file:
    config = json.load(config_file)

# Access Environment Variables
api_key = os.getenv("API_KEY")
api_base = os.getenv("API_BASE")

if not api_key or not api_base:
    raise EnvironmentError("API_KEY and API_BASE must be set as environment variables.")

# Set OpenAI API credentials
openai.api_key = api_key
openai.api_base = api_base

########################################
# DEFINE CHUNK PARAMETERS
########################################

reranking_threshold = config['reranking']['threshold']

########################################
# DEFINE DIRECTORIES
########################################

log_folder = Path(config['directories']['inference_log_folder'])
whoosh_index_path = Path(config['directories']['whoosh_index_path'])
faiss_index_path = Path(config['directories']['faiss_index_path'])

########################################
# SETUP LOGGING
########################################

def setup_logging(
    log_folder: Path,
    log_file: str = "inference.log",
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
) -> None:
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = log_folder / log_file

    rotating_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )

    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format'],
        handlers=[
            rotating_handler,
            logging.StreamHandler(),
        ],
        force=True,
    )

    logging.info("Logging with rotation has been successfully configured.")

# Initialize logging
setup_logging(
    log_folder=log_folder,
    log_file=config['logging']['log_file'],
    max_bytes=config['logging']['max_bytes'],
    backup_count=config['logging']['backup_count'],
)

logging.info("Test message: Logging setup verification.")

########################################
# LOAD MODELS
########################################

# Setup NebiusEmbedding model
custom_http_client = httpx.Client(timeout=60.0)
embedding_model = NebiusEmbedding(
    api_key=api_key,
    model_name=config['embedding_model']['model_name'],
    http_client=custom_http_client,
    api_base=api_base,
)

# Setup Nebius LLM
llm = NebiusLLM(
    api_key=api_key,
    model=config['llm']['model_name'],
    temperature=config['llm']['temperature'],
)

# Set the embedding model in LlamaIndex settings
Settings.embed_model = embedding_model
Settings.llm_model = llm

########################################
# LOAD INDICES
########################################

# Load FAISS Index
if (faiss_index_path / "index_store.json").exists():
    vector_store = FaissVectorStore.from_persist_path(str(faiss_index_path / "default__vector_store.faiss"))
    storage_context = StorageContext.from_defaults(
        persist_dir=str(faiss_index_path),
        vector_store=vector_store,
    )
    faiss_index = load_index_from_storage(storage_context, embedding=embedding_model)
    logging.info("FAISS index loaded successfully (LlamaIndex).")
else:
    logging.error(f"LlamaIndex FAISS index not found at {faiss_index_path}. Ensure the Inference pipeline has been run.")
    raise FileNotFoundError(f"No LlamaIndex index found at {faiss_index_path}.")

# Load Whoosh Index
if whoosh_index_path.exists():
    whoosh_index = open_dir(str(whoosh_index_path))
    logging.info("Whoosh index loaded successfully.")
else:
    logging.error(f"Whoosh index not found at {whoosh_index_path}. Ensure the Inference pipeline has been run.")
    raise FileNotFoundError(f"No Whoosh index found at {whoosh_index_path}.")

########################################
# SEARCH FUNCTIONS
########################################

def perform_vector_search(query: str, top_k: int = 10):
    retriever = faiss_index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)
    return [(res.node.text, res.score if res.score is not None else 0.0, res.node.metadata) for res in results]

def perform_bm25_search(query: str, top_k: int = 10):
    with whoosh_index.searcher() as searcher:
        parser = QueryParser("content", schema=whoosh_index.schema)
        try:
            parsed_query = parser.parse(query)
            results = searcher.search(parsed_query, limit=top_k)
            bm25_results = []
            for hit in results:
                content = hit["content"]
                score = hit.score
                metadata = {
                    "source": hit.get("source", "Unknown Source"),
                    "sheet": hit.get("sheet", None),
                    "row_number": hit.get("row_number", None),
                    "slide_number": hit.get("slide_number", None),
                    "section": hit.get("section", None),
                    "chunk_number": hit.get("chunk_number", None),
                    "total_chunks_in_section": hit.get("total_chunks_in_section", None),
                    "page_number": hit.get("page_number", None),
                    "doc_id": hit.get("doc_id", None),
                }
                bm25_results.append((content, score, metadata))
            logging.info(f"BM25 search results: {bm25_results}")
            return bm25_results
        except Exception as e:
            logging.error(f"Whoosh search error: {e}")
            return []

def combine_results(vector_results, bm25_results, alpha: float = 0.5):
    combined_scores = {}
    metadata_mapping = {}

    # Process FAISS results
    for idx, (text, score, metadata) in enumerate(vector_results):
        if text:
            combined_scores[text] = combined_scores.get(text, 0) + alpha / (idx + 1)
            metadata_mapping[text] = metadata

    # Process BM25 results
    for idx, (text, score, metadata) in enumerate(bm25_results):
        if text:
            combined_scores[text] = combined_scores.get(text, 0) + (1 - alpha) / (idx + 1)
            if text not in metadata_mapping:
                metadata_mapping[text] = metadata

    combined_results = [(text, score, metadata_mapping[text]) for text, score in combined_scores.items()]
    sorted_results = sorted(combined_results, key=lambda x: x[1], reverse=True)
    return sorted_results

########################################
# RERANKING FUNCTION
########################################

# Initialize CrossEncoder
cross_encoder = CrossEncoder(config['reranking']['cross_encoder_model'])

def rerank_with_cross_encoder(query: str, documents: list, top_k: int = 10, batch_size: int = 16):
    try:
        input_pairs = [(query, doc[0]) for doc in documents]
        scores = []

        for i in range(0, len(input_pairs), batch_size):
            batch = input_pairs[i:i + batch_size]
            try:
                batch_scores = cross_encoder.predict(batch)
                scores.extend(batch_scores)
            except Exception as e:
                logging.error(f"Cross-encoder prediction failed for batch {i // batch_size}: {e}")
                scores.extend([0.0] * len(batch))

        scored_docs = [(doc[0], score, doc[2]) for doc, score in zip(documents, scores)]
        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        filtered_reranked_docs = [doc for doc in reranked_docs if doc[1] >= reranking_threshold]

        return filtered_reranked_docs[:top_k]
    except Exception as e:
        logging.error(f"Unexpected error in rerank_with_cross_encoder: {e}")
        return []

########################################
# LLM RESPONSE FUNCTION
########################################

def generate_llm_response(prompt: str) -> str:
    try:
        response = llm.complete(prompt=prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"LLM generation failed: {e}")
        return "An error occurred while generating the response. Please try again later."

########################################
# GET ANSWER FUNCTION
########################################

def get_answer(query: str, top_k: int = 10, alpha: float = 0.5, history=None):
    try:
        start_time = time.time()
        logging.info("Pipeline started.")

        if not query.strip():
            logging.warning("Received an empty query.")
            return history, "Please enter a valid question."

        if history is None:
            history = []

        # Vector Search
        vector_search_start = time.time()
        vector_results = perform_vector_search(query, top_k * 2)
        vector_search_time = time.time() - vector_search_start
        logging.info(f"Vector search completed in {vector_search_time:.2f} seconds.")

        # BM25 Search
        bm25_search_start = time.time()
        bm25_results = perform_bm25_search(query, top_k * 2)
        bm25_search_time = time.time() - bm25_search_start
        logging.info(f"BM25 search completed in {bm25_search_time:.2f} seconds.")

        # Combine Results
        combine_results_start = time.time()
        hybrid_results = combine_results(vector_results, bm25_results, alpha)
        combine_results_time = time.time() - combine_results_start
        logging.info(f"Combining results completed in {combine_results_time:.2f} seconds.")

        # Rerank Results
        rerank_start = time.time()
        reranked_results = rerank_with_cross_encoder(query, hybrid_results, top_k=top_k, batch_size=top_k)
        rerank_time = time.time() - rerank_start
        logging.info(f"Reranking completed in {rerank_time:.2f} seconds.")

        # Prepare Context
        context_start = time.time()
        context = "\n\n".join([text for text, score, metadata in reranked_results])
        context_time = time.time() - context_start
        logging.info(f"Context preparation completed in {context_time:.2f} seconds.")

        # Generate LLM Response
        llm_start = time.time()
        llm_prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Based on the provided context, is there enough relevant information to answer the question? "
            "Respond with 'Yes' or 'No'. If yes, provide the answer; otherwise, say 'No relevant information available.'"
        )
        llm_response = generate_llm_response(llm_prompt)
        llm_time = time.time() - llm_start
        logging.info(f"LLM response generated in {llm_time:.2f} seconds.")

        # Format Final Response
        response_format_start = time.time()
        references = []
        for text, score, metadata in reranked_results:
            if metadata:
                source = metadata.get("source", "Unknown Source")
                sheet = f"Sheet: {metadata['sheet']}" if metadata.get("sheet") else ""
                row = f"Row: {metadata['row_number']}" if metadata.get("row_number") else ""
                slide = f"Slide: {metadata['slide_number']}" if metadata.get("slide_number") else ""
                section = f"Section: {metadata['section']}" if metadata.get("section") else ""
                page = f"Page: {metadata['page_number']}" if metadata.get("page_number") else ""
                chunk = f"Chunk: {metadata['chunk_number']}" if metadata.get("chunk_number") else ""
                total_chunks = f"Total Chunks: {metadata['total_chunks_in_section']}" if metadata.get("total_chunks_in_section") else ""

                metadata_str = ", ".join(filter(None, [source, sheet, row, slide, section, page, chunk, total_chunks]))
                references.append(f"{metadata_str}: {text[:50]}...")
            else:
                references.append(f"{text[:50]}...")

        references_text = "\n".join([f"- {ref}" for ref in references])
        if llm_response.lower() == 'no relevant information available.':
            final_answer = llm_response
        else:
            final_answer = f"{llm_response}\n\nSources:\n{references_text}"

        response_format_time = time.time() - response_format_start
        logging.info(f"Response formatting completed in {response_format_time:.2f} seconds.")

        # Update chat history
        history.append((query, final_answer))
        total_time = time.time() - start_time
        logging.info(f"Pipeline completed in {total_time:.2f} seconds.")

        return history, history

    except Exception as e:
        logging.error(f"Unexpected Error in get_answer: {e}")
        response = "An unexpected error occurred. Please try again later."
        if history is not None:
            history.append((query, response))
        return history, history

########################################
# CHATBOT INTERFACE FUNCTION
########################################

def chatbot_interface(user_input, history):
    try:
        logging.info(f"Received user input: {user_input}")
        if not user_input.strip():
            return history, "Please enter a valid query."

        updated_history, response = get_answer(user_input, history=history)
        logging.info(f"LLM Response: {response}")
        return updated_history, response

    except Exception as e:
        logging.error(f"Error in chatbot_interface: {e}")
        return history, "An unexpected error occurred. Please try again later."

########################################
# GRADIO INTERFACE SETUP
########################################

def launch_gradio_interface():
    with gr.Blocks(css="""
        #user-input {
            padding-top: 2px !important;
            font-size: 24px !important;
            line-height: 1.2 !important;
            background-color: #f9f9f9 !important;
        } """) as demo:

        gr.Markdown("### RAG Chatbot for Oil & Gas Drilling Engineers")

        # Chatbot Tab
        with gr.Tab("Chatbot"):
            with gr.Row():
                chatbot = gr.Chatbot(label="Chat Window", show_label=True)
            with gr.Row(equal_height=True):
                user_input = gr.Textbox(
                    lines=2,
                    placeholder="Enter your question here...",
                    label="Your Message",
                    elem_id="user-input"
                )
                send_button = gr.Button("Send", scale=0.25)
            state = gr.State(value=[])

            send_button.click(
                fn=chatbot_interface,
                inputs=[user_input, state],
                outputs=[chatbot, state]
            )

        # Settings Tab (Placeholder)
        with gr.Tab("Settings"):
            gr.Markdown("### Settings")
            gr.Markdown("Settings can be configured here in the future.")

        # Footer
        gr.Markdown("© Noname Company")

    demo.launch(server_name="0.0.0.0", server_port=7861)  # Change port to 7861

########################################
# RUN APPLICATION
########################################

def main():
    if os.getenv("CI") is None:
        # Launch FastAPI with Uvicorn in a separate thread
        uvicorn_thread = threading.Thread(
            target=lambda: uvicorn.run(app, host="0.0.0.0", port=8080)
        )
        uvicorn_thread.start()
        logging.info("FastAPI interface launched.")

        # Launch Gradio interface in blocking mode in the main thread
        launch_gradio_interface()

        # Optionally, wait for the FastAPI thread (it runs indefinitely)
        uvicorn_thread.join()
    else:
        logging.info("CI/CD detected. Skipping Gradio and FastAPI interface launch.")
        logging.info("Starting Inference Pipeline.")
        
if __name__ == "__main__":
    main()

