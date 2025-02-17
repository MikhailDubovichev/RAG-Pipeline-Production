import logging
import os
from pathlib import Path
import httpx

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import gradio as gr
import threading

from llama_index.embeddings.nebius import NebiusEmbedding
from llama_index.core import Settings

# Import services and utils
from .services import DocumentProcessor, IndexingService, FileService
from .utils import load_config, load_environment, setup_directories, setup_logging

# Initialize FastAPI app
app = FastAPI(title="Data Preparation Pipeline API")

def initialize_services(config, api_key, api_base):
    """Initialize all required services."""
    # Setup embedding model
    custom_http_client = httpx.Client(timeout=60.0)
    embedding_model = NebiusEmbedding(
        api_key=api_key,
        model_name=config['embedding_model']['model_name'],
        http_client=custom_http_client,
        api_base=api_base,
    )
    Settings.embed_model = embedding_model

    # Initialize services
    document_processor = DocumentProcessor(
        max_tokens=config['chunking']['max_tokens'],
        overlap_tokens=config['chunking']['overlap_tokens']
    )
    
    indexing_service = IndexingService(embedding_model)
    
    file_service = FileService(
        to_process_dir=Path(config['directories']['to_process_dir']),
        processed_dir=Path(config['directories']['processed_dir'])
    )
    
    return document_processor, indexing_service, file_service

def process_documents(doc_processor: DocumentProcessor, 
                     indexing_service: IndexingService,
                     file_service: FileService,
                     config: dict):
    """Process documents and update indices."""
    try:
        # Get paths for record keeping
        processed_files_record_pdf = Path(config['directories']['processed_dir']) / "processed_files_pdf.json"
        processed_files_record_excel = Path(config['directories']['processed_dir']) / "processed_files_excel.json"
        processed_files_record_ppt = Path(config['directories']['processed_dir']) / "processed_files_ppt.json"
        processed_files_record_doc = Path(config['directories']['processed_dir']) / "processed_files_doc.json"

        # Load records of processed files
        processed_files_pdf = file_service.load_processed_files(processed_files_record_pdf)
        processed_files_excel = file_service.load_processed_files(processed_files_record_excel)
        processed_files_ppt = file_service.load_processed_files(processed_files_record_ppt)
        processed_files_doc = file_service.load_processed_files(processed_files_record_doc)

        # Get all files by type
        pdf_files, excel_files, ppt_files, doc_files = file_service.get_files_by_type()

        # Identify new files
        new_pdf_files = file_service.get_new_files(pdf_files, processed_files_pdf)
        new_excel_files = file_service.get_new_files(excel_files, processed_files_excel)
        new_ppt_files = file_service.get_new_files(ppt_files, processed_files_ppt)
        new_doc_files = file_service.get_new_files(doc_files, processed_files_doc)

        # Process new documents
        all_new_docs = []
        
        if new_pdf_files:
            pdf_docs = doc_processor.load_pdf_docs(new_pdf_files)
            all_new_docs.extend(pdf_docs)
            
        if new_excel_files:
            excel_docs = doc_processor.load_excel_docs(new_excel_files)
            all_new_docs.extend(excel_docs)
            
        if new_ppt_files:
            ppt_docs = doc_processor.load_ppt_docs(new_ppt_files)
            all_new_docs.extend(ppt_docs)
            
        if new_doc_files:
            doc_docs = doc_processor.load_doc_docs(new_doc_files)
            all_new_docs.extend(doc_docs)

        if all_new_docs:
            # Update indices
            indexing_service.create_or_update_whoosh_index(
                all_new_docs, 
                Path(config['directories']['whoosh_index_path'])
            )
            indexing_service.create_or_update_faiss_index(
                all_new_docs, 
                Path(config['directories']['faiss_index_path'])
            )

            # Move processed files
            file_service.move_to_processed(new_pdf_files)
            file_service.move_to_processed(new_excel_files)
            file_service.move_to_processed(new_ppt_files)
            file_service.move_to_processed(new_doc_files)

            # Update records
            file_service.update_processed_files_record(processed_files_record_pdf, new_pdf_files)
            file_service.update_processed_files_record(processed_files_record_excel, new_excel_files)
            file_service.update_processed_files_record(processed_files_record_ppt, new_ppt_files)
            file_service.update_processed_files_record(processed_files_record_doc, new_doc_files)

        return len(all_new_docs)
    except Exception as e:
        logging.error(f"Error in document processing: {e}")
        raise

# FastAPI endpoints
@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload a file to the to_process directory."""
    try:
        file_path = Path(config['directories']['to_process_dir']) / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logging.info(f"File {file.filename} uploaded successfully.")
        return {"message": f"File {file.filename} uploaded successfully."}
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"File upload failed: {str(e)}"}
        )

@app.post("/process")
async def process_files():
    """Trigger document processing."""
    try:
        num_processed = process_documents(doc_processor, indexing_service, file_service, config)
        return {"message": f"Processed {num_processed} new documents."}
    except Exception as e:
        logging.error(f"Error processing documents: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Document processing failed: {str(e)}"}
        )

# Gradio interface
def upload_file_gradio(file):
    if not file:
        return "No file provided."
    try:
        file_path = Path(config['directories']['to_process_dir']) / file.name
        with open(file_path, "wb") as f:
            f.write(file.read())
        return f"File {file.name} uploaded successfully."
    except Exception as e:
        logging.error(f"Error uploading file via Gradio: {e}")
        return f"File upload failed: {str(e)}"

def process_files_gradio():
    try:
        num_processed = process_documents(doc_processor, indexing_service, file_service, config)
        return f"Processed {num_processed} new documents."
    except Exception as e:
        logging.error(f"Error processing documents via Gradio: {e}")
        return f"Document processing failed: {str(e)}"

def build_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Data Preparation Pipeline")
        
        with gr.Tab("Upload"):
            file_input = gr.File(label="Upload File")
            upload_button = gr.Button("Upload")
            upload_output = gr.Textbox(label="Upload Status")
            upload_button.click(
                fn=upload_file_gradio,
                inputs=file_input,
                outputs=upload_output
            )
            
        with gr.Tab("Process"):
            process_button = gr.Button("Process Documents")
            process_output = gr.Textbox(label="Processing Status")
            process_button.click(
                fn=process_files_gradio,
                inputs=None,
                outputs=process_output
            )
    
    return demo

def main():
    global config, doc_processor, indexing_service, file_service
    
    try:
        # Load configuration
        config = load_config(Path("config/config.sample.json"))
        api_key, api_base = load_environment()
        
        # Setup logging
        setup_logging(
            log_folder=Path(config['directories']['data_preparation_log_folder']),
            log_file=config['logging']['log_file'],
            max_bytes=config['logging']['max_bytes'],
            backup_count=config['logging']['backup_count'],
            log_level=config['logging']['level'],
            log_format=config['logging']['format']
        )
        
        # Setup directories
        setup_directories(config)
        
        # Initialize services
        doc_processor, indexing_service, file_service = initialize_services(config, api_key, api_base)
        
        if os.getenv("CI") is None:
            # Launch FastAPI with Uvicorn in a separate thread
            uvicorn_thread = threading.Thread(
                target=lambda: uvicorn.run(app, host="0.0.0.0", port=8080)
            )
            uvicorn_thread.start()
            logging.info("FastAPI interface launched.")

            # Launch Gradio interface
            demo = build_gradio_interface()
            # Try different ports in case the default is taken
            for port in range(7860, 7870):
                try:
                    demo.launch(server_name="0.0.0.0", server_port=port)
                    logging.info(f"Gradio interface launched on port {port}")
                    break
                except Exception as e:
                    if port == 7869:  # Last attempt
                        raise Exception(f"Could not find available port for Gradio: {e}")
                    continue
        else:
            logging.info("CI/CD environment detected. Running processing only.")
            process_documents(doc_processor, indexing_service, file_service, config)
            
    except Exception as e:
        logging.error(f"Application startup failed: {e}")
        raise

if __name__ == "__main__":
    main() 