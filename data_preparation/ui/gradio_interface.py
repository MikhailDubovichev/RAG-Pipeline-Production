"""
Gradio web interface implementation for the data preparation pipeline.

This module provides a user-friendly interface for:
1. Uploading PDF documents
2. Processing documents into chunks
3. Creating/updating search indices
"""

import logging
import gradio as gr
from pathlib import Path
import shutil
from datetime import datetime
from ..api.routes import DataPrepService
import os

class GradioInterface:
    def __init__(self, doc_processor, indexing_service, file_service, config):
        """
        Initialize Gradio interface with required services.
        
        Args:
            doc_processor (DocumentProcessor): Service for processing documents
            indexing_service (IndexingService): Service for managing indices
            file_service (FileService): Service for file operations
            config (dict): Configuration settings
        """
        self.config = config
        # Create DataPrepService instance for handling business logic
        self.data_prep_service = DataPrepService(
            doc_processor,
            indexing_service,
            file_service,
            config
        )
        logging.info("GradioInterface initialized with DataPrepService")

    def upload_file_gradio(self, file):
        """
        Handle file upload through the Gradio interface.
        
        Args:
            file: Gradio file object with name and path
            
        Returns:
            str: Success or error message
        """
        if file is None:
            return "No file provided."
            
        try:
            logging.info(f"Attempting to upload file: {file.name}")
            
            # Verify file extension
            file_ext = Path(file.name).suffix.lower()
            if file_ext != '.pdf':
                error_msg = f"Unsupported file type: {file_ext}. Only PDF files are supported."
                logging.error(error_msg)
                return error_msg
            
            # Get the to_process directory
            to_process_dir = Path(self.config['directories']['to_process_dir'])
            logging.info(f"Target directory: {to_process_dir}")
            
            # Create directory if needed
            try:
                to_process_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                error_msg = f"Permission denied when creating directory: {e}"
                logging.error(error_msg)
                return error_msg
            except OSError as e:
                error_msg = f"OS error when creating directory: {e}"
                logging.error(error_msg)
                return error_msg
            
            # Handle duplicate filenames
            target_path = to_process_dir / Path(file.name).name
            if target_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_path = target_path.with_name(f"{target_path.stem}_{timestamp}{target_path.suffix}")
                logging.info(f"File already exists, using new path: {target_path}")

            # Copy file with metadata preservation
            try:
                # Check if source file exists and is readable
                source_path = Path(file.name)
                if not source_path.exists():
                    error_msg = f"Source file not found: {source_path}"
                    logging.error(error_msg)
                    return error_msg
                
                if not os.access(source_path, os.R_OK):
                    error_msg = f"Source file not readable: {source_path}"
                    logging.error(error_msg)
                    return error_msg
                
                # Use proper context manager for file operations
                with open(source_path, 'rb') as src_file:
                    with open(target_path, 'wb') as dst_file:
                        dst_file.write(src_file.read())
                logging.info(f"File copied from {file.name} to {target_path}")
            except PermissionError as e:
                error_msg = f"Permission denied when copying file: {e}"
                logging.error(error_msg)
                return error_msg
            except IOError as e:
                error_msg = f"IO error when copying file: {e}"
                logging.error(error_msg)
                return error_msg
            except Exception as e:
                error_msg = f"Error copying file: {str(e)}"
                logging.error(error_msg)
                logging.exception("Detailed error information:")
                return error_msg
                
            # Verify file was copied successfully
            if target_path.exists():
                try:
                    file_size = target_path.stat().st_size
                    if file_size == 0:
                        error_msg = f"Error: File {target_path.name} is empty"
                        logging.error(error_msg)
                        # Clean up empty file
                        target_path.unlink()
                        return error_msg
                        
                    success_msg = (
                        f"File {target_path.name} uploaded successfully (size: {file_size:,} bytes).\n"
                        "Open 'Process' tab and click 'Process Documents' to process the file."
                    )
                    logging.info(f"File {target_path.name} uploaded successfully (size: {file_size} bytes)")
                    return success_msg
                except OSError as e:
                    error_msg = f"Error accessing file stats: {e}"
                    logging.error(error_msg)
                    return error_msg
            else:
                error_msg = f"File copy failed: {file.name} not found at target location"
                logging.error(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"File upload failed: {str(e)}"
            logging.error(error_msg)
            logging.exception("Detailed error information:")
            return error_msg

    def process_files_gradio(self):
        """
        Process documents through the Gradio interface.
        
        Returns:
            str: Message indicating success/failure and number of documents processed
        """
        try:
            logging.info("Starting document processing via Gradio interface")
            
            # Check for files to process
            to_process_dir = Path(self.config['directories']['to_process_dir'])
            
            # Ensure directory exists
            if not to_process_dir.exists():
                to_process_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created to_process directory: {to_process_dir}")
            
            # Find PDF files
            files = list(to_process_dir.glob("*.pdf"))
            logging.info(f"Found {len(files)} files in {to_process_dir}")
            
            if not files:
                return "No PDF files found in the to_process directory."
            
            # Use DataPrepService to handle document processing
            result = self.data_prep_service.process_documents_with_response()
            
            # Force garbage collection after processing
            import gc
            gc.collect()
            
            return result["message"]
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logging.error(error_msg)
            
            # Ensure cleanup even on error
            import gc
            gc.collect()
            
            return error_msg

    def build_interface(self):
        """
        Build and configure the Gradio interface.
        
        Returns:
            gr.Blocks: Configured Gradio interface
        """
        with gr.Blocks() as demo:
            gr.Markdown("# Data Preparation Pipeline")
            
            with gr.Tab("Upload"):
                file_input = gr.File(label="Upload PDF File")
                upload_button = gr.Button("Upload")
                upload_output = gr.Textbox(label="Upload Status")
                upload_button.click(
                    fn=self.upload_file_gradio,
                    inputs=file_input,
                    outputs=upload_output
                )
                
            with gr.Tab("Process"):
                process_button = gr.Button("Process Documents")
                process_output = gr.Textbox(label="Processing Status")
                process_button.click(
                    fn=self.process_files_gradio,
                    inputs=None,
                    outputs=process_output
                )
        
        return demo

    def launch(self, server_name="0.0.0.0", server_port=7860, share=False):
        """
        Launch the Gradio interface.
        
        Args:
            server_name (str): Host to bind to
            server_port (int): Port to run on
            share (bool): Whether to create a public link
        """
        demo = self.build_interface()
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share
        ) 