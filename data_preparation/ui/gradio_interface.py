"""
Gradio web interface implementation for the data preparation pipeline.

This module provides a user-friendly interface for:
1. Uploading documents (PDF, Excel, PowerPoint, Word)
2. Processing documents into chunks
3. Creating/updating search indices
"""

import logging
import gradio as gr
from pathlib import Path
import shutil
from datetime import datetime

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
        self.doc_processor = doc_processor
        self.indexing_service = indexing_service
        self.file_service = file_service
        self.config = config
        logging.info("GradioInterface initialized with required services")

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
            if file_ext not in ['.pdf', '.xlsx', '.xls', '.pptx', '.ppt', '.docx', '.doc']:
                error_msg = f"Unsupported file type: {file_ext}. Supported types: PDF, Excel, PowerPoint, Word"
                logging.error(error_msg)
                return error_msg
            
            # Get the to_process directory
            to_process_dir = Path(self.config['directories']['to_process_dir'])
            logging.info(f"Target directory: {to_process_dir}")
            
            # Create directory if needed
            to_process_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle duplicate filenames
            target_path = to_process_dir / Path(file.name).name
            if target_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_path = target_path.with_name(f"{target_path.stem}_{timestamp}{target_path.suffix}")
                logging.info(f"File already exists, using new path: {target_path}")

            # Copy file with metadata preservation
            try:
                shutil.copy2(file.name, target_path)
                logging.info(f"File copied from {file.name} to {target_path}")
            except Exception as e:
                error_msg = f"Error copying file: {str(e)}"
                logging.error(error_msg)
                return error_msg
                
            # Verify file was copied successfully
            if target_path.exists():
                file_size = target_path.stat().st_size
                if file_size == 0:
                    error_msg = f"Error: File {target_path.name} is empty"
                    logging.error(error_msg)
                    target_path.unlink()
                    return error_msg
                    
                success_msg = (
                    f"File {target_path.name} uploaded successfully (size: {file_size:,} bytes).\n"
                    "Click 'Process Documents' to process the file."
                )
                logging.info(f"File {target_path.name} uploaded successfully (size: {file_size} bytes)")
                return success_msg
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
            files = list(to_process_dir.glob("*.*"))
            logging.info(f"Found {len(files)} files in {to_process_dir}")
            
            if not files:
                return "No files found in the to_process directory."
            
            # Process the documents
            from data_preparation.main import process_documents
            num_processed = process_documents(
                self.doc_processor,
                self.indexing_service,
                self.file_service,
                self.config
            )
            
            result_msg = f"Processed {num_processed} new documents."
            logging.info(result_msg)
            return result_msg
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logging.error(error_msg)
            logging.exception("Detailed error information:")
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
                file_input = gr.File(label="Upload File")
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