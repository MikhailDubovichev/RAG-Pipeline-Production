import os
import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from data_preparation.services.document_processor import DocumentProcessor
from llama_index.core import Document

class TestDocumentProcessor:
    """Test cases for the DocumentProcessor class."""
    
    def setup_method(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir_path = Path(self.temp_dir)
        
        # Create test directories
        self.input_dir = self.test_dir_path / "input"
        self.processed_dir = self.test_dir_path / "processed"
        self.log_dir = self.test_dir_path / "logs"
        self.data_dir = self.test_dir_path / "data"
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create a test PDF file
        self.test_pdf_path = self.input_dir / "test.pdf"
        with open(self.test_pdf_path, "wb") as f:
            f.write(b"%PDF-1.5\n")  # Minimal PDF header
        
        # Create the document processor
        self.document_processor = DocumentProcessor(max_tokens=1000, overlap_tokens=200)
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Close all loggers
        import logging
        for logger_name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        
        # Close root logger handlers
        for handler in list(logging.getLogger().handlers):
            handler.close()
            logging.getLogger().removeHandler(handler)
        
        # Small delay to ensure file handles are released
        time.sleep(0.1)
        
        # Clean up the temporary directory
        try:
            shutil.rmtree(self.temp_dir)
        except (PermissionError, OSError) as e:
            # If we can't remove the whole directory, try to remove files individually
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                    except:
                        pass
                for dir in dirs:
                    try:
                        os.rmdir(os.path.join(root, dir))
                    except:
                        pass
            try:
                os.rmdir(self.temp_dir)
            except:
                pass
    
    def test_init(self):
        """Test initialization of the DocumentProcessor."""
        assert self.document_processor.max_tokens == 1000
        assert self.document_processor.overlap_tokens == 200
    
    @patch('data_preparation.services.document_processor.PDFReader')
    def test_load_pdf_docs_empty_list(self, mock_pdf_reader):
        """Test handling of empty PDF list."""
        # Call the method with an empty list
        docs, failed = self.document_processor.load_pdf_docs([])
        
        # Check that the results are empty
        assert len(docs) == 0
        assert len(failed) == 0
    
    def test_load_pdf_docs_nonexistent_file(self):
        """Test handling of nonexistent PDF file."""
        # Create a path to a nonexistent file
        nonexistent_path = self.test_dir_path / "nonexistent.pdf"
        
        # Call the method with the nonexistent file
        docs, failed = self.document_processor.load_pdf_docs([nonexistent_path])
        
        # Check that the file is reported as failed
        assert len(docs) == 0
        assert len(failed) == 1
        assert failed[0][0] == nonexistent_path
        
        # Check that the error message indicates the file was not found
        error_message = failed[0][1].lower()
        assert any(phrase in error_message for phrase in ["not found", "cannot find", "no such file"])
    
    @patch('data_preparation.services.document_processor.PDFReader')
    def test_load_single_pdf(self, mock_pdf_reader_class):
        """Test loading a single PDF file."""
        # Set up the mock
        mock_pdf_reader = MagicMock()
        mock_pdf_reader.load_data.return_value = [
            Document(text="Test content", metadata={"source": str(self.test_pdf_path)})
        ]
        mock_pdf_reader_class.return_value = mock_pdf_reader
        
        # Set the mock on the document processor
        self.document_processor.pdf_reader = mock_pdf_reader
        
        # Call the method
        docs = self.document_processor._load_single_pdf(self.test_pdf_path)
        
        # Check that the document was loaded correctly
        assert len(docs) == 1
        assert docs[0].text == "Test content"
        
        # Check that the PDF reader was called with the correct arguments
        mock_pdf_reader.load_data.assert_called_once_with(str(self.test_pdf_path))
    
    def test_assign_doc_ids(self):
        """Test that document IDs are assigned correctly."""
        # Create test documents
        docs = [
            Document(text="Doc 1", metadata={}),
            Document(text="Doc 2", metadata={}),
            Document(text="Doc 3", metadata={})
        ]
        
        # Assign IDs
        result = self.document_processor._assign_doc_ids(docs)
        
        # Check that each document has a unique ID
        ids = [doc.metadata["doc_id"] for doc in result]
        assert len(ids) == len(set(ids))  # All IDs should be unique
        
        # Check that all IDs are valid UUIDs
        for doc_id in ids:
            assert len(doc_id) > 0
    
    def test_extract_metadata(self):
        """Test that metadata is extracted correctly."""
        # Create a test document
        doc = Document(text="Test content", metadata={})
        
        # Extract metadata
        result = self.document_processor._extract_metadata(doc, "test.pdf")
        
        # Check that the metadata was added
        assert result.metadata["source"] == "test.pdf"
    
    @patch('data_preparation.services.document_processor.PDFReader')
    def test_load_pdf_docs_successful(self, mock_pdf_reader_class):
        """Test successful loading of PDF documents."""
        # Set up the mock
        mock_pdf_reader = MagicMock()
        mock_pdf_reader.load_data.return_value = [
            Document(text="Test content", metadata={"source": str(self.test_pdf_path)})
        ]
        mock_pdf_reader_class.return_value = mock_pdf_reader
        
        # Set the mock on the document processor
        self.document_processor.pdf_reader = mock_pdf_reader
        
        # Call the method
        docs, failed = self.document_processor.load_pdf_docs([self.test_pdf_path])
        
        # Check that the document was loaded successfully
        assert len(docs) == 1
        assert len(failed) == 0
        
        # Check that the document has the expected metadata
        assert docs[0].metadata is not None
        assert "doc_id" in docs[0].metadata
        assert "source" in docs[0].metadata
        assert docs[0].metadata["source"] == "test.pdf"
    
    @patch('data_preparation.services.document_processor.PDFReader')
    def test_load_pdf_docs_exception(self, mock_pdf_reader_class):
        """Test handling of exceptions during PDF loading."""
        # Set up the mock to raise an exception
        mock_pdf_reader = MagicMock()
        mock_pdf_reader.load_data.side_effect = Exception("Test exception")
        mock_pdf_reader_class.return_value = mock_pdf_reader
        
        # Set the mock on the document processor
        self.document_processor.pdf_reader = mock_pdf_reader
        
        # Call the method
        docs, failed = self.document_processor.load_pdf_docs([self.test_pdf_path])
        
        # Check that the file is reported as failed
        assert len(docs) == 0
        assert len(failed) == 1
        assert failed[0][0] == self.test_pdf_path
        
        # Check that the error message indicates a failure
        error_message = failed[0][1].lower()
        assert "error" in error_message 