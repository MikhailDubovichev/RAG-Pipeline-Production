"""
Unit tests for the DocumentProcessor class.

This module tests the functionality of the DocumentProcessor class, which is responsible for:
1. Loading PDF documents
2. Extracting text content
3. Chunking documents into manageable pieces
4. Assigning metadata and IDs

The tests use pytest and include:
- Basic functionality tests
- Edge case handling
- Error handling
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the class to test
from data_preparation.services import DocumentProcessor
from llama_index.core import Document

class TestDocumentProcessor:
    """Test suite for the DocumentProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a document processor with default settings
        self.processor = DocumentProcessor(max_tokens=500, overlap_tokens=50)
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = Path(self.temp_dir.name)
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up the temporary directory
        self.temp_dir.cleanup()
    
    @pytest.mark.parametrize("max_tokens,overlap_tokens", [
        (500, 50),  # Default values
        (1000, 100),  # Larger chunks
        (200, 20),  # Smaller chunks
    ])
    def test_init(self, max_tokens, overlap_tokens):
        """Test that the DocumentProcessor initializes with correct parameters."""
        processor = DocumentProcessor(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        # Check that the processor was initialized with the correct parameters
        assert processor.max_tokens == max_tokens
        assert processor.overlap_tokens == overlap_tokens
    
    @patch('data_preparation.services.document_processor.PDFReader')
    def test_load_pdf_docs_empty_list(self, mock_pdf_reader):
        """Test handling of empty PDF list."""
        # Call the method with an empty list
        docs, failed = self.processor.load_pdf_docs([])
        
        # Check that the results are empty
        assert len(docs) == 0
        assert len(failed) == 0
        
        # Verify that the PDF reader was not called
        mock_pdf_reader.return_value.load_data.assert_not_called()
    
    @patch('data_preparation.services.document_processor.PDFReader')
    def test_load_pdf_docs_nonexistent_file(self, mock_pdf_reader):
        """Test handling of nonexistent PDF file."""
        # Create a path to a nonexistent file
        nonexistent_path = self.test_dir_path / "nonexistent.pdf"
        
        # Call the method with the nonexistent file
        docs, failed = self.processor.load_pdf_docs([nonexistent_path])
        
        # Check that the file is reported as failed
        assert len(docs) == 0
        assert len(failed) == 1
        assert failed[0][0] == nonexistent_path
        
        # Check that the error message indicates the file was not found
        # The exact wording might vary by OS, so we check for common phrases
        error_message = failed[0][1].lower()
        assert any(phrase in error_message for phrase in ["not found", "cannot find", "no such file"])
        
        # Verify that the PDF reader was not called
        mock_pdf_reader.return_value.load_data.assert_not_called()
    
    @patch('data_preparation.services.document_processor.PDFReader')
    def test_load_pdf_docs_successful(self, mock_pdf_reader):
        """Test successful loading of PDF documents."""
        # Create a mock document
        mock_doc = Document(text="Test content", metadata={"source": "test.pdf"})
        mock_pdf_reader.return_value.load_data.return_value = [mock_doc]
        
        # Create a test PDF file with some content
        test_pdf_path = self.test_dir_path / "test.pdf"
        with open(test_pdf_path, 'wb') as f:
            # Write some binary data to make it a non-empty file
            f.write(b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF')
        
        # Patch the _load_single_pdf method to return our mock document
        with patch.object(self.processor, '_load_single_pdf', return_value=[mock_doc]):
            # Call the method with the test file
            docs, failed = self.processor.load_pdf_docs([test_pdf_path])
            
            # Check that the document was loaded successfully
            assert len(docs) == 1
            assert len(failed) == 0
            
            # Check that the document has the expected metadata
            assert docs[0].metadata is not None
            assert "doc_id" in docs[0].metadata
            assert "source" in docs[0].metadata
    
    @patch('data_preparation.services.document_processor.PDFReader')
    def test_load_pdf_docs_exception(self, mock_pdf_reader):
        """Test handling of exceptions during PDF loading."""
        # Configure the mock to raise an exception
        mock_pdf_reader.return_value.load_data.side_effect = Exception("Test exception")
        
        # Create a test PDF file with some content
        test_pdf_path = self.test_dir_path / "test.pdf"
        with open(test_pdf_path, 'wb') as f:
            # Write some binary data to make it a non-empty file
            f.write(b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF')
        
        # Patch the _load_single_pdf method to raise our exception
        with patch.object(self.processor, '_load_single_pdf', side_effect=Exception("Test exception")):
            # Call the method with the test file
            docs, failed = self.processor.load_pdf_docs([test_pdf_path])
            
            # Check that the file is reported as failed
            assert len(docs) == 0
            assert len(failed) == 1
            assert failed[0][0] == test_pdf_path
            
            # Check that the error message indicates a failure
            error_message = failed[0][1].lower()
            assert any(word in error_message for word in ["error", "fail", "exception"])
    
    def test_assign_doc_ids(self):
        """Test that document IDs are assigned correctly."""
        # Create test documents
        docs = [
            Document(text="Doc 1", metadata={}),
            Document(text="Doc 2", metadata={}),
            Document(text="Doc 3", metadata={})
        ]
        
        # Assign IDs
        result = self.processor._assign_doc_ids(docs)
        
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
        result = self.processor._extract_metadata(doc, "test.pdf")
        
        # Check that the metadata was added
        assert result.metadata["source"] == "test.pdf" 