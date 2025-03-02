"""
Unit tests for the FileService class.

This module tests the functionality of the FileService class, which is responsible for:
1. Finding PDF files in a directory
2. Tracking processed files
3. Moving files after processing
4. Maintaining a record of processed files

The tests use pytest and include:
- Directory scanning functionality
- File movement operations
- Record keeping
- Error handling
"""

import os
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open

# Import the class to test
from data_preparation.services import FileService

class TestFileService:
    """Test suite for the FileService class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        
        # Create source and destination directories
        self.source_dir = self.base_path / "to_process"
        self.dest_dir = self.base_path / "processed"
        self.source_dir.mkdir()
        self.dest_dir.mkdir()
        
        # Initialize the file service
        self.file_service = FileService(
            to_process_dir=self.source_dir,
            processed_dir=self.dest_dir
        )
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up the temporary directory
        self.temp_dir.cleanup()
    
    def test_init(self):
        """Test that the FileService initializes with correct parameters."""
        # Check that the directories were set correctly
        assert self.file_service.to_process_dir == self.source_dir
        assert self.file_service.processed_dir == self.dest_dir
        assert ".pdf" in self.file_service.supported_extensions
    
    def test_get_files_by_type_empty_dir(self):
        """Test finding files in an empty directory."""
        # Call the method on an empty directory
        files = self.file_service.get_files_by_type()
        
        # Check that no files were found
        assert len(files) == 0
    
    def test_get_files_by_type_nonexistent_dir(self):
        """Test handling of nonexistent directory."""
        # Create a file service with a nonexistent directory
        nonexistent_dir = self.base_path / "nonexistent"
        file_service = FileService(
            to_process_dir=nonexistent_dir,
            processed_dir=self.dest_dir
        )
        
        # Call the method
        files = file_service.get_files_by_type()
        
        # Check that no files were found and no exception was raised
        assert len(files) == 0
    
    def test_get_files_by_type_with_files(self):
        """Test finding PDF files in a directory with mixed file types."""
        # Create test files of different types
        (self.source_dir / "test1.pdf").touch()
        (self.source_dir / "test2.pdf").touch()
        (self.source_dir / "test3.txt").touch()
        (self.source_dir / "test4.docx").touch()
        
        # Create a subdirectory with more files
        subdir = self.source_dir / "subdir"
        subdir.mkdir()
        (subdir / "test5.pdf").touch()
        (subdir / "test6.txt").touch()
        
        # Call the method
        files = self.file_service.get_files_by_type()
        
        # Check that only PDF files were found (including in subdirectory)
        assert len(files) == 3
        file_names = [f.name for f in files]
        assert "test1.pdf" in file_names
        assert "test2.pdf" in file_names
        assert "test5.pdf" in file_names
        assert "test3.txt" not in file_names
        assert "test4.docx" not in file_names
        assert "test6.txt" not in file_names
    
    def test_load_processed_files_nonexistent(self):
        """Test loading processed files when the record file doesn't exist."""
        # Create a path to a nonexistent record file
        record_path = self.base_path / "processed_files.json"
        
        # Call the method
        processed_files, error = self.file_service.load_processed_files(record_path)
        
        # Check that an empty set was returned with no error
        assert len(processed_files) == 0
        assert error == ""
    
    @patch("builtins.open", new_callable=mock_open, read_data='["file1.pdf", "file2.pdf"]')
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_processed_files_existing(self, mock_exists, mock_file):
        """Test loading processed files from an existing record file."""
        # Create a path to a record file
        record_path = self.base_path / "processed_files.json"
        
        # Call the method
        processed_files, error = self.file_service.load_processed_files(record_path)
        
        # Check that the files were loaded correctly
        assert len(processed_files) == 2
        assert "file1.pdf" in processed_files
        assert "file2.pdf" in processed_files
        assert error == ""
        
        # Verify that the file was opened
        mock_file.assert_called_once_with(record_path, "r")
    
    @patch("builtins.open", side_effect=Exception("Test exception"))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_processed_files_error(self, mock_exists, mock_file):
        """Test handling of errors when loading processed files."""
        # Create a path to a record file
        record_path = self.base_path / "processed_files.json"
        
        # Call the method
        processed_files, error = self.file_service.load_processed_files(record_path)
        
        # Check that an empty set was returned with an error message
        assert len(processed_files) == 0
        assert "test exception" in error.lower()
    
    def test_move_to_processed(self):
        """Test moving files to the processed directory."""
        # Create test files
        file1 = self.source_dir / "test1.pdf"
        file2 = self.source_dir / "test2.pdf"
        file1.touch()
        file2.touch()
        
        # Call the method
        moved, failed = self.file_service.move_to_processed([file1, file2])
        
        # Check that the files were moved successfully
        assert len(moved) == 2
        assert len(failed) == 0
        assert not file1.exists()
        assert not file2.exists()
        assert (self.dest_dir / "test1.pdf").exists()
        assert (self.dest_dir / "test2.pdf").exists()
    
    def test_move_to_processed_nonexistent(self):
        """Test handling of nonexistent files when moving."""
        # Create a path to a nonexistent file
        nonexistent_file = self.source_dir / "nonexistent.pdf"
        
        # Call the method
        moved, failed = self.file_service.move_to_processed([nonexistent_file])
        
        # Check that the file was reported as failed
        assert len(moved) == 0
        assert len(failed) == 1
        assert failed[0][0] == nonexistent_file
    
    def test_get_new_files(self):
        """Test filtering out already processed files."""
        # Create test files
        file1 = self.source_dir / "test1.pdf"
        file2 = self.source_dir / "test2.pdf"
        file3 = self.source_dir / "test3.pdf"
        file1.touch()
        file2.touch()
        file3.touch()
        
        # Create a set of processed files
        processed_files = {"test1.pdf", "test2.pdf"}
        
        # Call the method
        new_files = self.file_service.get_new_files([file1, file2, file3], processed_files)
        
        # Check that only the new file was returned
        assert len(new_files) == 1
        assert new_files[0] == file3 