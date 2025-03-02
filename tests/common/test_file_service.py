import os
import tempfile
import shutil
import time
import json
from unittest.mock import MagicMock, patch
import pytest
from pathlib import Path

from data_preparation.services.file_service import FileService

class TestFileService:
    """Test cases for the FileService class."""
    
    def setup_method(self):
        """Set up test environment before each test method."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create subdirectories for to_process and processed
        self.to_process_dir = Path(self.temp_dir) / "to_process"
        self.processed_dir = Path(self.temp_dir) / "processed"
        
        # Create the directories
        self.to_process_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Create the file service
        self.file_service = FileService(to_process_dir=self.to_process_dir, processed_dir=self.processed_dir)
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Close any open file handlers
        if hasattr(self.file_service, 'logger'):
            for handler in list(self.file_service.logger.handlers):
                handler.close()
                self.file_service.logger.removeHandler(handler)
        
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
    
    def test_get_files_by_type(self):
        """Test getting PDF files from the to_process directory."""
        # Create some test PDF files
        pdf_file1 = self.to_process_dir / "file1.pdf"
        pdf_file2 = self.to_process_dir / "file2.pdf"
        txt_file = self.to_process_dir / "file3.txt"
        
        # Create subdirectory with PDF file
        subdir = self.to_process_dir / "subdir"
        subdir.mkdir(exist_ok=True)
        pdf_file3 = subdir / "file4.pdf"
        
        # Create the files
        pdf_file1.touch()
        pdf_file2.touch()
        txt_file.touch()
        pdf_file3.touch()
        
        # Call the method
        pdf_files = self.file_service.get_files_by_type()
        
        # Check that only PDF files were returned
        assert len(pdf_files) == 3
        assert pdf_file1 in pdf_files
        assert pdf_file2 in pdf_files
        assert pdf_file3 in pdf_files
        assert txt_file not in pdf_files
    
    def test_load_processed_files(self):
        """Test loading processed files from a record file."""
        # Create a test record file
        record_path = Path(self.temp_dir) / "processed_files.json"
        processed_files = ["file1.pdf", "file2.pdf"]
        
        with open(record_path, "w") as f:
            json.dump(processed_files, f)
        
        # Call the method
        loaded_files, error_msg = self.file_service.load_processed_files(record_path)
        
        # Check that the files were loaded correctly
        assert error_msg == ""
        assert loaded_files == set(processed_files)
    
    def test_load_processed_files_nonexistent(self):
        """Test loading processed files from a non-existent record file."""
        # Define a non-existent record file path
        record_path = Path(self.temp_dir) / "nonexistent.json"
        
        # Call the method
        loaded_files, error_msg = self.file_service.load_processed_files(record_path)
        
        # Check that an empty set was returned
        assert error_msg == ""
        assert loaded_files == set()
    
    def test_move_to_processed(self):
        """Test moving files to the processed directory."""
        # Create some test files
        file1 = self.to_process_dir / "file1.pdf"
        file1.touch()
        
        subdir = self.to_process_dir / "subdir"
        subdir.mkdir(exist_ok=True)
        file2 = subdir / "file2.pdf"
        file2.touch()
        
        # Call the method
        moved_files, failed_moves = self.file_service.move_to_processed([file1, file2])
        
        # Check that the files were moved correctly
        assert len(moved_files) == 2
        assert len(failed_moves) == 0
        
        # Check that the files exist in the processed directory
        assert (self.processed_dir / "file1.pdf").exists()
        assert (self.processed_dir / "subdir" / "file2.pdf").exists()
        
        # Check that the original files no longer exist
        assert not file1.exists()
        assert not file2.exists()
    
    def test_update_processed_files_record(self):
        """Test updating the processed files record."""
        # Create a test record file
        record_path = Path(self.temp_dir) / "processed_files.json"
        existing_files = ["existing1.pdf", "existing2.pdf"]
        
        with open(record_path, "w") as f:
            json.dump(existing_files, f)
        
        # Create some test files to add to the record
        file1 = Path(self.temp_dir) / "new1.pdf"
        file2 = Path(self.temp_dir) / "new2.pdf"
        
        # Call the method
        success, error_msg = self.file_service.update_processed_files_record(record_path, [file1, file2])
        
        # Check that the record was updated successfully
        assert success
        assert error_msg == ""
        
        # Check that the record file contains all files
        with open(record_path, "r") as f:
            updated_files = set(json.load(f))
        
        assert updated_files == set(existing_files + ["new1.pdf", "new2.pdf"])
    
    def test_get_new_files(self):
        """Test filtering out already processed files."""
        # Create some test files
        file1 = self.to_process_dir / "file1.pdf"
        file2 = self.to_process_dir / "file2.pdf"
        file3 = self.to_process_dir / "file3.pdf"
        
        file1.touch()
        file2.touch()
        file3.touch()
        
        # Define a set of already processed files
        processed_files = {"file1.pdf", "file2.pdf"}
        
        # Call the method
        new_files = self.file_service.get_new_files([file1, file2, file3], processed_files)
        
        # Check that only the unprocessed file was returned
        assert len(new_files) == 1
        assert file3 in new_files
        assert file1 not in new_files
        assert file2 not in new_files 