import json
import logging
import shutil  # shutil stands for shell utilities, provides high-level operations on files
from pathlib import Path  # pathlib provides an object-oriented interface to filesystem paths
from typing import List, Set  # These are type hints that help with code clarity and IDE support

logger = logging.getLogger(__name__)

class FileService:
    """
    A service class that handles all file-related operations in the data preparation pipeline.
    This class is responsible for:
    1. Finding PDF files in a source directory
    2. Tracking which files have been processed
    3. Moving processed files to a designated directory
    4. Maintaining a record of processed files
    
    The class follows the principle of separation of concerns by handling only file operations,
    making the code more maintainable and easier to test.
    """

    def __init__(self, to_process_dir: Path, processed_dir: Path):
        """
        Initialize the FileService with source and destination directories.
        
        Detailed explanation:
        - to_process_dir: The source directory where original files are stored waiting to be processed
        - processed_dir: The destination directory where files will be moved after processing
        - supported_extensions: A set of file types that our pipeline can handle
        
        Args:
            to_process_dir (Path): Directory containing files that need to be processed
            processed_dir (Path): Directory where processed files will be stored
        """
        self.to_process_dir = to_process_dir
        self.processed_dir = processed_dir
        self.supported_extensions = {".pdf"}

    def get_files_by_type(self) -> List[Path]:
        """
        Scans the to_process directory and finds all PDF files.
        
        Process:
        1. Checks if the source directory exists
        2. Recursively finds all PDF files (using rglob - recursive global pattern matching)
        
        Returns:
            List[Path]: List of paths to PDF files
            
        If any error occurs or the directory doesn't exist, returns empty list.
        """
        try:
            # Log the start of file discovery
            logger.info(f"Scanning directory {self.to_process_dir} for PDF files")
            
            # Ensure the directory exists
            if not self.to_process_dir.exists():
                logger.error(f"Directory not found: {self.to_process_dir}")
                return []
                
            # Get all PDF files
            pdf_files = [f for f in self.to_process_dir.rglob("*") 
                        if f.is_file() and f.suffix.lower() == ".pdf"]
            
            # Log the total number of files found
            logger.info(f"Found {len(pdf_files)} PDF files")

            return pdf_files
            
        except Exception as e:
            logger.error(f"Error scanning directory {self.to_process_dir}: {str(e)}")
            logger.exception("Detailed error information:")
            return []

    def load_processed_files(self, record_path: Path) -> tuple[set[str], str]:
        """
        Reads and returns a set of filenames that have already been processed.
        
        Process:
        1. Checks if a record file exists at the specified path
        2. If it exists, loads the JSON data into a set of filenames
        3. If it doesn't exist, returns an empty set
        
        The record is stored as a JSON file to maintain persistence between program runs
        and to make it human-readable if needed.
        
        Args:
            record_path (Path): Path to the JSON file that stores processed filenames
            
        Returns:
            tuple: (set[str], str) - A set of processed filenames and error message if any
                - First element is a set of filenames that have already been processed
                - Second element is an error message if loading failed, empty string otherwise
        """
        processed = set()
        error_msg = ""
        
        try:
            if record_path.exists():
                try:
                    with open(record_path, 'r') as f:
                        processed = set(json.load(f))
                    logger.info(f"Loaded processed files from {record_path}")
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON format in record file: {e}"
                    logger.error(error_msg)
                except PermissionError as e:
                    error_msg = f"Permission denied when reading record file: {e}"
                    logger.error(error_msg)
                except Exception as e:
                    error_msg = f"Error reading record file: {e}"
                    logger.error(error_msg)
                    logger.exception("Detailed error information:")
            else:
                logger.info(f"No processed files record found at {record_path}")
        except Exception as e:
            error_msg = f"Unexpected error accessing record file: {e}"
            logger.error(error_msg)
            logger.exception("Detailed error information:")
            
        return processed, error_msg

    def move_to_processed(self, files: List[Path]) -> tuple[List[Path], List[tuple[Path, str]]]:
        """
        Moves processed files to their new location in the processed directory.
        
        Process:
        1. For each file:
           - Calculates the target path while preserving the original directory structure
           - Creates necessary subdirectories
           - Moves the file using shutil.move
           - Verifies the move was successful
        
        The method maintains the original directory structure in the processed directory.
        For example, if a file was in 'to_process/subfolder/file.pdf',
        it will be moved to 'processed/subfolder/file.pdf'
        
        Error handling:
        - Logs errors if source file doesn't exist
        - Warns if target file already exists (will be overwritten)
        - Catches and logs any exceptions during the move operation
        
        Args:
            files (List[Path]): List of file paths to move
            
        Returns:
            tuple: (List[Path], List[tuple[Path, str]]) - Successful and failed moves
                - First element is a list of successfully moved files
                - Second element is a list of tuples containing (file_path, error_message) for failed moves
        """
        if not files:
            logger.info("No files to move to processed directory")
            return [], []
            
        logger.info(f"Moving {len(files)} files to processed directory")
        
        successful_moves = []
        failed_moves = []
        
        for file in files:
            try:
                # Calculate target path
                target_path = self.processed_dir / file.relative_to(self.to_process_dir)
                logger.info(f"Moving {file.name} to {target_path}")
                
                # Ensure target directory exists
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                except PermissionError as e:
                    error_msg = f"Permission denied when creating directory {target_path.parent}: {e}"
                    logger.error(error_msg)
                    failed_moves.append((file, error_msg))
                    continue
                except OSError as e:
                    error_msg = f"OS error when creating directory {target_path.parent}: {e}"
                    logger.error(error_msg)
                    failed_moves.append((file, error_msg))
                    continue
                    
                # Check if source file exists
                if not file.exists():
                    error_msg = f"Source file not found: {file}"
                    logger.error(error_msg)
                    failed_moves.append((file, error_msg))
                    continue
                    
                # Check if target file already exists
                if target_path.exists():
                    logger.warning(f"Target file already exists, will be overwritten: {target_path}")
                
                # Move the file
                try:
                    shutil.move(str(file), str(target_path))
                    
                    # Verify the move
                    if target_path.exists():
                        logger.info(f"Successfully moved {file.name} to {target_path}")
                        successful_moves.append(target_path)
                    else:
                        error_msg = f"Failed to verify moved file at {target_path}"
                        logger.error(error_msg)
                        failed_moves.append((file, error_msg))
                except PermissionError as e:
                    error_msg = f"Permission denied when moving file: {e}"
                    logger.error(error_msg)
                    failed_moves.append((file, error_msg))
                except shutil.Error as e:
                    error_msg = f"Shutil error when moving file: {e}"
                    logger.error(error_msg)
                    failed_moves.append((file, error_msg))
                    
            except Exception as e:
                error_msg = f"Failed to move file {file.name}: {str(e)}"
                logger.error(error_msg)
                logger.exception("Detailed error information:")
                failed_moves.append((file, error_msg))
                
        logger.info(f"Successfully moved {len(successful_moves)} files, failed to move {len(failed_moves)} files")
        return successful_moves, failed_moves

    def update_processed_files_record(self, record_path: Path, file_list: List[Path]) -> tuple[bool, str]:
        """
        Updates the JSON record file with newly processed files.
        
        Process:
        1. Loads existing record if it exists
        2. Adds new filenames to the set of processed files
        3. Saves the updated set back to the JSON file
        
        This record helps track which files have been processed across multiple runs
        of the pipeline, preventing duplicate processing.
        
        Args:
            record_path (Path): Path to the JSON record file
            file_list (List[Path]): List of newly processed files to add to the record
            
        Returns:
            tuple: (bool, str) - Success status and error message if any
                - First element is True if update was successful, False otherwise
                - Second element is an error message if update failed, empty string otherwise
        """
        if not file_list:
            logger.info("No files to add to processed record")
            return True, ""
            
        try:
            processed = set()
            if record_path.exists():
                try:
                    with open(record_path, 'r') as f:
                        processed = set(json.load(f))
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON format in existing record file: {e}"
                    logger.error(error_msg)
                    return False, error_msg
                except PermissionError as e:
                    error_msg = f"Permission denied when reading record file: {e}"
                    logger.error(error_msg)
                    return False, error_msg
                except Exception as e:
                    error_msg = f"Error reading existing record file: {e}"
                    logger.error(error_msg)
                    logger.exception("Detailed error information:")
                    return False, error_msg
            
            # Ensure record directory exists
            record_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update the processed files set
            processed.update([file.name for file in file_list])
            
            try:
                with open(record_path, 'w') as f:
                    json.dump(list(processed), f)
                logger.info(f"Updated processed files record at {record_path}")
                return True, ""
            except PermissionError as e:
                error_msg = f"Permission denied when writing record file: {e}"
                logger.error(error_msg)
                return False, error_msg
            except Exception as e:
                error_msg = f"Error writing record file: {e}"
                logger.error(error_msg)
                logger.exception("Detailed error information:")
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Unexpected error updating processed files record: {e}"
            logger.error(error_msg)
            logger.exception("Detailed error information:")
            return False, error_msg

    def get_new_files(self, files: List[Path], processed_files: Set[str]) -> List[Path]:
        """
        Filters out already processed files from a list of files.
        
        Process:
        Compares each filename against the set of processed files
        and returns only those that haven't been processed yet.
        
        This method is crucial for incremental processing, ensuring that:
        1. We don't waste resources processing the same file multiple times
        2. We can add new files to the to_process directory at any time
        
        Args:
            files (List[Path]): List of all files found
            processed_files (Set[str]): Set of filenames that have already been processed
            
        Returns:
            List[Path]: List of files that haven't been processed yet
        """
        return [f for f in files if f.name not in processed_files] 