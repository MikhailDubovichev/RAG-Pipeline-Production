import json
import logging
import shutil
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self, to_process_dir: Path, processed_dir: Path):
        """
        Initialize file service with directory paths.
        
        Args:
            to_process_dir: Directory containing files to process
            processed_dir: Directory for processed files
        """
        self.to_process_dir = to_process_dir
        self.processed_dir = processed_dir
        self.supported_extensions = {".pdf", ".xls", ".xlsx", ".ppt", ".pptx", ".doc", ".docx"}

    def get_files_by_type(self) -> tuple[List[Path], List[Path], List[Path], List[Path]]:
        """Get lists of files by type from the to_process directory."""
        all_files = [f for f in self.to_process_dir.rglob("*") 
                    if f.is_file() and f.suffix.lower() in self.supported_extensions]

        pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]
        excel_files = [f for f in all_files if f.suffix.lower() in {".xls", ".xlsx"}]
        ppt_files = [f for f in all_files if f.suffix.lower() in {".ppt", ".pptx"}]
        doc_files = [f for f in all_files if f.suffix.lower() in {".doc", ".docx"}]

        return pdf_files, excel_files, ppt_files, doc_files

    def load_processed_files(self, record_path: Path) -> Set[str]:
        """Load the set of already processed file names."""
        if record_path.exists():
            with open(record_path, 'r') as f:
                processed = set(json.load(f))
            logger.info(f"Loaded processed files from {record_path}")
        else:
            processed = set()
            logger.info(f"No processed files record found at {record_path}")
        return processed

    def move_to_processed(self, files: List[Path]) -> None:
        """Move processed files to the processed directory."""
        for file in files:
            try:
                target_path = self.processed_dir / file.relative_to(self.to_process_dir)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(target_path))
                logger.info(f"Moved {file.name} to {target_path}")
            except Exception as e:
                logger.error(f"Failed to move file {file.name}: {e}")

    def update_processed_files_record(self, record_path: Path, file_list: List[Path]) -> None:
        """Update the JSON record of processed files."""
        try:
            processed = set()
            if record_path.exists():
                with open(record_path, 'r') as f:
                    processed = set(json.load(f))
            
            processed.update([file.name for file in file_list])
            
            with open(record_path, 'w') as f:
                json.dump(list(processed), f)
            logger.info(f"Updated processed files record at {record_path}")
            
        except Exception as e:
            logger.error(f"Error updating processed files record: {e}")

    def get_new_files(self, files: List[Path], processed_files: Set[str]) -> List[Path]:
        """Get list of new files that haven't been processed yet."""
        return [f for f in files if f.name not in processed_files] 