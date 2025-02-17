import logging
from pathlib import Path
from typing import List, Dict

from llama_index.core import Document
from llama_index.readers.file import PDFReader
import pandas as pd
from docx import Document as WordDocument
from pptx import Presentation
import uuid

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, max_tokens: int = 1024, overlap_tokens: int = 50):
        """
        Initialize document processor with chunking parameters.
        
        Args:
            max_tokens: Maximum number of tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.pdf_reader = PDFReader()

    def load_pdf_docs(self, pdf_paths: List[Path]) -> List[Document]:
        """Load and process PDF documents."""
        all_docs = []
        for pdf_path in pdf_paths:
            try:
                docs = self.pdf_reader.load_data(pdf_path)
                docs = self._assign_doc_ids(docs)
                for doc in docs:
                    doc = self._extract_metadata(doc, source_file=pdf_path.name)
                all_docs.extend(docs)
                logger.info(f"Processed PDF: {pdf_path.name} with {len(docs)} pages.")
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf_path.name}: {e}")
        return all_docs

    def load_excel_docs(self, excel_paths: List[Path]) -> List[Document]:
        """Load and process Excel documents."""
        all_docs = []
        for excel_path in excel_paths:
            try:
                xls = pd.ExcelFile(excel_path)
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    for idx, row in df.iterrows():
                        row_dict = ", ".join(f"{col.strip().lower()}: {str(row[col]).strip()}" 
                                           for col in df.columns)
                        metadata = {
                            "sheet": sheet_name,
                            "row_number": idx + 1,
                        }
                        doc = Document(text=row_dict, metadata=metadata)
                        all_docs.append(doc)
                logger.info(f"Processed Excel: {excel_path.name} with {len(df)} rows.")
            except Exception as e:
                logger.error(f"Failed to process Excel {excel_path.name}: {e}")
        
        all_docs = self._assign_doc_ids(all_docs)
        for doc in all_docs:
            doc = self._extract_metadata(doc, source_file=excel_path.name)
        return all_docs

    def load_ppt_docs(self, ppt_paths: List[Path]) -> List[Document]:
        """Load and process PowerPoint documents."""
        all_docs = []
        for ppt_path in ppt_paths:
            try:
                prs = Presentation(ppt_path)
                for idx, slide in enumerate(prs.slides):
                    slide_text = "\n".join(shape.text for shape in slide.shapes 
                                         if hasattr(shape, "text"))
                    metadata = {
                        "slide_number": idx + 1,
                    }
                    doc = Document(text=slide_text, metadata=metadata)
                    all_docs.append(doc)
                logger.info(f"Processed PowerPoint: {ppt_path.name} with {len(prs.slides)} slides.")
            except Exception as e:
                logger.error(f"Failed to process PowerPoint {ppt_path.name}: {e}")
        
        all_docs = self._assign_doc_ids(all_docs)
        for doc in all_docs:
            doc = self._extract_metadata(doc, source_file=ppt_path.name)
        return all_docs

    def load_doc_docs(self, doc_paths: List[Path]) -> List[Document]:
        """Load and process Word documents."""
        all_docs = []
        for doc_path in doc_paths:
            try:
                word_doc = WordDocument(doc_path)
                docs = self._chunk_by_headings(word_doc)
                for doc in docs:
                    doc = self._extract_metadata(doc, source_file=doc_path.name)
                all_docs.extend(docs)
                logger.info(f"Processed Word Document: {doc_path.name} with {len(docs)} chunks.")
            except Exception as e:
                logger.error(f"Failed to process Word Document {doc_path.name}: {e}")
        return all_docs

    def _chunk_by_headings(self, word_doc: WordDocument) -> List[Document]:
        """Chunk Word documents based on headings."""
        current_section = []
        current_metadata = None
        all_docs = []

        for paragraph in word_doc.paragraphs:
            if paragraph.style.name.startswith("Heading"):
                if current_section:
                    docs = self._split_large_section(current_section, current_metadata)
                    all_docs.extend(docs)
                    current_section = []
                current_metadata = {"section": paragraph.text.strip()}
            current_section.append(paragraph.text.strip())

        if current_section:
            docs = self._split_large_section(current_section, current_metadata)
            all_docs.extend(docs)

        return self._assign_doc_ids(all_docs)

    def _split_large_section(self, section: List[str], metadata: Dict) -> List[Document]:
        """Split large text sections into manageable chunks."""
        text = "\n".join(section)
        chunk_size = self.max_tokens
        overlap = self.overlap_tokens
        
        # Note: In a real implementation, you would use the tokenizer here
        # For now, we'll use a simple character-based approach
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                end = text.rfind(" ", start, end)
            chunk = text[start:end]
            
            chunk_metadata = {
                **(metadata or {}),
                "chunk_number": len(chunks) + 1,
                "doc_id": str(uuid.uuid4()),
            }
            
            chunks.append(Document(text=chunk, metadata=chunk_metadata))
            start = end - overlap if end < len(text) else len(text)
        
        return chunks

    @staticmethod
    def _assign_doc_ids(documents: List[Document]) -> List[Document]:
        """Assign unique UUIDs to documents."""
        for doc in documents:
            doc.metadata["doc_id"] = str(uuid.uuid4())
        return documents

    @staticmethod
    def _extract_metadata(doc: Document, source_file: str) -> Document:
        """Extract and assign metadata to a document."""
        doc.metadata["source"] = source_file
        return doc 