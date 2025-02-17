import logging
from pathlib import Path
from typing import List
import os

import faiss
from llama_index.core import Document, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, NUMERIC

logger = logging.getLogger(__name__)

class IndexingService:
    def __init__(self, embedding_model):
        """
        Initialize indexing service with embedding model.
        
        Args:
            embedding_model: Model to generate embeddings (e.g., NebiusEmbedding)
        """
        self.embedding_model = embedding_model
        self.embedding_dimension = self._get_embedding_dimension()

    def _get_embedding_dimension(self) -> int:
        """Determine embedding dimension from the model."""
        sample_text = "test text to create an embedding"
        sample_embedding = self.embedding_model.get_text_embedding(sample_text)
        return len(sample_embedding)

    def create_or_update_whoosh_index(self, documents: List[Document], index_dir: Path) -> None:
        """Create or update Whoosh index with documents."""
        schema = Schema(
            content=TEXT(stored=True),
            source=ID(stored=True),
            sheet=ID(stored=True),
            row_number=NUMERIC(stored=True, sortable=True),
            slide_number=NUMERIC(stored=True, sortable=True),
            section=TEXT(stored=True),
            chunk_number=NUMERIC(stored=True, sortable=True),
            total_chunks_in_section=NUMERIC(stored=True, sortable=True),
            page_number=NUMERIC(stored=True, sortable=True),
            doc_id=ID(stored=True, unique=True),
        )

        try:
            os.makedirs(index_dir, exist_ok=True)

            if not exists_in(index_dir):
                idx = create_in(index_dir, schema)
                logger.info(f"Created new Whoosh index at {index_dir}")
            else:
                idx = open_dir(index_dir)
                logger.info(f"Opened existing Whoosh index at {index_dir}")

            writer = idx.writer()

            for doc in documents:
                try:
                    doc_fields = {
                        "content": doc.text,
                        "source": doc.metadata.get("source", "unknown_source"),
                        "doc_id": doc.metadata.get("doc_id", ""),
                    }

                    # Add optional metadata fields if they exist
                    for field in ["sheet", "row_number", "slide_number", "section",
                                "chunk_number", "total_chunks_in_section", "page_number"]:
                        if field in doc.metadata:
                            doc_fields[field] = doc.metadata[field]

                    writer.update_document(**doc_fields)
                except Exception as e:
                    logger.error(f"Failed to index document {doc.metadata.get('doc_id', '')}: {e}")

            writer.commit()
            logger.info("Whoosh index updated successfully")

        except Exception as e:
            logger.error(f"Failed to create/update Whoosh index: {e}")
            raise

    def create_or_update_faiss_index(self, documents: List[Document], index_path: Path) -> None:
        """Create or update FAISS index with documents."""
        try:
            if (index_path / "vector_store.faiss").exists():
                # Load existing index
                vector_store = FaissVectorStore.from_persist_path(str(index_path / "vector_store.faiss"))
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=str(index_path)
                )
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    embedding=self.embedding_model
                )
                logger.info("Updated existing FAISS index")
            else:
                # Create new index
                vector_store = FaissVectorStore(faiss.IndexFlatIP(self.embedding_dimension))
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    docstore=SimpleDocumentStore(),
                    index_store=SimpleIndexStore(),
                    persist_dir=str(index_path),
                )
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    embedding=self.embedding_model
                )
                logger.info("Created new FAISS index")

            # Persist the index
            index.storage_context.persist(persist_dir=str(index_path))
            logger.info(f"FAISS index persisted at {index_path}")

        except Exception as e:
            logger.error(f"Failed to create/update FAISS index: {e}")
            raise 