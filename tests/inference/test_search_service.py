"""
Unit tests for the SearchService class.

This module tests the functionality of the SearchService class, which is responsible for:
1. Performing semantic search using FAISS
2. Performing keyword search using Whoosh
3. Combining and reranking search results
4. Filtering results based on relevance thresholds

The tests use pytest and include:
- Vector search functionality
- Keyword search functionality
- Result combination and reranking
- Error handling
"""

import os
import pytest
import tempfile
import shutil
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import unittest

from inference.services.search_service import SearchService
from common.config_utils import load_config, setup_directories

class TestSearchService(unittest.TestCase):
    """Test cases for the SearchService class."""

    def setup_method(self, method):
        """Set up test fixtures.
        
        Args:
            method: The test method being run
        """
        # Create mock objects for the required components
        self.mock_faiss_index = MagicMock()
        self.mock_whoosh_index = MagicMock()
        self.mock_cross_encoder = MagicMock()

        # Initialize the SearchService with mock objects
        self.search_service = SearchService(
            faiss_index=self.mock_faiss_index,
            whoosh_index=self.mock_whoosh_index,
            cross_encoder=self.mock_cross_encoder
        )

    def test_init(self):
        """Test initialization of SearchService."""
        # Assert that the service has the correct attributes
        assert self.search_service.faiss_index == self.mock_faiss_index
        assert self.search_service.whoosh_index == self.mock_whoosh_index
        assert self.search_service.cross_encoder == self.mock_cross_encoder

    def test_perform_vector_search(self):
        """Test performing vector search."""
        # Set up the mock retriever
        mock_retriever = MagicMock()
        self.mock_faiss_index.as_retriever.return_value = mock_retriever
        
        # Create mock results
        mock_result1 = MagicMock()
        mock_result1.node.text = "This is test chunk 1"
        mock_result1.node.metadata = {"source": "test.pdf", "page_number": 1}
        mock_result1.score = 0.8
        
        mock_result2 = MagicMock()
        mock_result2.node.text = "This is test chunk 2"
        mock_result2.node.metadata = {"source": "test.pdf", "page_number": 2}
        mock_result2.score = 0.6
        
        mock_retriever.retrieve.return_value = [mock_result1, mock_result2]

        # Call the method
        results = self.search_service.perform_vector_search("test query", top_k=10)

        # Check that the search returned the correct results
        assert len(results) == 2
        assert results[0][0] == "This is test chunk 1"
        assert results[0][1] == 0.8
        assert results[0][2] == {"source": "test.pdf", "page_number": 1}
        assert results[1][0] == "This is test chunk 2"
        assert results[1][1] == 0.6
        assert results[1][2] == {"source": "test.pdf", "page_number": 2}
        
        # Check that the retriever was called with the correct arguments
        self.mock_faiss_index.as_retriever.assert_called_once_with(similarity_top_k=10)
        mock_retriever.retrieve.assert_called_once_with("test query")

    def test_perform_bm25_search(self):
        """Test performing BM25 search."""
        # Set up the mock searcher
        mock_searcher = MagicMock()
        
        # Create a dictionary for the first hit
        hit1_data = {
            "content": "This is test chunk 1",
            "source": "test.pdf",
            "page_number": 1,
            "doc_id": "doc1"
        }
        
        # Create a dictionary for the second hit
        hit2_data = {
            "content": "This is test chunk 2",
            "source": "test.pdf",
            "page_number": 2,
            "doc_id": "doc2"
        }
        
        # Create mock hit objects that behave like dictionaries
        mock_hit1 = MagicMock()
        mock_hit1.__getitem__ = lambda self, key: hit1_data[key]
        mock_hit1.get = lambda key, default=None: hit1_data.get(key, default)
        mock_hit1.score = 0.9
        
        mock_hit2 = MagicMock()
        mock_hit2.__getitem__ = lambda self, key: hit2_data[key]
        mock_hit2.get = lambda key, default=None: hit2_data.get(key, default)
        mock_hit2.score = 0.7
        
        mock_searcher.search.return_value = [mock_hit1, mock_hit2]
        self.mock_whoosh_index.searcher.return_value.__enter__.return_value = mock_searcher

        # Set up the mock query parser
        mock_parser = MagicMock()
        mock_parser.parse.return_value = "parsed_query"

        # Call the method
        with patch('whoosh.qparser.QueryParser', return_value=mock_parser):
            results = self.search_service.perform_bm25_search("test query", top_k=10)

        # Check that the search returned the correct results
        assert len(results) == 2
        assert results[0][0] == "This is test chunk 1"
        assert results[0][1] == 0.9
        assert results[0][2] == {"source": "test.pdf", "page_number": 1, "doc_id": "doc1"}
        assert results[1][0] == "This is test chunk 2"
        assert results[1][1] == 0.7
        assert results[1][2] == {"source": "test.pdf", "page_number": 2, "doc_id": "doc2"}
        
        # Check that the searcher was called with the correct limit
        assert mock_searcher.search.call_args[1]['limit'] == 10
        # We don't check the exact query object since it's constructed internally

    def test_combine_results(self):
        """Test combining vector and BM25 search results."""
        # Set up test data
        vector_results = [
            ("This is test chunk 1", 0.8, {"source": "test.pdf", "page_number": 1}),
            ("This is test chunk 2", 0.6, {"source": "test.pdf", "page_number": 2})
        ]

        bm25_results = [
            ("This is test chunk 2", 0.9, {"source": "test.pdf", "page_number": 2}),
            ("This is test chunk 3", 0.7, {"source": "test.pdf", "page_number": 3})
        ]

        # Call the method
        results = self.search_service.combine_results(vector_results, bm25_results, alpha=0.6)

        # Check that the results were combined correctly
        assert len(results) == 3  # Unique documents
        
        # Check that the first result is the one with the highest combined score
        assert results[0][0] == "This is test chunk 2"  # This appears in both result sets
        
        # Check that all documents are included
        document_texts = [result[0] for result in results]
        assert "This is test chunk 1" in document_texts
        assert "This is test chunk 2" in document_texts
        assert "This is test chunk 3" in document_texts

    def test_rerank_results(self):
        """Test reranking search results."""
        # Set up test data
        documents = [
            ("This is test chunk 1", 0.8, {"source": "test.pdf", "page_number": 1}),
            ("This is test chunk 2", 0.6, {"source": "test.pdf", "page_number": 2})
        ]
        
        # Set up the mock cross-encoder
        self.mock_cross_encoder.predict.return_value = [0.9, 0.3]
        
        # Call the method
        results = self.search_service.rerank_results("test query", documents, top_k=2, threshold=0.2)
        
        # Check that the results were reranked correctly
        assert len(results) == 2
        assert results[0][0] == "This is test chunk 1"
        assert results[0][1] == 0.9
        assert results[1][0] == "This is test chunk 2"
        assert results[1][1] == 0.3
        
        # Check that the cross-encoder was called with the correct arguments
        self.mock_cross_encoder.predict.assert_called_once_with([
            ("test query", "This is test chunk 1"),
            ("test query", "This is test chunk 2")
        ])

    def test_search(self):
        """Test the complete search process."""
        # Set up mocks for the individual search methods
        with patch.object(self.search_service, 'perform_vector_search') as mock_vector_search, \
             patch.object(self.search_service, 'perform_bm25_search') as mock_bm25_search, \
             patch.object(self.search_service, 'combine_results') as mock_combine, \
             patch.object(self.search_service, 'rerank_results') as mock_rerank:

            # Configure the mocks
            mock_vector_search.return_value = [("vector result", 0.8, {})]
            mock_bm25_search.return_value = [("bm25 result", 0.7, {})]
            mock_combine.return_value = [("combined result", 0.75, {})]
            mock_rerank.return_value = [("reranked result", 0.9, {})]
    
            # Call the method
            results = self.search_service.search("test query", top_k=10, alpha=0.6, reranking_threshold=0.3)

            # Check that all the methods were called with the correct arguments
            mock_vector_search.assert_called_once_with("test query", 20)
            mock_bm25_search.assert_called_once_with("test query", 20)
            mock_combine.assert_called_once_with([("vector result", 0.8, {})], [("bm25 result", 0.7, {})], 0.6)
            mock_rerank.assert_called_once_with("test query", [("combined result", 0.75, {})], 10, 0.3)
            
            # Check that the search returned the correct results
            assert results == [("reranked result", 0.9, {})] 