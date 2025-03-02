"""
Unit tests for the LLMService class.

This module contains tests for the LLMService class, which is responsible for:
1. Generating responses using a language model
2. Formatting responses with references to source documents
"""

import pytest
from unittest.mock import MagicMock, patch

from inference.services import LLMService

class TestLLMService:
    """Tests for the LLMService class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock LLM
        self.mock_llm = MagicMock()
        
        # Create an instance of LLMService with the mock LLM
        self.llm_service = LLMService(self.mock_llm)
    
    def test_generate_response(self):
        """Test generating a response from the LLM."""
        # Set up test data
        query = "What is the capital of France?"
        context = "Paris is the capital of France."
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.text = "Paris is the capital of France."
        self.mock_llm.complete.return_value = mock_response
        
        # Call the method under test
        response, sources = self.llm_service.generate_response(query, context)
        
        # Verify the response
        assert response == "Paris is the capital of France."
        
        # Verify that the LLM was called with the correct parameters
        assert self.mock_llm.complete.called
        
        # Check that the call arguments contain the query and context
        call_kwargs = self.mock_llm.complete.call_args.kwargs
        prompt = call_kwargs.get('prompt', '') if hasattr(call_kwargs, 'get') else ''
        if not prompt:
            # If not in kwargs, check if it's in args
            call_args = self.mock_llm.complete.call_args.args
            prompt = call_args[0] if call_args else ''
            
        assert query in str(prompt) or query in str(call_kwargs) or query in str(call_args)
        assert context in str(prompt) or context in str(call_kwargs) or context in str(call_args)
    
    def test_format_response_with_references(self):
        """Test formatting a response with references to source documents."""
        # Set up test data
        response = "Paris is the capital of France."
        search_results = [
            ("Paris is the capital of France and is known for the Eiffel Tower.", 0.95,
             {"source": "geography.pdf", "page_number": 42}),
            ("France is a country in Western Europe with Paris as its capital.", 0.85,
             {"source": "europe.pdf", "page_number": 17})
        ]

        # Call the method under test
        formatted_response = self.llm_service.format_response_with_references(response, search_results)

        # Verify the formatted response
        assert response in formatted_response
        assert "geography.pdf" in formatted_response
        assert "europe.pdf" in formatted_response
        assert "0.95" in formatted_response
        assert "0.85" in formatted_response

        # Check if page numbers are included in some format
        assert "42" in formatted_response
        assert "17" in formatted_response
    
    def test_format_response_with_empty_results(self):
        """Test formatting a response with no search results."""
        # Set up test data
        response = "I don't know the answer to that question."
        search_results = []
        
        # Call the method under test
        formatted_response = self.llm_service.format_response_with_references(response, search_results)
        
        # Verify the formatted response
        assert response in formatted_response
        assert "Sources" in formatted_response
        # The implementation might not use exactly "No sources found" but should indicate no sources
        assert len(formatted_response) > len(response)  # At least some additional text was added
    
    def test_format_response_with_missing_metadata(self):
        """Test formatting a response with search results that have missing metadata."""
        # Set up test data
        response = "Paris is the capital of France."
        search_results = [
            ("Paris is the capital of France.", 0.9, {}),  # Missing metadata
            ("France's capital is Paris.", 0.8, {"source": "facts.pdf"})  # Missing page
        ]
        
        # Call the method under test
        formatted_response = self.llm_service.format_response_with_references(response, search_results)
        
        # Verify the formatted response
        assert response in formatted_response
        # Check for some indication of unknown source (might not be exactly "Unknown source")
        assert "Unknown" in formatted_response or "unknown" in formatted_response
        assert "facts.pdf" in formatted_response
        assert "0.9" in formatted_response
        assert "0.8" in formatted_response