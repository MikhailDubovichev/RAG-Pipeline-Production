"""
Language Model Service for the RAG Pipeline's Inference Component.

This module handles the interaction with Large Language Models (LLMs) to:
1. Generate natural language responses to user queries
2. Format responses with source references
3. Handle context-aware question answering
4. Manage error cases gracefully

Key Features:
1. Prompt engineering for RAG
2. Source attribution
3. Metadata preservation
4. Error handling and logging
5. Response formatting

The service ensures:
- Context-aware responses
- Source transparency
- Graceful degradation
- User-friendly output
"""

import logging  # Python's built-in logging facility
from typing import List, Dict, Optional, Tuple  # Type hints for better code clarity

class LLMService:
    """
    Service for managing Large Language Model interactions.
    
    This service handles:
    1. Response Generation:
       - Context-aware answers
       - Relevance checking
       - Natural language output
       
    2. Response Formatting:
       - Source attribution
       - Metadata inclusion
       - Text previews
       
    3. Error Management:
       - Graceful fallbacks
       - User-friendly errors
       - Detailed logging
       
    The service ensures:
    - Consistent response quality
    - Source transparency
    - Error resilience
    - User-friendly output
    """

    def __init__(self, llm_model):
        """
        Initialize LLM service with a language model.
        
        The service requires a language model that:
        1. Has a completion/generation capability
        2. Accepts text prompts
        3. Returns structured responses
        
        Args:
            llm_model: Language model instance (e.g., NebiusLLM) that:
                - Has complete() method
                - Accepts text prompts
                - Returns response objects
                
        Note:
            The model should be pre-configured with appropriate:
            - Temperature (randomness)
            - Max tokens (length)
            - Other relevant parameters
        """
        self.llm = llm_model

    def generate_response(self, query: str, context: str) -> Tuple[str, List[Dict]]:
        """
        Generate a response using the LLM based on query and context.
        
        This method:
        1. Constructs a RAG-optimized prompt
        2. Checks context relevance
        3. Generates natural response
        4. Handles error cases
        
        The prompt structure:
        1. User question
        2. Retrieved context
        3. Relevance check
        4. Response instructions
        
        Args:
            query (str): User's question or request
            context (str): Retrieved passages from search
                Multiple passages joined with newlines
                
        Returns:
            Tuple[str, List[Dict]]: Contains:
                - str: Generated response text
                - List[Dict]: Reference metadata (for future use)
                
        Error Handling:
            - Returns user-friendly error message on failure
            - Logs detailed error information
            - Preserves system stability
        """
        try:
            logging.info(f"Generating response for query: {query}")
            logging.info(f"Context length: {len(context)} characters")
            
            # Construct RAG-optimized prompt
            prompt = (
                f"Question: {query}\n\n"
                f"Context:\n{context}\n\n"
                "Based on the provided context, is there enough relevant information to answer the question? "
                "Respond with 'Yes, there is enough information to answer the question' or 'No, there is not enough information to answer the question'."
                "If yes, say that you have enough information to answer the question and provide the answer; otherwise, say 'No relevant information available.'"
            )
            
            logging.info("Sending request to LLM")
            # Generate response using LLM
            response = self.llm.complete(prompt=prompt)
            
            if not response:
                logging.warning("LLM returned empty response")
                return "I couldn't generate a response. Please try again.", []
                
            logging.info(f"LLM response received, length: {len(response.text)}")
            return response.text.strip(), []
            
        except Exception as e:
            logging.error(f"LLM generation failed: {str(e)}", exc_info=True)
            return "An error occurred while generating the response. Please try again later.", []

    def format_response_with_references(self, response: str, search_results: List[Tuple[str, float, Dict]]) -> str:
        """
        Format the LLM response with detailed PDF source references.
        
        This method enhances responses with:
        1. PDF source attribution
        2. PDF metadata details
        3. Text previews
        4. Structured formatting
        5. Similarity scores for relevance analysis
        
        Args:
            response (str): Raw LLM-generated response
            search_results (List[Tuple[str, float, Dict]]): Search results containing:
                - str: Document text content
                - float: Relevance score (dot product/similarity score)
                - Dict: Document metadata
                
        Returns:
            str: Formatted response with references including similarity scores
                
        Note:
            - Handles missing metadata gracefully
            - Truncates long text previews
            - Maintains readable formatting
            - Includes similarity scores for threshold tuning
        """
        # Return as is if no relevant information
        if response.lower() == 'no relevant information available.':
            return response

        # Build reference list
        references = []
        for text, score, metadata in search_results:
            # Format source information in a more readable way
            source_info = []
            
            # Document name
            source_info.append(f"Document: {metadata.get('source', 'Unknown Source')}")
            
            # Extract page number from the first few lines if it's a number on its own line
            lines = text.split('\n')
            content_start = 0
            
            # Try to identify the structure of the content
            if len(lines) >= 2 and lines[1].strip().isdigit():
                # Add page number
                source_info.append(f"Page: {lines[1].strip()}")
                content_start = 3  # Skip section name, page number, and date
            elif metadata.get('page_number'):
                # Use metadata page number if available
                source_info.append(f"Page: {metadata.get('page_number')}")
            
            # Add section if available
            if len(lines) > 0 and content_start > 0:
                source_info.append(f"Section: {lines[0].strip()}")
            elif metadata.get('section'):
                source_info.append(f"Section: {metadata.get('section')}")
            
            # Add similarity score (rounded to 4 decimal places for readability)
            source_info.append(f"Relevance: {score:.4f}")
            
            # Add content preview
            content_text = '\n'.join(lines[content_start:]) if content_start > 0 else text
            content_text = content_text.strip()
            if len(content_text) > 150:
                content_text = content_text[:150] + "..."
            source_info.append(f"Content: \"{content_text}\"")
            
            # Join all parts with newlines and proper indentation
            references.append("\n  ".join(source_info))

        # Format final response with references
        references_text = "\n\n- ".join(references)
        return f"{response}\n\nSources:\n\n- {references_text}" 