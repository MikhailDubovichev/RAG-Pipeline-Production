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
        
        The formatting includes:
        - Source PDF filenames
        - Location details (page number)
        - Content previews
        - Clean, readable layout
        
        Args:
            response (str): Raw LLM-generated response
            search_results (List[Tuple[str, float, Dict]]): Search results containing:
                - str: Document text content
                - float: Relevance score
                - Dict: Document metadata including:
                    - source: PDF filename
                    - section: Document section
                    - page_number: PDF page
                    - chunk_number: Position in document
                    - total_chunks_in_section: Total chunks
                
        Returns:
            str: Formatted response with references in the format:
                <response text>
                
                Sources:
                
                - Document: document.pdf
                  Page: 5
                  Created: September 2024
                  Content: "Text preview..."
                
        Note:
            - Handles missing metadata gracefully
            - Extracts page numbers and dates from content when not in metadata
            - Truncates long text previews
            - Maintains readable formatting
        """
        # Return as is if no relevant information
        if response.lower() == 'no relevant information available.':
            return response

        # Build reference list
        references = []
        for text, score, metadata in search_results:
            if metadata:
                # Format source information in a more readable way
                source_info = []
                
                # Document name
                source_info.append(f"Document: {metadata.get('source', 'Unknown Source')}")
                
                # Extract information from text if it's in a common format
                # This handles cases where metadata is embedded in the text content
                page_number = None
                creation_date = None
                actual_content = text
                
                # Try to extract page number and date from text content
                lines = text.split('\n')
                if len(lines) >= 3:
                    # Check if second line might be a page number
                    if len(lines) > 1 and lines[1].strip().isdigit():
                        page_number = lines[1].strip()
                        
                    # Check if third line might be a date
                    if len(lines) > 2 and "20" in lines[2]:  # Simple check for year
                        creation_date = lines[2].strip()
                    
                    # Remove metadata lines from content if we extracted them
                    if page_number or creation_date:
                        # Skip the first few lines that contain metadata
                        metadata_lines = 3  # Usually section name, page number, date
                        actual_content = '\n'.join(lines[metadata_lines:])
                
                # Page number with proper formatting
                if metadata.get('page_number'):
                    source_info.append(f"Page: {metadata.get('page_number')}")
                elif page_number:
                    source_info.append(f"Page: {page_number}")
                
                # Section information if available
                if metadata.get('section'):
                    source_info.append(f"Section: {metadata.get('section')}")
                elif len(lines) > 0:
                    # First line might be section
                    source_info.append(f"Section: {lines[0].strip()}")
                
                # Creation date if available
                if metadata.get('creation_date'):
                    source_info.append(f"Created: {metadata.get('creation_date')}")
                elif creation_date:
                    source_info.append(f"Created: {creation_date}")
                
                # Add text preview with truncation
                text_preview = actual_content.strip()
                if len(text_preview) > 150:
                    text_preview = text_preview[:150] + "..."
                source_info.append(f"Content: \"{text_preview}\"")
                
                # Join all parts with newlines and proper indentation
                references.append("\n  ".join(source_info))
            else:
                references.append(f"Unknown Source:\n  Content: \"{text[:150]}...\"")

        # Format final response with references
        references_text = "\n\n- ".join(references)
        return f"{response}\n\nSources:\n\n- {references_text}" 