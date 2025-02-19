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
            # Construct RAG-optimized prompt
            prompt = (
                f"Question: {query}\n\n"
                f"Context:\n{context}\n\n"
                "Based on the provided context, is there enough relevant information to answer the question? "
                "Respond with 'Yes' or 'No'. If yes, provide the answer; otherwise, say 'No relevant information available.'"
            )
            
            # Generate response using LLM
            response = self.llm.complete(prompt=prompt)
            return response.text.strip(), []
            
        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            return "An error occurred while generating the response. Please try again later.", []

    def format_response_with_references(self, 
                                     response: str, 
                                     search_results: List[Tuple[str, float, Dict]]) -> str:
        """
        Format the LLM response with detailed source references.
        
        This method enhances responses with:
        1. Source attribution
        2. Metadata details
        3. Text previews
        4. Structured formatting
        
        The formatting includes:
        - Source filenames
        - Location details (page, slide, etc.)
        - Content previews
        - Clean, readable layout
        
        Args:
            response (str): Raw LLM-generated response
            search_results (List[Tuple[str, float, Dict]]): Search results containing:
                - str: Document text content
                - float: Relevance score
                - Dict: Document metadata including:
                    - source: Document source/filename
                    - sheet: Excel sheet name
                    - row_number: Spreadsheet row
                    - slide_number: Presentation slide
                    - section: Document section
                    - page_number: PDF page
                    - chunk_number: Position in document
                    - total_chunks_in_section: Total chunks
                
        Returns:
            str: Formatted response with references in the format:
                <response text>
                
                Sources:
                - Source: doc.pdf, Page: 5: "Text preview..."
                - Source: sheet.xlsx, Sheet: Data, Row: 3: "Text preview..."
                
        Note:
            - Handles missing metadata gracefully
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
                ref_parts = []
                
                # Add source information
                ref_parts.append(f"Source: {metadata.get('source', 'Unknown Source')}")
                
                # Add additional metadata if available
                for key in ['sheet', 'row_number', 'slide_number', 'section', 
                           'page_number', 'chunk_number', 'total_chunks_in_section']:
                    if metadata.get(key):
                        ref_parts.append(f"{key.replace('_', ' ').title()}: {metadata[key]}")
                
                # Add text preview with truncation
                text_preview = text[:50] + "..." if len(text) > 50 else text
                references.append(f"{', '.join(ref_parts)}: {text_preview}")
            else:
                references.append(f"Unknown Source: {text[:50]}...")

        # Format final response with references
        references_text = "\n".join([f"- {ref}" for ref in references])
        return f"{response}\n\nSources:\n{references_text}" 