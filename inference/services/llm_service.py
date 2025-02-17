import logging
from typing import List, Dict, Optional, Tuple

class LLMService:
    def __init__(self, llm_model):
        """
        Initialize LLM service with a language model.
        
        Args:
            llm_model: An instance of a language model (e.g., NebiusLLM)
        """
        self.llm = llm_model

    def generate_response(self, query: str, context: str) -> Tuple[str, List[Dict]]:
        """
        Generate a response using the LLM based on the query and context.
        
        Args:
            query: User's question
            context: Retrieved context from search
            
        Returns:
            Tuple of (response text, list of reference metadata)
        """
        try:
            prompt = (
                f"Question: {query}\n\n"
                f"Context:\n{context}\n\n"
                "Based on the provided context, is there enough relevant information to answer the question? "
                "Respond with 'Yes' or 'No'. If yes, provide the answer; otherwise, say 'No relevant information available.'"
            )
            
            response = self.llm.complete(prompt=prompt)
            return response.text.strip(), []
            
        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            return "An error occurred while generating the response. Please try again later.", []

    def format_response_with_references(self, 
                                     response: str, 
                                     search_results: List[Tuple[str, float, Dict]]) -> str:
        """
        Format the LLM response with reference information.
        
        Args:
            response: The LLM-generated response
            search_results: List of search results with metadata
            
        Returns:
            Formatted response with references
        """
        if response.lower() == 'no relevant information available.':
            return response

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
                
                # Add text preview
                text_preview = text[:50] + "..." if len(text) > 50 else text
                references.append(f"{', '.join(ref_parts)}: {text_preview}")
            else:
                references.append(f"Unknown Source: {text[:50]}...")

        references_text = "\n".join([f"- {ref}" for ref in references])
        return f"{response}\n\nSources:\n{references_text}" 