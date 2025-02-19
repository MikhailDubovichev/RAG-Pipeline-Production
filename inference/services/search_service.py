"""
Search Service for the RAG Pipeline's Inference Component.

This module implements a hybrid search system that combines:
1. Semantic Search (via FAISS)
2. Keyword Search (via Whoosh/BM25)
3. Cross-Encoder Reranking

Key Features:
1. Hybrid search combining multiple approaches
2. Configurable result combination
3. Optional result reranking
4. Rich metadata preservation
5. Batched processing for efficiency

The search process follows these steps:
1. Parallel semantic and keyword search
2. Score-based result combination
3. Cross-encoder reranking (if enabled)
4. Threshold-based filtering
5. Final result ranking

Technical Details:
- FAISS for efficient vector similarity search
- Whoosh/BM25 for traditional keyword search
- Cross-encoder for accurate result reranking
- Batch processing for performance optimization
"""

from typing import List, Tuple, Dict  # Type hints for better code clarity
import logging  # Python's built-in logging facility
from whoosh.qparser import QueryParser  # Whoosh query parsing

class SearchService:
    """
    Service for performing hybrid search over document collections.
    
    This service combines multiple search approaches:
    1. Vector Search (FAISS):
       - Semantic similarity matching
       - Embedding-based retrieval
       - Fast approximate nearest neighbors
       
    2. Keyword Search (Whoosh/BM25):
       - Traditional keyword matching
       - Field-specific queries
       - Metadata filtering
       
    3. Result Reranking:
       - Cross-encoder scoring
       - Threshold filtering
       - Batch processing
       
    The service ensures:
    - Optimal result combination
    - Efficient search execution
    - Graceful error handling
    - Metadata preservation
    """

    def __init__(self, faiss_index, whoosh_index, cross_encoder=None):
        """
        Initialize search service with required indices and models.
        
        The service requires:
        1. FAISS index for vector search
        2. Whoosh index for keyword search
        3. Optional cross-encoder for reranking
        
        Args:
            faiss_index: Initialized FAISS index for vector search
            whoosh_index: Initialized Whoosh index for keyword search
            cross_encoder: Optional cross-encoder model for reranking
                If not provided, reranking step will be skipped
                
        Note:
            The indices should be pre-built by the data preparation
            pipeline and loaded in read-only mode for search.
        """
        self.faiss_index = faiss_index
        self.whoosh_index = whoosh_index
        self.cross_encoder = cross_encoder

    def perform_vector_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Perform semantic search using the FAISS vector index.
        
        This method:
        1. Converts query to vector using embeddings
        2. Finds nearest neighbors in vector space
        3. Retrieves corresponding documents
        
        The semantic search is effective for:
        - Finding conceptually similar content
        - Handling paraphrases and synonyms
        - Language-agnostic matching
        
        Args:
            query (str): User's search query
            top_k (int): Number of results to retrieve
                Default is 10 results
                
        Returns:
            List[Tuple[str, float, Dict]]: List of results, each containing:
                - str: Document text content
                - float: Similarity score (higher is better)
                - Dict: Document metadata
                
        Note:
            Scores are normalized to [0, 1] range where 1 is most similar.
            Returns empty list on error with error logged.
        """
        try:
            # Create retriever with top-k configuration
            retriever = self.faiss_index.as_retriever(similarity_top_k=top_k)
            
            # Perform retrieval
            results = retriever.retrieve(query)
            
            # Format results with scores and metadata
            return [(res.node.text, 
                    res.score if res.score is not None else 0.0, 
                    res.node.metadata) 
                   for res in results]
        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            return []

    def perform_bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Perform keyword search using Whoosh with BM25 scoring.
        
        This method:
        1. Parses query into searchable terms
        2. Searches using BM25 algorithm
        3. Retrieves matching documents with metadata
        
        BM25 search is effective for:
        - Exact keyword matching
        - Phrase matching
        - Field-specific searches
        
        Args:
            query (str): User's search query
            top_k (int): Number of results to retrieve
                Default is 10 results
                
        Returns:
            List[Tuple[str, float, Dict]]: List of results, each containing:
                - str: Document text content
                - float: BM25 score (higher is better)
                - Dict: Document metadata including:
                    - source: Document source/filename
                    - sheet: Excel sheet name (if applicable)
                    - row_number: Spreadsheet row (if applicable)
                    - slide_number: Presentation slide (if applicable)
                    - section: Document section/heading
                    - chunk_number: Position in chunked document
                    - total_chunks_in_section: Total chunks
                    - page_number: PDF page number (if applicable)
                    - doc_id: Unique document identifier
                
        Note:
            Returns empty list on error with error logged.
        """
        with self.whoosh_index.searcher() as searcher:
            # Create parser for content field
            parser = QueryParser("content", schema=self.whoosh_index.schema)
            try:
                # Parse and execute search
                parsed_query = parser.parse(query)
                results = searcher.search(parsed_query, limit=top_k)
                
                # Format results with all metadata
                return [
                    (hit["content"], 
                     hit.score, 
                     {
                        "source": hit.get("source", "Unknown Source"),
                        "sheet": hit.get("sheet"),
                        "row_number": hit.get("row_number"),
                        "slide_number": hit.get("slide_number"),
                        "section": hit.get("section"),
                        "chunk_number": hit.get("chunk_number"),
                        "total_chunks_in_section": hit.get("total_chunks_in_section"),
                        "page_number": hit.get("page_number"),
                        "doc_id": hit.get("doc_id"),
                     }) 
                    for hit in results
                ]
            except Exception as e:
                logging.error(f"BM25 search failed: {e}")
                return []

    def combine_results(self, vector_results: List[Tuple], bm25_results: List[Tuple], 
                       alpha: float = 0.5) -> List[Tuple[str, float, Dict]]:
        """
        Combine and score results from vector and keyword search.
        
        This method implements a hybrid scoring approach:
        1. Reciprocal rank fusion for both result sets
        2. Weighted combination using alpha parameter
        3. Deduplication of results
        4. Final score-based ranking
        
        The combination strategy:
        - Uses position-based scoring (1/rank)
        - Weights vector and keyword scores
        - Preserves metadata from first occurrence
        - Sorts by combined score
        
        Args:
            vector_results (List[Tuple]): Results from vector search
            bm25_results (List[Tuple]): Results from keyword search
            alpha (float): Weight for vector search results
                Range: 0.0 to 1.0 (default: 0.5)
                - 1.0: Only vector search results
                - 0.0: Only keyword search results
                - 0.5: Equal weight to both
                
        Returns:
            List[Tuple[str, float, Dict]]: Combined and ranked results
                - str: Document text content
                - float: Combined score (higher is better)
                - Dict: Document metadata from first occurrence
                
        Note:
            Results are sorted by descending score (best first).
        """
        combined_scores = {}  # Track combined scores
        metadata_mapping = {}  # Track metadata

        # Process vector search results with alpha weight
        for idx, (text, score, metadata) in enumerate(vector_results):
            if text:  # Skip empty results
                combined_scores[text] = combined_scores.get(text, 0) + alpha / (idx + 1)
                metadata_mapping[text] = metadata

        # Process BM25 results with (1-alpha) weight
        for idx, (text, score, metadata) in enumerate(bm25_results):
            if text:  # Skip empty results
                combined_scores[text] = combined_scores.get(text, 0) + (1 - alpha) / (idx + 1)
                if text not in metadata_mapping:
                    metadata_mapping[text] = metadata

        # Create final sorted result list
        combined_results = [(text, score, metadata_mapping[text]) 
                           for text, score in combined_scores.items()]
        return sorted(combined_results, key=lambda x: x[1], reverse=True)

    def rerank_results(self, query: str, documents: List[Tuple], 
                      top_k: int = 10, threshold: float = 0.2, 
                      batch_size: int = 16) -> List[Tuple[str, float, Dict]]:
        """
        Rerank search results using cross-encoder model.
        
        This method:
        1. Pairs query with each document
        2. Processes pairs in batches
        3. Scores relevance using cross-encoder
        4. Filters and ranks by score
        
        The reranking process:
        - Creates (query, document) pairs
        - Processes in batches for efficiency
        - Applies relevance threshold
        - Returns top-k results
        
        Args:
            query (str): Original search query
            documents (List[Tuple]): Initial search results
            top_k (int): Number of results to return (default: 10)
            threshold (float): Minimum relevance score (default: 0.2)
            batch_size (int): Batch size for processing (default: 16)
                
        Returns:
            List[Tuple[str, float, Dict]]: Reranked results
                - str: Document text content
                - float: Relevance score from cross-encoder
                - Dict: Document metadata
                
        Note:
            If no cross-encoder is available, returns original results.
            On error, falls back to original ranking.
        """
        # Return original results if no cross-encoder
        if not self.cross_encoder:
            return documents[:top_k]

        try:
            # Create query-document pairs
            input_pairs = [(query, doc[0]) for doc in documents]
            scores = []

            # Process in batches
            for i in range(0, len(input_pairs), batch_size):
                batch = input_pairs[i:i + batch_size]
                try:
                    # Score current batch
                    batch_scores = self.cross_encoder.predict(batch)
                    scores.extend(batch_scores)
                except Exception as e:
                    logging.error(f"Cross-encoder prediction failed for batch {i // batch_size}: {e}")
                    scores.extend([0.0] * len(batch))  # Use zero scores on error

            # Create scored results
            scored_docs = [(doc[0], score, doc[2]) 
                          for doc, score in zip(documents, scores)]
            
            # Sort by score and apply threshold
            reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            filtered_docs = [doc for doc in reranked_docs if doc[1] >= threshold]

            return filtered_docs[:top_k]
            
        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            return documents[:top_k]  # Fall back to original ranking

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5, 
              reranking_threshold: float = 0.2) -> List[Tuple[str, float, Dict]]:
        """
        Execute complete search pipeline with hybrid search and reranking.
        
        This method orchestrates the full search process:
        1. Parallel vector and keyword search
        2. Result combination with weights
        3. Cross-encoder reranking
        4. Final result selection
        
        The process ensures:
        - Comprehensive search coverage
        - Optimal result ranking
        - Efficient execution
        - Error resilience
        
        Args:
            query (str): User's search query
            top_k (int): Number of final results (default: 10)
            alpha (float): Vector vs keyword weight (default: 0.5)
                Range: 0.0 to 1.0
                - 1.0: Only vector search
                - 0.0: Only keyword search
            reranking_threshold (float): Minimum relevance score
                Default: 0.2
                
        Returns:
            List[Tuple[str, float, Dict]]: Final search results
                - str: Document text content
                - float: Final relevance score
                - Dict: Document metadata
                
        Note:
            This is the main entry point for search functionality.
            Each step has fallbacks for robustness.
        """
        # Perform both search types with expanded initial pool
        vector_results = self.perform_vector_search(query, top_k * 2)
        bm25_results = self.perform_bm25_search(query, top_k * 2)

        # Combine results with specified weights
        hybrid_results = self.combine_results(vector_results, bm25_results, alpha)

        # Rerank if cross-encoder is available
        if self.cross_encoder:
            return self.rerank_results(query, hybrid_results, top_k, reranking_threshold)
        
        # Return top results if no reranking
        return hybrid_results[:top_k] 