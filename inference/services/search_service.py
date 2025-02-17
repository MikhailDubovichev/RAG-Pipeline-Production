from typing import List, Tuple, Dict
import logging
from whoosh.qparser import QueryParser

class SearchService:
    def __init__(self, faiss_index, whoosh_index, cross_encoder=None):
        self.faiss_index = faiss_index
        self.whoosh_index = whoosh_index
        self.cross_encoder = cross_encoder

    def perform_vector_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Perform semantic search using FAISS index.
        """
        try:
            retriever = self.faiss_index.as_retriever(similarity_top_k=top_k)
            results = retriever.retrieve(query)
            return [(res.node.text, res.score if res.score is not None else 0.0, res.node.metadata) 
                    for res in results]
        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            return []

    def perform_bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Perform keyword search using Whoosh/BM25.
        """
        with self.whoosh_index.searcher() as searcher:
            parser = QueryParser("content", schema=self.whoosh_index.schema)
            try:
                parsed_query = parser.parse(query)
                results = searcher.search(parsed_query, limit=top_k)
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
        Combine and re-rank results from vector and keyword search.
        """
        combined_scores = {}
        metadata_mapping = {}

        # Process vector search results
        for idx, (text, score, metadata) in enumerate(vector_results):
            if text:
                combined_scores[text] = combined_scores.get(text, 0) + alpha / (idx + 1)
                metadata_mapping[text] = metadata

        # Process BM25 results
        for idx, (text, score, metadata) in enumerate(bm25_results):
            if text:
                combined_scores[text] = combined_scores.get(text, 0) + (1 - alpha) / (idx + 1)
                if text not in metadata_mapping:
                    metadata_mapping[text] = metadata

        # Sort and format results
        combined_results = [(text, score, metadata_mapping[text]) 
                           for text, score in combined_scores.items()]
        return sorted(combined_results, key=lambda x: x[1], reverse=True)

    def rerank_results(self, query: str, documents: List[Tuple], 
                      top_k: int = 10, threshold: float = 0.2, 
                      batch_size: int = 16) -> List[Tuple[str, float, Dict]]:
        """
        Rerank results using cross-encoder if available.
        """
        if not self.cross_encoder:
            return documents[:top_k]

        try:
            input_pairs = [(query, doc[0]) for doc in documents]
            scores = []

            for i in range(0, len(input_pairs), batch_size):
                batch = input_pairs[i:i + batch_size]
                try:
                    batch_scores = self.cross_encoder.predict(batch)
                    scores.extend(batch_scores)
                except Exception as e:
                    logging.error(f"Cross-encoder prediction failed for batch {i // batch_size}: {e}")
                    scores.extend([0.0] * len(batch))

            scored_docs = [(doc[0], score, doc[2]) for doc, score in zip(documents, scores)]
            reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            filtered_docs = [doc for doc in reranked_docs if doc[1] >= threshold]

            return filtered_docs[:top_k]
        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            return documents[:top_k]

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5, 
              reranking_threshold: float = 0.2) -> List[Tuple[str, float, Dict]]:
        """
        Perform complete search pipeline including hybrid search and reranking.
        """
        # Perform both search types
        vector_results = self.perform_vector_search(query, top_k * 2)
        bm25_results = self.perform_bm25_search(query, top_k * 2)

        # Combine results
        hybrid_results = self.combine_results(vector_results, bm25_results, alpha)

        # Rerank if cross-encoder is available
        if self.cross_encoder:
            return self.rerank_results(query, hybrid_results, top_k, reranking_threshold)
        
        return hybrid_results[:top_k] 