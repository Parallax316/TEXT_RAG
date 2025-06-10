# backend/app/services/citation_manager.py
"""
CitationManager: Verifies, deduplicates, and formats citations for RAG outputs.
"""
from typing import List, Dict, Any, Tuple

class CitationManager:
    def __init__(self, all_chunks: List[Dict[str, Any]]):
        """
        all_chunks: List of all chunk metadata dicts in the collection (from vector store)
        """
        self.chunk_index = self._build_chunk_index(all_chunks)

    def _build_chunk_index(self, chunks: List[Dict[str, Any]]) -> set:
        """
        Build a set of (doc_id, page, paragraph) for fast lookup.
        """
        index = set()
        for meta in chunks:
            index.add((
                meta.get('source_doc_id'),
                meta.get('page_number'),
                meta.get('paragraph_number_in_page')
            ))
        return index

    def verify_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Only keep citations that exist in the chunk index.
        """
        verified = []
        seen = set()
        for c in citations:
            key = (
                c.get('source_doc_id'),
                c.get('page_number'),
                c.get('paragraph_number')
            )
            if key in self.chunk_index and key not in seen:
                verified.append(c)
                seen.add(key)
        return verified

    def deduplicate_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate citations by (doc_id, page, paragraph).
        """
        seen = set()
        deduped = []
        for c in citations:
            key = (
                c.get('source_doc_id'),
                c.get('page_number'),
                c.get('paragraph_number')
            )
            if key not in seen:
                deduped.append(c)
                seen.add(key)
        return deduped

    def format_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format citations for output (e.g., for UI or LLM prompt).
        """
        # This can be customized as needed
        return citations
