"""Document retrieval for Medical RAG Summarizer."""

from typing import List, Tuple, Dict
from embeddings import EmbeddingGenerator
from vector_store import FAISSVectorStore
from utils import print_stage_header, print_success, print_error, print_info, print_retrieved_chunks_table
import config


class DocumentRetriever:
    """Retrieve relevant documents for a query."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: FAISSVectorStore):
        """Initialize the retriever.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: FAISSVectorStore instance
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> List[Tuple[str, float, Dict]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of (text, score, metadata) tuples
        """
        print_stage_header(4, "SEMANTIC RETRIEVAL")
        
        print_info(f"Query: '{query}'")
        print_info(f"Retrieving top {top_k} chunks...")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        if not results:
            print_error("No relevant documents found")
            return []
        
        print_success(f"Retrieved {len(results)} chunk(s)")
        
        # Print results table
        print_retrieved_chunks_table(results)
        
        return results
