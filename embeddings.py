"""Embedding generation for Medical RAG Summarizer."""

from typing import List, Optional
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import print_stage_header, print_success, print_error, print_info
import config


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL, batch_size: int = config.EMBEDDING_BATCH_SIZE):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
    
    def load_model(self) -> None:
        """Load the embedding model."""
        print_info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            print_success(f"Model loaded successfully")
            print_info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print_error(f"Failed to load model: {str(e)}")
            raise
    
    def embed_chunks(self, chunks: List[dict]) -> np.ndarray:
        """Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            
        Returns:
            NumPy array of embeddings (n_chunks, embedding_dim)
        """
        print_stage_header(2, "EMBEDDING GENERATION")
        
        if self.model is None:
            self.load_model()
        
        if not chunks:
            print_error("No chunks to embed")
            return np.array([])
        
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        print_info(f"Generating embeddings for {len(texts)} chunk(s)...")
        
        # Encode with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print_success(f"Generated {len(embeddings)} embedding(s)")
        print_info(f"Embedding shape: {embeddings.shape}")
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            NumPy array of shape (1, embedding_dim)
        """
        if self.model is None:
            self.load_model()
        
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0]  # Return 1D array
