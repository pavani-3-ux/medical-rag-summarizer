"""FAISS vector store management for Medical RAG Summarizer."""

import json
import os
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from utils import print_stage_header, print_success, print_error, print_info, get_file_size_mb
import config


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self, index_path: str = config.FAISS_INDEX_PATH, metadata_path: str = config.METADATA_PATH):
        """Initialize the vector store.
        
        Args:
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata JSON
        """
        if faiss is None:
            raise ImportError("faiss-cpu is required")
        
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        self.embedding_dim = None
    
    def add_documents(self, chunks: List[dict], embeddings: np.ndarray) -> None:
        """Add documents and their embeddings to the index.
        
        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: NumPy array of embeddings
        """
        print_stage_header(3, "VECTOR STORE INDEXING")
        
        if len(chunks) != len(embeddings):
            print_error(f"Chunk count ({len(chunks)}) != Embedding count ({len(embeddings)})")
            return
        
        if len(embeddings) == 0:
            print_error("No embeddings to add")
            return
        
        # Set embedding dimension
        self.embedding_dim = embeddings.shape[1]
        print_info(f"Embedding dimension: {self.embedding_dim}")
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add embeddings to index
        embeddings_f32 = embeddings.astype(np.float32)
        self.index.add(embeddings_f32)
        
        # Store metadata
        self.metadata = [
            {
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            }
            for chunk in chunks
        ]
        
        print_success(f"Added {len(embeddings)} vector(s) to index")
        print_info(f"Index stats: {self.index.ntotal} vectors, dimension {self.embedding_dim}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding (1D array)
            top_k: Number of results to return
            
        Returns:
            List of (text, score, metadata) tuples
        """
        if self.index is None:
            print_error("Index not loaded. Load or create an index first.")
            return []
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding_f32 = query_embedding.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding_f32, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                distance = distances[0][i]
                # Convert L2 distance to similarity score (0-1)
                similarity_score = 1.0 / (1.0 + distance)
                
                chunk_data = self.metadata[idx]
                results.append((
                    chunk_data["text"],
                    similarity_score,
                    chunk_data["metadata"]
                ))
        
        return results
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        if self.index is None:
            print_error("No index to save")
            return
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            print_success(f"FAISS index saved to {self.index_path}")
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print_success(f"Metadata saved to {self.metadata_path}")
            
            # Print file sizes
            index_size = get_file_size_mb(self.index_path)
            metadata_size = get_file_size_mb(self.metadata_path)
            print_info(f"Index size: {index_size:.2f} MB")
            print_info(f"Metadata size: {metadata_size:.2f} MB")
        
        except Exception as e:
            print_error(f"Failed to save: {str(e)}")
    
    def load(self) -> bool:
        """Load index and metadata from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.index_path):
                print_error(f"Index file not found: {self.index_path}")
                return False
            
            if not os.path.exists(self.metadata_path):
                print_error(f"Metadata file not found: {self.metadata_path}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            print_success(f"FAISS index loaded from {self.index_path}")
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print_success(f"Metadata loaded from {self.metadata_path}")
            
            # Set embedding dimension
            if self.index.ntotal > 0:
                self.embedding_dim = self.index.d
            
            print_info(f"Index loaded: {self.index.ntotal} vectors, dimension {self.embedding_dim}")
            return True
        
        except Exception as e:
            print_error(f"Failed to load: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """Get index statistics.
        
        Returns:
            Dictionary with index stats
        """
        if self.index is None:
            return {}
        
        total_chunks = self.index.ntotal
        total_docs = len(set(
            chunk["metadata"]["source_file"]
            for chunk in self.metadata
        ))
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "embedding_dimension": self.embedding_dim,
            "index_size_mb": get_file_size_mb(self.index_path)
        }
