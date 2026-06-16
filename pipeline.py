"""Complete RAG pipeline for Medical RAG Summarizer."""

import time
from typing import Optional
from ingest import DocumentIngester
from embeddings import EmbeddingGenerator
from vector_store import FAISSVectorStore
from retriever import DocumentRetriever
from summarizer import MedicalSummarizer
from utils import print_info, print_success, print_error, print_timing, print_index_stats
import config


class RAGPipeline:
    """Complete Retrieval-Augmented Generation pipeline."""
    
    def __init__(self):
        """Initialize the pipeline components."""
        self.ingester = DocumentIngester(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.embedding_generator = EmbeddingGenerator(
            model_name=config.EMBEDDING_MODEL,
            batch_size=32
        )
        self.vector_store = FAISSVectorStore(
            index_path=config.FAISS_INDEX_PATH,
            metadata_path=config.METADATA_PATH
        )
        self.retriever = DocumentRetriever(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store
        )
        self.summarizer = MedicalSummarizer(model=config.LLM_MODEL)
    
    def ingest_and_index(self, docs_path: str) -> bool:
        """Ingest documents and build vector index.
        
        Args:
            docs_path: Path to documents folder or file
            
        Returns:
            True if successful
        """
        start_time = time.time()
        
        # Stage 1: Ingest
        ingest_start = time.time()
        chunks = self.ingester.ingest_documents(docs_path)
        ingest_elapsed = time.time() - ingest_start
        print_timing("Document Ingestion", ingest_elapsed)
        
        if not chunks:
            print_error("No documents ingested")
            return False
        
        # Stage 2: Generate embeddings
        embed_start = time.time()
        embeddings = self.embedding_generator.embed_chunks(chunks)
        embed_elapsed = time.time() - embed_start
        print_timing("Embedding Generation", embed_elapsed)
        
        if embeddings is None or len(embeddings) == 0:
            print_error("Failed to generate embeddings")
            return False
        
        # Stage 3: Build index
        index_start = time.time()
        self.vector_store.add_documents(chunks, embeddings)
        index_elapsed = time.time() - index_start
        print_timing("Vector Indexing", index_elapsed)
        
        # Stage 4: Save to disk
        save_start = time.time()
        self.vector_store.save()
        save_elapsed = time.time() - save_start
        print_timing("Saving to Disk", save_elapsed)
        
        total_elapsed = time.time() - start_time
        print_timing("Total Pipeline Time", total_elapsed)
        
        return True
    
    def query_and_summarize(self, question: str) -> Optional[str]:
        """Retrieve relevant documents and generate summary.
        
        Args:
            question: User question
            
        Returns:
            Summary text or None if failed
        """
        start_time = time.time()
        
        # Load existing index
        print_info("Loading vector index...")
        if not self.vector_store.load():
            print_error("Failed to load vector index")
            return None
        
        print_success("Vector index loaded successfully")
        
        # Stage 4: Retrieve
        retrieval_start = time.time()
        retrieved_chunks = self.retriever.retrieve(question, top_k=config.TOP_K_RETRIEVAL)
        retrieval_elapsed = time.time() - retrieval_start
        print_timing("Semantic Retrieval", retrieval_elapsed)
        
        if not retrieved_chunks:
            print_error("No relevant documents found")
            return None
        
        # Stage 5: Summarize
        summarize_start = time.time()
        summary = self.summarizer.summarize(retrieved_chunks, question)
        summarize_elapsed = time.time() - summarize_start
        print_timing("LLM Summarization", summarize_elapsed)
        
        total_elapsed = time.time() - start_time
        print_timing("Total Query Time", total_elapsed)
        
        return summary
    
    def show_stats(self) -> None:
        """Show index statistics."""
        print_info("Loading vector index...")
        if not self.vector_store.load():
            print_error("Failed to load vector index")
            return
        
        stats = self.vector_store.get_stats()
        if stats:
            print_index_stats(
                total_docs=stats["total_documents"],
                total_chunks=stats["total_chunks"],
                embedding_model=config.EMBEDDING_MODEL,
                index_size_mb=stats["index_size_mb"]
            )
        else:
            print_error("Could not retrieve statistics")
