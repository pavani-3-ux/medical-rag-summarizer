"""Document ingestion and chunking for Medical RAG Summarizer."""

import os
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from utils import print_stage_header, print_success, print_error, print_info, clean_text

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None


class DocumentIngester:
    """Ingest and chunk documents from various formats."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the ingester.
        
        Args:
            chunk_size: Target size of each chunk in tokens (approximate)
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def ingest_documents(self, path: str) -> List[Dict]:
        """Ingest documents from a file or directory.
        
        Args:
            path: Path to a single file or directory containing files
            
        Returns:
            List of document chunks with metadata
        """
        print_stage_header(1, "DOCUMENT INGESTION")
        
        chunks = []
        path_obj = Path(path)
        
        if not path_obj.exists():
            print_error(f"Path does not exist: {path}")
            return chunks
        
        # Get all files to process
        files_to_process = []
        if path_obj.is_file():
            files_to_process = [path_obj]
        else:
            # Support common document formats
            for ext in ['*.txt', '*.pdf', '*.docx']:
                files_to_process.extend(path_obj.glob(ext))
        
        if not files_to_process:
            print_error(f"No supported documents found in {path}")
            print_info("Supported formats: .txt, .pdf, .docx")
            return chunks
        
        print_info(f"Found {len(files_to_process)} document(s) to ingest")
        
        # Process each file with progress bar
        for file_path in tqdm(files_to_process, desc="Ingesting documents", unit="file"):
            file_chunks = self._process_file(file_path)
            chunks.extend(file_chunks)
        
        print_success(f"Ingested {len(files_to_process)} document(s) into {len(chunks)} chunk(s)")
        return chunks
    
    def _process_file(self, file_path: Path) -> List[Dict]:
        """Process a single file and return chunks.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of chunks with metadata
        """
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == '.txt':
                text = self._read_txt(file_path)
            elif file_ext == '.pdf':
                text = self._read_pdf(file_path)
            elif file_ext == '.docx':
                text = self._read_docx(file_path)
            else:
                print_warning(f"Unsupported format: {file_ext}")
                return []
            
            if not text:
                print_warning(f"No text extracted from {file_path.name}")
                return []
            
            # Clean and chunk the text
            text = clean_text(text)
            chunks = self._chunk_text(text, file_path.name)
            return chunks
        
        except Exception as e:
            print_error(f"Error processing {file_path.name}: {str(e)}")
            return []
    
    def _read_txt(self, file_path: Path) -> str:
        """Read text from a .txt file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _read_pdf(self, file_path: Path) -> str:
        """Read text from a .pdf file."""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required to read PDF files")
        
        text = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text())
        
        return "\n".join(text)
    
    def _read_docx(self, file_path: Path) -> str:
        """Read text from a .docx file."""
        if Document is None:
            raise ImportError("python-docx is required to read DOCX files")
        
        doc = Document(file_path)
        text = [para.text for para in doc.paragraphs]
        return "\n".join(text)
    
    def _chunk_text(self, text: str, source_file: str) -> List[Dict]:
        """Chunk text into overlapping segments.
        
        Args:
            text: Full text to chunk
            source_file: Name of source file for metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Simple word-based chunking (approximate token counting)
        words = text.split()
        words_per_chunk = max(1, self.chunk_size // 1.3)  # Approximate: 1 token ≈ 1.3 words
        overlap_words = max(1, self.chunk_overlap // 1.3)
        
        chunks = []
        chunk_index = 0
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + int(words_per_chunk)]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source_file": source_file,
                        "chunk_index": chunk_index,
                        "total_chunks": None  # Will be updated later
                    }
                })
                chunk_index += 1
            
            # Move to next chunk with overlap
            i += int(words_per_chunk) - int(overlap_words)
        
        # Update total chunks count
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        return chunks
