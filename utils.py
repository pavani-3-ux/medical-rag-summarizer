"""Utility functions for Medical RAG Summarizer."""

import os
import re
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def print_stage_header(stage_number: int, stage_name: str) -> None:
    """Print a formatted stage header."""
    header = f"{'═' * 40}\n🔍 STAGE {stage_number}: {stage_name}\n{'═' * 40}"
    console.print(header, style="bold cyan")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"✅ {message}", style="bold green")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"❌ {message}", style="bold red")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"⚠️  {message}", style="bold yellow")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"ℹ️  {message}", style="bold blue")


def print_timing(stage_name: str, elapsed_seconds: float) -> None:
    """Print timing information."""
    console.print(f"⏱️  {stage_name}: {elapsed_seconds:.2f}s", style="dim magenta")


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep medical terms
    text = re.sub(r'[^\w\s.,;:()\-/]', '', text)
    return text.strip()


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


def print_retrieved_chunks_table(chunks: List[tuple]) -> None:
    """Print retrieved chunks in a formatted table."""
    table = Table(title="Retrieved Medical Documents", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Source", style="yellow")
    table.add_column("Preview", style="white")
    
    for i, (chunk_text, score, metadata) in enumerate(chunks, 1):
        source = metadata.get("source_file", "Unknown")
        preview = truncate_text(chunk_text, 60)
        table.add_row(
            str(i),
            f"{score:.4f}",
            source,
            preview
        )
    
    console.print(table)


def print_index_stats(total_docs: int, total_chunks: int, embedding_model: str, index_size_mb: float) -> None:
    """Print index statistics."""
    panel_content = f"""
[bold cyan]Total Documents:[/bold cyan] {format_number(total_docs)}
[bold cyan]Total Chunks:[/bold cyan] {format_number(total_chunks)}
[bold cyan]Embedding Model:[/bold cyan] {embedding_model}
[bold cyan]Index Size:[/bold cyan] {index_size_mb:.2f} MB
    """
    console.print(Panel(panel_content, title="📊 Index Statistics", border_style="green"))


def validate_api_key() -> bool:
    """Validate that ANTHROPIC_API_KEY is set."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print_error("ANTHROPIC_API_KEY environment variable is not set.")
        print_info("Please set your API key: export ANTHROPIC_API_KEY='your-key-here'")
        print_info("Or create a .env file with: ANTHROPIC_API_KEY=your-key-here")
        return False
    return True


def validate_index_exists(index_path: str, metadata_path: str) -> bool:
    """Validate that FAISS index and metadata files exist."""
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print_error("FAISS index or metadata file not found.")
        print_info("Run 'python main.py ingest --path ./sample_docs' first.")
        return False
    return True
