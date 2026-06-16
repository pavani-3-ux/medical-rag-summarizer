"""LLM-based summarization for Medical RAG Summarizer."""

from typing import List, Tuple, Dict
from anthropic import Anthropic
from utils import print_stage_header, print_success, print_error, print_info
import config


class MedicalSummarizer:
    """Summarize medical documents using Claude."""
    
    def __init__(self, model: str = config.LLM_MODEL):
        """Initialize the summarizer.
        
        Args:
            model: Claude model to use
        """
        self.model = model
        self.client = Anthropic()
    
    def summarize(self, retrieved_chunks: List[Tuple[str, float, Dict]], query: str) -> str:
        """Generate a summary from retrieved chunks.
        
        Args:
            retrieved_chunks: List of (text, score, metadata) tuples
            query: Original query
            
        Returns:
            Summary text
        """
        print_stage_header(5, "LLM SUMMARIZATION")
        
        if not retrieved_chunks:
            print_error("No chunks provided for summarization")
            return ""
        
        # Build context from chunks
        context_parts = []
        for i, (chunk_text, score, metadata) in enumerate(retrieved_chunks, 1):
            source = metadata.get("source_file", "Unknown")
            context_parts.append(
                f"[Document {i} - {source}]\n{chunk_text}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""Medical Context:
{context}

User Query: {query}

Based on the medical context provided above, please provide a comprehensive and professional summary."""
        
        print_info(f"Streaming response from {self.model}...")
        print("\n" + "="*50)
        print("📋 MEDICAL SUMMARY")
        print("="*50 + "\n")
        
        # Stream the response
        full_response = ""
        input_tokens = 0
        output_tokens = 0
        
        try:
            with self.client.messages.stream(
                max_tokens=config.MAX_TOKENS,
                system=config.SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_response += text
                
                # Get final message for token usage
                final_message = stream.get_final_message()
                input_tokens = final_message.usage.input_tokens
                output_tokens = final_message.usage.output_tokens
        
        except Exception as e:
            print_error(f"\nError during summarization: {str(e)}")
            return ""
        
        print("\n" + "="*50 + "\n")
        
        # Print token usage
        total_tokens = input_tokens + output_tokens
        print_info(f"Input tokens: {input_tokens}")
        print_info(f"Output tokens: {output_tokens}")
        print_info(f"Total tokens: {total_tokens}")
        
        print_success("Summarization complete")
        
        return full_response
