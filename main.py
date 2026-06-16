#!/usr/bin/env python3
"""CLI entry point for Medical RAG Summarizer."""

import os
import sys
import argparse
from dotenv import load_dotenv
from pipeline import RAGPipeline
from utils import print_success, print_error, print_info, validate_api_key

# Load environment variables from .env file
load_dotenv()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Medical RAG Summarizer - Terminal-based document analysis with Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest --path ./sample_docs
  python main.py query --question "What medications were prescribed?"
  python main.py ingest-and-query --path ./sample_docs --question "Summarize the patient diagnosis"
  python main.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents and build vector index"
    )
    ingest_parser.add_argument(
        "--path",
        default="./sample_docs",
        help="Path to documents folder or file (default: ./sample_docs)"
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query existing index and get summary"
    )
    query_parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the medical documents"
    )
    
    # Ingest and query command
    combined_parser = subparsers.add_parser(
        "ingest-and-query",
        help="Ingest documents and query in one command"
    )
    combined_parser.add_argument(
        "--path",
        default="./sample_docs",
        help="Path to documents folder or file (default: ./sample_docs)"
    )
    combined_parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the medical documents"
    )
    
    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show index statistics"
    )
    
    args = parser.parse_args()
    
    # Show banner
    print("\n" + "="*60)
    print("🏥 MEDICAL RAG SUMMARIZER")
    print("Retrieval-Augmented Generation for Medical Documents")
    print("="*60 + "\n")
    
    if not args.command:
        parser.print_help()
        return
    
    # Validate API key for commands that need it
    if args.command in ["query", "ingest-and-query"]:
        if not validate_api_key():
            sys.exit(1)
    
    try:
        pipeline = RAGPipeline()
        
        if args.command == "ingest":
            print_info(f"Starting ingestion from: {args.path}")
            if pipeline.ingest_and_index(args.path):
                print_success("Ingestion complete! Index saved to disk.")
            else:
                print_error("Ingestion failed")
                sys.exit(1)
        
        elif args.command == "query":
            print_info(f"Question: {args.question}")
            result = pipeline.query_and_summarize(args.question)
            if result is None:
                sys.exit(1)
        
        elif args.command == "ingest-and-query":
            print_info(f"Starting ingestion from: {args.path}")
            if not pipeline.ingest_and_index(args.path):
                print_error("Ingestion failed")
                sys.exit(1)
            
            print("\n" + "="*60)
            print_info(f"Question: {args.question}")
            result = pipeline.query_and_summarize(args.question)
            if result is None:
                sys.exit(1)
        
        elif args.command == "stats":
            pipeline.show_stats()
        
        print("\n" + "="*60)
        print_success("Done!")
        print("="*60 + "\n")
    
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print_info("Operation cancelled by user")
        print("="*60 + "\n")
        sys.exit(0)
    
    except Exception as e:
        print("\n" + "="*60)
        print_error(f"Fatal error: {str(e)}")
        print("="*60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
