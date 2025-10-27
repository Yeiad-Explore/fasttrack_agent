#!/usr/bin/env python3
"""
Knowledge Base Loader Script
Embeds all PDF and Markdown files from the kb/ directory into the vector store
Handles deduplication automatically

Load New Files
# Add files to kb/ directory, then run:
python load_kb.py
Check Status
python load_kb.py --stats
List Files
python load_kb.py --list
Force Reindex
python load_kb.py --force
Get Help
python load_kb.py --help








"""

import sys
from pathlib import Path
import argparse
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import settings
from src.ingestion import ingestion_pipeline
from src.vector_store import vector_store


def print_banner():
    """Print script banner"""
    print("=" * 70)
    print("  FASTTRACK KB LOADER - Knowledge Base Vectorization Tool")
    print("=" * 70)
    print()


def print_current_stats():
    """Print current vector store statistics"""
    stats = vector_store.get_stats()

    print("\n" + "─" * 70)
    print("📊 CURRENT VECTOR STORE STATUS")
    print("─" * 70)
    print(f"  Total Chunks:     {stats['total_documents']}")
    print(f"  Total Files:      {stats['total_files']}")
    print(f"  Index Size:       {stats['index_size']}")
    print(f"  Dimension:        {stats['dimension']}")

    if stats.get('file_types'):
        print(f"\n  File Types:")
        for file_type, count in stats['file_types'].items():
            print(f"    - {file_type}: {count} chunks")

    if stats.get('indexed_files'):
        print(f"\n  Indexed Files:")
        for file_path in stats['indexed_files']:
            print(f"    - {Path(file_path).name}")

    print("─" * 70)


def load_kb(kb_dir: Path = None, force_reindex: bool = False, skip_duplicates: bool = True):
    """
    Load all knowledge base files into vector store

    Args:
        kb_dir: Path to KB directory (defaults to settings.KB_DIR)
        force_reindex: If True, clear existing index before loading
        skip_duplicates: If True, skip already indexed files
    """
    if kb_dir is None:
        kb_dir = settings.KB_DIR

    print(f"📁 KB Directory: {kb_dir}")
    print(f"🔍 Duplicate Check: {'Enabled' if skip_duplicates else 'Disabled'}")
    print(f"🔄 Force Reindex: {'Yes' if force_reindex else 'No'}")
    print()

    # Show current status
    print_current_stats()

    # Clear index if force reindex
    if force_reindex:
        print("\n⚠️  CLEARING EXISTING INDEX...")
        ingestion_pipeline.clear_index()
        print("✅ Index cleared")
        print()

    # Load files
    print("\n🚀 STARTING INGESTION PROCESS")
    print("─" * 70)

    try:
        result = ingestion_pipeline.ingest_directory(
            directory=kb_dir,
            check_duplicate=skip_duplicates
        )

        print("\n" + "─" * 70)
        print("✅ INGESTION COMPLETE")
        print("─" * 70)
        print(f"  Files Found:      {result['total_files']}")
        print(f"  Chunks Created:   {result['total_chunks']}")
        print(f"  Duplicates Skipped: {result.get('skipped_duplicates', 0)}")
        print()

        # Show detailed results
        if result['files_processed']:
            print("  File Processing Results:")
            for file_info in result['files_processed']:
                status_icon = {
                    'success': '✅',
                    'skipped': '⏭️ ',
                    'failed': '❌'
                }.get(file_info['status'], '❓')

                filename = file_info['filename']
                chunks = file_info.get('chunks', 0)
                status = file_info['status']

                if status == 'success':
                    print(f"    {status_icon} {filename}: {chunks} chunks")
                elif status == 'skipped':
                    reason = file_info.get('reason', 'unknown')
                    print(f"    {status_icon} {filename}: {reason}")
                elif status == 'failed':
                    error = file_info.get('error', 'unknown error')
                    print(f"    {status_icon} {filename}: {error}")

        print("─" * 70)

        # Show final stats
        print_current_stats()

        print("\n✨ Knowledge base successfully loaded!")
        print(f"📦 Vector store saved to: {settings.VECTOR_STORE_DIR}")

        return result

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def list_files(kb_dir: Path = None):
    """List all files in the KB directory"""
    if kb_dir is None:
        kb_dir = settings.KB_DIR

    print(f"\n📁 Files in {kb_dir}:")
    print("─" * 70)

    # Get all supported files
    all_files = []
    for ext in ['.pdf', '.md', '.markdown']:
        all_files.extend(kb_dir.glob(f"*{ext}"))

    if not all_files:
        print("  No files found")
        return

    # Get indexed files
    indexed_files = set(vector_store.get_indexed_files())

    for file_path in sorted(all_files):
        file_size = file_path.stat().st_size / 1024  # KB
        is_indexed = str(file_path) in indexed_files
        status_icon = "✅" if is_indexed else "⚪"
        status_text = "indexed" if is_indexed else "not indexed"

        print(f"  {status_icon} {file_path.name}")
        print(f"     Size: {file_size:.2f} KB | Type: {file_path.suffix} | Status: {status_text}")

    print("─" * 70)
    print(f"  Total: {len(all_files)} files ({len(indexed_files)} indexed)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Load and vectorize knowledge base files (PDF, Markdown)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load all files (skip duplicates)
  python load_kb.py

  # Force reindex everything
  python load_kb.py --force

  # Allow duplicates
  python load_kb.py --no-skip-duplicates

  # List files without indexing
  python load_kb.py --list

  # Load from custom directory
  python load_kb.py --kb-dir /path/to/kb
        """
    )

    parser.add_argument(
        '--kb-dir',
        type=Path,
        default=None,
        help=f'Path to KB directory (default: {settings.KB_DIR})'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Clear existing index before loading (force reindex)'
    )

    parser.add_argument(
        '--no-skip-duplicates',
        action='store_true',
        help='Disable duplicate checking (allow re-indexing same files)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List files in KB directory without indexing'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show current vector store statistics'
    )

    args = parser.parse_args()

    print_banner()

    # Handle different modes
    if args.stats:
        print_current_stats()
    elif args.list:
        list_files(args.kb_dir)
        print_current_stats()
    else:
        # Load KB
        result = load_kb(
            kb_dir=args.kb_dir,
            force_reindex=args.force,
            skip_duplicates=not args.no_skip_duplicates
        )

        if result is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
