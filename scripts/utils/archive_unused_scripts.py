#!/usr/bin/env python
# filepath: /Users/zhang/Desktop/huawei/infi_llama/nanotron-infini/scripts/utils/archive_unused_scripts.py
"""
This utility script identifies and archives older or unused scripts in the Infini-Llama workflow.
Rather than deleting files, it moves them to an 'archived' directory to ensure they're preserved.

Usage:
    python scripts/utils/archive_unused_scripts.py [--dry-run]
    
Options:
    --dry-run    Only list files that would be archived, but don't move them
"""

import os
import sys
import shutil
import argparse
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Archive unused scripts in the Infini-Llama workflow")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be archived, don't move files")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get project root
    project_root = Path(__file__).resolve().parents[2]
    scripts_dir = project_root / "scripts"
    archive_dir = scripts_dir / "archived"
    
    # Create archive directory if it doesn't exist
    if not args.dry_run:
        archive_dir.mkdir(exist_ok=True)
    
    # Files to archive - list of (file_path, reason) tuples
    to_archive = [
        # Older workflow script now replaced by flexible_training_workflow.sh
        (scripts_dir / "run_infini_llama_workflow.sh", "Replaced by flexible_training_workflow.sh"),
        
        # Look for old preprocessing script (assuming we're using preprocess_data_fixed.py now)
        (scripts_dir / "preprocessing" / "preprocess_data.py", "Replaced by preprocess_data_fixed.py"),
        
        # Any other files you identify as unused or deprecated
        # (scripts_dir / "path" / "to" / "unused_file.py", "Reason for archiving"),
    ]
    
    # Process each file
    archived_count = 0
    for file_path, reason in to_archive:
        if file_path.exists():
            if args.dry_run:
                print(f"Would archive: {file_path.relative_to(project_root)} - {reason}")
            else:
                # Create archive destination with timestamp
                timestamp = datetime.now().strftime("%Y%m%d")
                archive_path = archive_dir / f"{file_path.name}.{timestamp}"
                
                # Copy the file to archive with timestamp
                shutil.copy2(file_path, archive_path)
                
                # Add comment at the top of the file noting it's archived
                with open(archive_path, 'r+') as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write(f"# ARCHIVED: {datetime.now().strftime('%Y-%m-%d')} - {reason}\n")
                    f.write(content)
                
                print(f"Archived: {file_path.relative_to(project_root)} -> {archive_path.relative_to(project_root)}")
                archived_count += 1
        else:
            print(f"File not found: {file_path.relative_to(project_root)}")
    
    if archived_count > 0:
        print(f"\nArchived {archived_count} files to {archive_dir.relative_to(project_root)}/")
    else:
        print("\nNo files were archived.")
    
    if args.dry_run:
        print("\nThis was a dry run. No files were actually moved.")
        print("Run without --dry-run to perform the archiving operation.")

if __name__ == "__main__":
    main()
