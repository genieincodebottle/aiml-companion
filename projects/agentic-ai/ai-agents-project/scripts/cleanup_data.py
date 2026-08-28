#!/usr/bin/env python3
"""Clean up all generated data and reset to fresh state.

Removes the SQLite cache and __pycache__ directories so the
project can be tested from a clean starting point.

Run: python scripts/cleanup_data.py
"""

import os
import shutil

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DB = os.path.join(PROJECT_ROOT, "data", "research_cache.db")


def main():
    answer = input("Are you sure you want to delete all cached data? (y/N): ").strip().lower()
    if answer != "y":
        print("Cleanup cancelled.")
        return

    deleted = []

    # Remove SQLite cache
    if os.path.exists(CACHE_DB):
        os.remove(CACHE_DB)
        deleted.append("data/research_cache.db")

    # Clear __pycache__ directories recursively
    pycache_count = 0
    for dirpath, dirnames, _ in os.walk(PROJECT_ROOT):
        for dirname in dirnames:
            if dirname == "__pycache__":
                full_path = os.path.join(dirpath, dirname)
                shutil.rmtree(full_path)
                pycache_count += 1
        # Prune so os.walk doesn't descend into deleted dirs
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]

    if pycache_count:
        deleted.append(f"{pycache_count} __pycache__ directories")

    if deleted:
        print("Deleted:")
        for item in deleted:
            print(f"  - {item}")
    else:
        print("Nothing to clean up. Already fresh.")


if __name__ == "__main__":
    main()
