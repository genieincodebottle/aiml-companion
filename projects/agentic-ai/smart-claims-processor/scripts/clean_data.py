"""
Clean claims data while preserving users and policies.

Usage:
    python scripts/clean_data.py              # Clean claims, appeals, HITL queue
    python scripts/clean_data.py --all        # Delete ALL DBs (full reset, loses users too)
    python scripts/clean_data.py --dry-run    # Show what would be deleted without doing it

What gets cleaned (default):
    - All claims (claim table in api.db)
    - All appeals (appeal table in api.db)
    - HITL queue (hitl_queue.db)
    - LangGraph checkpoints (claims_checkpoints.db)
    - Audit logs (data/audit_logs/)

What is preserved (default):
    - Users (user table in api.db)
    - Policies (policies.db)
"""
from __future__ import annotations

import argparse
import sqlite3
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
API_DB = DATA_DIR / "api.db"
HITL_DB = DATA_DIR / "hitl_queue.db"
POLICIES_DB = DATA_DIR / "policies.db"
CHECKPOINTS_DB = DATA_DIR / "claims_checkpoints.db"
AUDIT_DIR = DATA_DIR / "audit_logs"
MEMORY_DIR = DATA_DIR / "memory"


def _delete_file(path: Path, dry_run: bool) -> None:
    if path.exists():
        if dry_run:
            print(f"  [dry-run] Would delete: {path}")
        else:
            path.unlink()
            print(f"  Deleted: {path}")


def clean_claims_only(dry_run: bool = False) -> None:
    """Delete claims + appeals from api.db, keep users intact."""
    if not API_DB.exists():
        print("  api.db not found - nothing to clean")
        return

    if dry_run:
        conn = sqlite3.connect(str(API_DB))
        claims = conn.execute("SELECT COUNT(*) FROM claim").fetchone()[0]
        appeals = conn.execute("SELECT COUNT(*) FROM appeal").fetchone()[0]
        users = conn.execute("SELECT COUNT(*) FROM user").fetchone()[0]
        conn.close()
        print(f"  [dry-run] Would delete {claims} claims, {appeals} appeals (keeping {users} users)")
        return

    conn = sqlite3.connect(str(API_DB))
    try:
        conn.execute("DELETE FROM claim")
        conn.execute("DELETE FROM appeal")
        conn.commit()
        deleted_claims = conn.total_changes
        print(f"  Cleared claims + appeals from api.db")
    finally:
        conn.close()


def clean_all(dry_run: bool = False) -> None:
    """Delete ALL databases - full reset."""
    for db in [API_DB, HITL_DB, POLICIES_DB, CHECKPOINTS_DB]:
        _delete_file(db, dry_run)


def clean_supporting(dry_run: bool = False) -> None:
    """Delete HITL queue, checkpoints, audit logs."""
    _delete_file(HITL_DB, dry_run)
    _delete_file(CHECKPOINTS_DB, dry_run)

    if AUDIT_DIR.exists():
        if dry_run:
            count = sum(1 for _ in AUDIT_DIR.rglob("*") if _.is_file())
            print(f"  [dry-run] Would delete {count} audit log files in {AUDIT_DIR}")
        else:
            shutil.rmtree(AUDIT_DIR, ignore_errors=True)
            AUDIT_DIR.mkdir(parents=True, exist_ok=True)
            print(f"  Cleared audit logs: {AUDIT_DIR}")

    if MEMORY_DIR.exists():
        if dry_run:
            count = sum(1 for _ in MEMORY_DIR.rglob("*") if _.is_file())
            print(f"  [dry-run] Would delete {count} memory files in {MEMORY_DIR}")
        else:
            shutil.rmtree(MEMORY_DIR, ignore_errors=True)
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            print(f"  Cleared memory store: {MEMORY_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Clean Smart Claims Processor data")
    parser.add_argument("--all", action="store_true", help="Delete ALL DBs including users and policies (full reset)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without doing it")
    args = parser.parse_args()

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Smart Claims Data Cleanup")
    print("=" * 50)

    if args.all:
        print("\nMode: FULL RESET (deleting everything)")
        clean_all(args.dry_run)
        clean_supporting(args.dry_run)
    else:
        print("\nMode: Clean claims data (preserving users + policies)")
        print("\n1. Claims & Appeals:")
        clean_claims_only(args.dry_run)
        print("\n2. Supporting data:")
        clean_supporting(args.dry_run)

    print("\nDone!" + (" (dry run - no changes made)" if args.dry_run else ""))
    if not args.dry_run and not args.all:
        print("Tip: Restart the server to re-initialize tables.\n")


if __name__ == "__main__":
    main()
