#!/usr/bin/env python3
"""
Cleanly reset project structure for compression-galaxy-morphology
according to research-grade standards.

Run this script from the project root:
compression-galaxy-morphology/
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Desired directory structure
# ------------------------------------------------------------

DIRS_TO_CREATE = [
    "data/raw",
    "results/distances",
    "results/rms",
    "results/figures",
    "src/experiments",
    "notebooks",
]

# ------------------------------------------------------------
# Files to KEEP (relative to ROOT)
# ------------------------------------------------------------

FILES_TO_KEEP = {
    "src/compression.py",
    "src/preprocess.py",
    "src/nearest_neighbors.py",
    "README.md",
    "requirements.txt",
}

# ------------------------------------------------------------
# Files to CREATE (empty placeholders if missing)
# ------------------------------------------------------------

FILES_TO_CREATE = [
    "src/clustering.py",
    "src/metrics.py",
    "src/experiments/test_distances.py",
    "src/experiments/rms_vs_k.py",
]

# ------------------------------------------------------------
# Files / folders to REMOVE (explicit, conservative)
# ------------------------------------------------------------

PATHS_TO_REMOVE = [
    "src/debug.py",
    "src/app.py",
    "src/build_features.py",
    "src/knn_classify.py",
    "src/knn_embed.py",
    "src/svm_rbf_classify.py",
    "src/dataDownloader.py",
    "src/python_precompute_C.py",
    "src/output",
    "src/plots",
    "src/results",
    "results/output",
    "results/plots",
]

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def safe_remove(path: Path):
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        print(f"Removed directory: {path}")
    else:
        path.unlink()
        print(f"Removed file: {path}")

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------

def main():
    print("\n=== Resetting project structure ===\n")

    # Create directories
    for d in DIRS_TO_CREATE:
        p = ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory: {p}")

    # Remove unwanted paths
    for rel in PATHS_TO_REMOVE:
        safe_remove(ROOT / rel)

    # Create missing placeholder files
    for rel in FILES_TO_CREATE:
        p = ROOT / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                '"""\n'
                f"{p.name}\n"
                "Auto-generated placeholder.\n"
                "To be implemented.\n"
                '"""\n'
            )
            print(f"Created file: {p}")

    # Warn about unexpected files in src/
    print("\n--- Sanity check: src/ contents ---")
    for p in (ROOT / "src").iterdir():
        rel = str(p.relative_to(ROOT))
        if rel not in FILES_TO_KEEP and not p.is_dir():
            print(f"⚠️  Review manually: {rel}")

    print("\n=== Structure reset complete ===\n")
    print("Next steps:")
    print("1) Implement test_distances.py")
    print("2) Implement metrics.py (RMS-within-cluster)")
    print("3) Commit to NEW GitHub repo")

if __name__ == "__main__":
    main()
