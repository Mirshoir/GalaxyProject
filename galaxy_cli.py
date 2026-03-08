#!/usr/bin/env python3
"""
Galaxy Morphology Analyzer - Terminal CLI
==========================================
Compression-based galaxy morphology analysis for large datasets.
Uses Normalized Compression Distance (NCD) for unsupervised classification.

Usage:
    python galaxy_cli.py <command> [options]

Commands:
    scan        Scan directory for images
    preprocess  Preprocess images
    compress    Test compression/NCD on pairs
    distances   Compute full distance matrix
    neighbors   Find nearest neighbors
    cluster     Cluster by morphology
    rms         RMS vs k analysis
    pipeline    Run full pipeline end-to-end
    info        Show dataset info / session state
"""

import os
import sys
import json
import time
import gzip
import zlib
import bz2
import lzma
import hashlib
import argparse
import pickle
import math
import struct
import itertools
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Any

import numpy as np

# ─────────────────────────────────────────────
#  Terminal helpers
# ─────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"

CLUSTER_COLORS = [RED, GREEN, YELLOW, CYAN, MAGENTA, BLUE, WHITE]

def c(text, color): return f"{color}{text}{RESET}"
def bold(text):     return f"{BOLD}{text}{RESET}"
def dim(text):      return f"{DIM}{text}{RESET}"

def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗
║        🌌  Galaxy Morphology Analyzer  (CLI)             ║
║   Normalized Compression Distance — Unsupervised         ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")

def header(title: str):
    w = 60
    print(f"\n{BLUE}{BOLD}{'─'*w}{RESET}")
    print(f"{BLUE}{BOLD}  {title}{RESET}")
    print(f"{BLUE}{BOLD}{'─'*w}{RESET}")

def success(msg):  print(f"  {GREEN}✔  {msg}{RESET}")
def warn(msg):     print(f"  {YELLOW}⚠  {msg}{RESET}")
def error(msg):    print(f"  {RED}✘  {msg}{RESET}")
def info(msg):     print(f"  {CYAN}ℹ  {msg}{RESET}")
def step(msg):     print(f"  {MAGENTA}→  {msg}{RESET}")


class ProgressBar:
    """Thread-safe ASCII progress bar."""
    def __init__(self, total: int, label: str = "", width: int = 40):
        self.total   = max(total, 1)
        self.current = 0
        self.label   = label
        self.width   = width
        self.start   = time.time()
        self._lock   = threading.Lock()
        self._draw(0)

    def update(self, n: int = 1):
        with self._lock:
            self.current = min(self.current + n, self.total)
            self._draw(self.current)

    def _draw(self, current):
        pct  = current / self.total
        done = int(self.width * pct)
        bar  = "█" * done + "░" * (self.width - done)
        elapsed = time.time() - self.start
        eta = (elapsed / pct - elapsed) if pct > 0 else 0
        sys.stdout.write(
            f"\r  {CYAN}{bar}{RESET} {pct:5.1%}  "
            f"{current}/{self.total}  "
            f"ETA {eta:5.0f}s  {self.label}    "
        )
        sys.stdout.flush()

    def finish(self):
        elapsed = time.time() - self.start
        sys.stdout.write(f"\r  {GREEN}{'█'*self.width}{RESET} 100.0%  done in {elapsed:.1f}s\n")
        sys.stdout.flush()


def table(rows: List[Dict], cols: Optional[List[str]] = None, max_rows: int = 999):
    """Pretty-print a list-of-dicts as a table."""
    if not rows:
        print(dim("  (no data)"))
        return
    if cols is None:
        cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    sep = "  " + "  ".join("─" * widths[c] for c in cols)
    header_row = "  " + "  ".join(f"{BOLD}{str(c):<{widths[c]}}{RESET}" for c in cols)
    print(sep)
    print(header_row)
    print(sep)
    for i, row in enumerate(rows[:max_rows]):
        print("  " + "  ".join(f"{str(row.get(c,'')):<{widths[c]}}" for c in cols))
    if len(rows) > max_rows:
        print(dim(f"  … {len(rows)-max_rows} more rows …"))
    print(sep)


# ─────────────────────────────────────────────
#  Image loading (pure stdlib + optional cv2/PIL)
# ─────────────────────────────────────────────

def _load_image_bytes_raw(path: str) -> bytes:
    """Load image file bytes directly."""
    with open(path, "rb") as f:
        return f.read()


def _load_image_numpy(path: str, size: int = 128) -> Optional[np.ndarray]:
    """Load and resize image to grayscale numpy array. Falls back gracefully."""
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("cv2 returned None")
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        pass
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize((size, size))
        return np.array(img, dtype=np.uint8)
    except Exception:
        pass
    # Fallback: treat raw bytes as data
    return None


def canonical_bytes(path: str, size: int = 128,
                    normalize: bool = True,
                    apply_blur: bool = True,
                    sigma: float = 1.0) -> bytes:
    """Convert image to canonical byte representation for NCD."""
    arr = _load_image_numpy(path, size)
    if arr is None:
        return _load_image_bytes_raw(path)

    if normalize:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

    if apply_blur and sigma > 0:
        try:
            import cv2
            arr = cv2.GaussianBlur(arr, (0, 0), sigma)
        except Exception:
            pass  # skip blur if cv2 unavailable

    return arr.tobytes()


# ─────────────────────────────────────────────
#  Compression
# ─────────────────────────────────────────────

COMPRESSORS = {
    "zlib": lambda data: zlib.compress(data, level=9),
    "bz2":  lambda data: bz2.compress(data, compresslevel=9),
    "lzma": lambda data: lzma.compress(data, preset=9),
}


def compress(data: bytes, algo: str = "zlib") -> int:
    """Return compressed length in bytes."""
    fn = COMPRESSORS.get(algo)
    if fn is None:
        raise ValueError(f"Unknown algorithm: {algo}. Choose from {list(COMPRESSORS)}")
    return len(fn(data))


def ncd(bytes_x: bytes, bytes_y: bytes, algo: str = "zlib") -> float:
    """
    Normalized Compression Distance:
        NCD(x,y) = (C(xy) - min(C(x),C(y))) / max(C(x),C(y))
    """
    cx  = compress(bytes_x, algo)
    cy  = compress(bytes_y, algo)
    cxy = compress(bytes_x + bytes_y, algo)
    denom = max(cx, cy)
    if denom == 0:
        return 0.0
    return (cxy - min(cx, cy)) / denom


# ─────────────────────────────────────────────
#  Session / cache
# ─────────────────────────────────────────────

SESSION_FILE = "galaxy_session.pkl"


def load_session() -> Dict:
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def save_session(sess: Dict):
    with open(SESSION_FILE, "wb") as f:
        pickle.dump(sess, f)
    info(f"Session saved → {SESSION_FILE}")


# ─────────────────────────────────────────────
#  Commands
# ─────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".fits"}


def cmd_scan(args):
    """Scan a directory and save file list to session."""
    header("📁  Scan Directory")

    directory = args.directory
    if not os.path.isdir(directory):
        error(f"Directory not found: {directory}")
        sys.exit(1)

    files = sorted(
        str(p) for p in Path(directory).rglob("*")
        if p.suffix.lower() in IMG_EXTS
    )

    if not files:
        warn("No image files found.")
        return

    success(f"Found {len(files)} images in {directory}")

    # Basic stats
    total_bytes = sum(os.path.getsize(f) for f in files)
    ext_counts  = Counter(Path(f).suffix.lower() for f in files)

    rows = [{"Extension": k, "Count": v} for k, v in ext_counts.items()]
    table(rows, cols=["Extension", "Count"])

    info(f"Total size: {total_bytes / 1_048_576:.1f} MB")
    info(f"Preview (first 10):")
    for f in files[:10]:
        print(f"    {dim(f)}")
    if len(files) > 10:
        print(dim(f"    … {len(files)-10} more"))

    sess = load_session()
    sess["files"] = files
    sess["scan_dir"] = directory
    save_session(sess)


def cmd_preprocess(args):
    """Test preprocessing on one image or report stats for all."""
    header("🛠  Preprocessing")

    sess = load_session()
    files = sess.get("files", [])
    if not files:
        error("No files in session. Run 'scan' first.")
        sys.exit(1)

    size   = args.size
    sample = min(args.sample, len(files))

    step(f"Image size: {size}×{size}")
    step(f"Normalize: {args.normalize}")
    step(f"Blur sigma: {args.sigma}")

    rows = []
    bar  = ProgressBar(sample, "preprocessing")
    for path in files[:sample]:
        try:
            t0 = time.time()
            b  = canonical_bytes(path, size=size,
                                 normalize=args.normalize,
                                 apply_blur=args.sigma > 0,
                                 sigma=args.sigma)
            elapsed = time.time() - t0
            rows.append({
                "File"   : Path(path).name[:40],
                "Bytes"  : len(b),
                "Time ms": f"{elapsed*1000:.1f}",
                "Status" : "OK"
            })
        except Exception as e:
            rows.append({"File": Path(path).name[:40], "Bytes": 0,
                         "Time ms": "—", "Status": f"ERR: {e}"})
        bar.update()
    bar.finish()

    table(rows, max_rows=20)

    ok  = sum(1 for r in rows if r["Status"] == "OK")
    err = sample - ok
    success(f"{ok} images preprocessed successfully")
    if err:
        warn(f"{err} errors")

    sess["preprocess_config"] = {"size": size, "normalize": args.normalize,
                                  "sigma": args.sigma}
    save_session(sess)


def cmd_compress(args):
    """Test NCD on one or more image pairs."""
    header("⚡  Compression / NCD Test")

    sess  = load_session()
    files = sess.get("files", [])
    cfg   = sess.get("preprocess_config", {"size": 128, "normalize": True, "sigma": 1.0})
    algo  = args.algo

    if args.paths:
        pairs_files = args.paths
        if len(pairs_files) < 2:
            error("Provide at least 2 image paths (or use session files).")
            sys.exit(1)
        test_files = pairs_files
    else:
        if len(files) < 2:
            error("No files in session. Run 'scan' first.")
            sys.exit(1)
        n = min(args.n, len(files))
        test_files = files[:n]

    step(f"Algorithm: {algo}")
    step(f"Images   : {len(test_files)}")

    # Self-distance sanity check
    info("Self-distance sanity check (should be ≈ 0):")
    for path in test_files[:3]:
        b = canonical_bytes(path, **{k: cfg[k] for k in ("size","normalize","sigma")})
        d = ncd(b, b, algo)
        print(f"    {dim(Path(path).name[:45])}  →  self-dist = {d:.6f}  "
              f"{'✔' if d < 0.02 else '⚠ high'}")

    # Pairwise test
    pairs = list(itertools.combinations(range(len(test_files)), 2))
    if len(pairs) > 45:
        import random; random.seed(42); pairs = random.sample(pairs, 45)

    rows = []
    bar  = ProgressBar(len(pairs), f"NCD ({algo})")
    for i, j in pairs:
        bi = canonical_bytes(test_files[i], **{k: cfg[k] for k in ("size","normalize","sigma")})
        bj = canonical_bytes(test_files[j], **{k: cfg[k] for k in ("size","normalize","sigma")})
        d  = ncd(bi, bj, algo)
        rows.append({
            "Image A"  : Path(test_files[i]).name[:30],
            "Image B"  : Path(test_files[j]).name[:30],
            "NCD"      : f"{d:.4f}",
            "Similarity": f"{(1-d)*100:.1f}%",
            "Note"     : "very similar" if d < .1 else ("similar" if d < .3 else
                         ("different" if d < .5 else "very different"))
        })
        bar.update()
    bar.finish()

    table(rows, max_rows=30)

    dists = [float(r["NCD"]) for r in rows]
    print(f"\n  Mean: {np.mean(dists):.4f}  |  Std: {np.std(dists):.4f}  |  "
          f"Min: {np.min(dists):.4f}  |  Max: {np.max(dists):.4f}")


def cmd_distances(args):
    """Compute full pairwise distance matrix."""
    header("📏  Distance Matrix")

    sess  = load_session()
    files = sess.get("files", [])
    cfg   = sess.get("preprocess_config", {"size": 128, "normalize": True, "sigma": 1.0})
    algo  = args.algo

    if not files:
        error("No files in session. Run 'scan' first.")
        sys.exit(1)

    n = min(args.n, len(files))
    step(f"Images     : {n}")
    step(f"Algorithm  : {algo}")
    step(f"Total pairs: {n*(n-1)//2}")

    selected = files[:n]

    # Pre-compute canonical bytes with progress
    info("Pre-computing canonical representations…")
    byte_cache = {}
    bar = ProgressBar(n, "encoding")
    for path in selected:
        byte_cache[path] = canonical_bytes(path, **{k: cfg[k] for k in ("size","normalize","sigma")})
        bar.update()
    bar.finish()

    # Compute distance matrix
    distances = np.zeros((n, n), dtype=np.float32)
    total_pairs = n * (n - 1) // 2
    bar = ProgressBar(total_pairs, f"NCD ({algo})")

    for i in range(n):
        for j in range(i + 1, n):
            d = ncd(byte_cache[selected[i]], byte_cache[selected[j]], algo)
            distances[i, j] = d
            distances[j, i] = d
            bar.update()
    bar.finish()

    # Statistics
    triu = distances[np.triu_indices(n, k=1)]
    print(f"\n  Mean:   {np.mean(triu):.4f}")
    print(f"  Std:    {np.std(triu):.4f}")
    print(f"  Min:    {np.min(triu):.4f}")
    print(f"  Max:    {np.max(triu):.4f}")
    print(f"  Median: {np.median(triu):.4f}")

    # Save distance matrix
    out = args.output or "distances.npy"
    np.save(out, distances)
    success(f"Distance matrix saved → {out}")

    sess["distance_matrix_file"] = out
    sess["selected_files"]       = selected
    save_session(sess)


def cmd_neighbors(args):
    """Find k nearest neighbors for query images."""
    header("🔍  Nearest Neighbors")

    sess = load_session()
    dm_file = sess.get("distance_matrix_file")
    selected = sess.get("selected_files", [])

    if not dm_file or not os.path.exists(dm_file):
        error("No distance matrix found. Run 'distances' first.")
        sys.exit(1)

    distances = np.load(dm_file)
    n = len(distances)

    # Resolve query indices
    if args.query is not None:
        query_indices = [args.query % n]
    else:
        query_indices = list(range(min(args.n_queries, n)))

    k = min(args.k, n - 1)

    for qi in query_indices:
        name = Path(selected[qi]).name if qi < len(selected) else f"Image {qi}"
        print(f"\n  Query: {bold(name)}  (index {qi})")

        row  = distances[qi].copy()
        row[qi] = np.inf
        nbrs = np.argsort(row)[:k]

        rows = []
        for rank, ni in enumerate(nbrs, 1):
            nbr_name = Path(selected[ni]).name if ni < len(selected) else f"Image {ni}"
            d = distances[qi, ni]
            rows.append({
                "Rank"      : rank,
                "Index"     : ni,
                "Filename"  : nbr_name[:45],
                "Distance"  : f"{d:.4f}",
                "Similarity": f"{(1-d)*100:.1f}%",
                "Relation"  : ("≈ identical" if d < .05 else
                               "very similar" if d < .15 else
                               "similar"      if d < .30 else
                               "different"    if d < .55 else "very different")
            })
        table(rows)


def cmd_cluster(args):
    """Cluster images using k-medoids or hierarchical clustering."""
    header("🗃  Clustering")

    sess = load_session()
    dm_file  = sess.get("distance_matrix_file")
    selected = sess.get("selected_files", [])

    if not dm_file or not os.path.exists(dm_file):
        error("No distance matrix found. Run 'distances' first.")
        sys.exit(1)

    distances = np.load(dm_file)
    n = len(distances)
    k = min(args.k, n)

    step(f"Algorithm  : {args.algo}")
    step(f"Clusters k : {k}")
    step(f"Images     : {n}")

    if args.algo == "kmedoids":
        labels = _kmedoids(distances, k, seed=args.seed)
    elif args.algo == "hierarchical":
        labels = _hierarchical(distances, k, linkage=args.linkage)
    else:
        error(f"Unknown algorithm: {args.algo}")
        sys.exit(1)

    # Summary
    counts = Counter(labels)
    rows = []
    for cid in sorted(counts):
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        members = [i for i, l in enumerate(labels) if l == cid]
        medoid_name = ""
        if selected and members:
            medoid_name = Path(selected[members[0]]).name[:40]
        rows.append({
            "Cluster"  : f"{color}{cid}{RESET}",
            "Size"     : counts[cid],
            "Pct"      : f"{counts[cid]/n*100:.1f}%",
            "Sample"   : medoid_name
        })
    table(rows, cols=["Cluster","Size","Pct","Sample"])

    # Intra-cluster distances
    info("Intra-cluster distance stats:")
    for cid in sorted(counts):
        members = [i for i, l in enumerate(labels) if l == cid]
        if len(members) < 2:
            continue
        idx = np.ix_(members, members)
        sub = distances[idx]
        mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
        vals = sub[mask]
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        print(f"    {color}Cluster {cid}{RESET}:  "
              f"mean={np.mean(vals):.3f}  "
              f"std={np.std(vals):.3f}  "
              f"max={np.max(vals):.3f}")

    # Save
    out = args.output or "clusters.json"
    result = {
        "algorithm"   : args.algo,
        "k"           : k,
        "labels"      : labels.tolist(),
        "files"       : selected,
        "cluster_sizes": {str(k): v for k, v in counts.items()}
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    success(f"Cluster assignments saved → {out}")

    sess["cluster_file"] = out
    save_session(sess)


def cmd_rms(args):
    """RMS vs k analysis to find optimal cluster count."""
    header("📈  RMS vs k Analysis")

    sess = load_session()
    dm_file = sess.get("distance_matrix_file")

    if not dm_file or not os.path.exists(dm_file):
        error("No distance matrix found. Run 'distances' first.")
        sys.exit(1)

    distances = np.load(dm_file)
    n         = len(distances)
    k_min     = max(2, args.k_min)
    k_max     = min(args.k_max, n - 1)
    n_runs    = args.runs

    step(f"k range: {k_min} … {k_max}")
    step(f"Runs/k : {n_runs}")
    step(f"Images : {n}")

    k_values  = list(range(k_min, k_max + 1))
    mean_rms  = []
    std_rms   = []

    total_iters = len(k_values) * n_runs
    bar = ProgressBar(total_iters, "RMS analysis")

    for k in k_values:
        run_rms = []
        for seed in range(n_runs):
            labels = _kmedoids(distances, k, seed=seed)
            rms    = _compute_rms(distances, labels)
            run_rms.append(rms)
            bar.update()
        mean_rms.append(float(np.mean(run_rms)))
        std_rms.append(float(np.std(run_rms)))
    bar.finish()

    # Find elbow
    best_k = _find_elbow(k_values, mean_rms)

    # Print table
    rows = []
    for k, mu, sigma in zip(k_values, mean_rms, std_rms):
        rows.append({
            "k"      : k,
            "Mean RMS": f"{mu:.4f}",
            "Std Dev" : f"{sigma:.4f}",
            "Elbow"  : "⭐ suggested" if k == best_k else ""
        })
    table(rows, cols=["k","Mean RMS","Std Dev","Elbow"])

    success(f"Suggested optimal k = {best_k}")

    # Save JSON
    out = args.output or "rms_analysis.json"
    result = {
        "k_values" : k_values,
        "mean_rms" : mean_rms,
        "std_rms"  : std_rms,
        "best_k"   : best_k
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    success(f"Results saved → {out}")

    # ── Matplotlib elbow plot ─────────────────────────────────
    plot_out = Path(out).stem + "_elbow.png"
    _plot_elbow(k_values, mean_rms, std_rms, best_k, plot_out)

    sess["rms_file"] = out
    save_session(sess)


def cmd_pipeline(args):
    """Run full pipeline end-to-end."""
    header("🚀  Full Pipeline")

    step(f"Directory : {args.directory}")
    step(f"Images    : up to {args.n}")
    step(f"Clusters  : {args.k}")
    step(f"Algorithm : {args.algo}")

    # 1 — Scan
    info("Step 1/5: Scanning directory…")
    args_scan = argparse.Namespace(directory=args.directory)
    cmd_scan(args_scan)

    # 2 — Preprocess config
    info("Step 2/5: Configuring preprocessing…")
    sess = load_session()
    sess["preprocess_config"] = {
        "size"     : args.img_size,
        "normalize": True,
        "sigma"    : 1.0
    }
    save_session(sess)
    success(f"Image size {args.img_size}×{args.img_size}, normalize=True, sigma=1.0")

    # 3 — Distances
    info("Step 3/5: Computing distance matrix…")
    args_dist = argparse.Namespace(
        n=args.n, algo=args.algo, output="distances.npy"
    )
    cmd_distances(args_dist)

    # 4 — RMS
    info("Step 4/5: RMS analysis…")
    args_rms = argparse.Namespace(
        k_min=2, k_max=min(args.k + 3, args.n - 1),
        runs=2, output="rms_analysis.json"
    )
    cmd_rms(args_rms)

    # 5 — Cluster
    info("Step 5/5: Clustering…")
    args_clust = argparse.Namespace(
        k=args.k, algo="kmedoids", linkage="average",
        seed=42, output="clusters.json"
    )
    cmd_cluster(args_clust)

    success("Pipeline complete! Outputs: distances.npy, rms_analysis.json, clusters.json")


def cmd_info(args):
    """Show current session state."""
    header("ℹ  Session Info")

    sess = load_session()

    if not sess:
        warn("No session found. Run 'scan' to start.")
        return

    rows = []
    for k, v in sess.items():
        if k == "files":
            rows.append({"Key": k, "Value": f"[{len(v)} files]"})
        elif k == "distance_matrix_file":
            dm_info = ""
            if v and os.path.exists(v):
                m = np.load(v)
                dm_info = f"{m.shape[0]}×{m.shape[1]}"
            rows.append({"Key": k, "Value": f"{v}  ({dm_info})"})
        elif isinstance(v, dict):
            rows.append({"Key": k, "Value": json.dumps(v)[:80]})
        else:
            rows.append({"Key": k, "Value": str(v)[:80]})

    table(rows, cols=["Key", "Value"])

    if "files" in sess:
        info(f"Sample files:")
        for f in sess["files"][:5]:
            print(f"    {dim(f)}")


# ─────────────────────────────────────────────
#  Clustering algorithms (pure numpy, no sklearn)
# ─────────────────────────────────────────────

def _kmedoids(distances: np.ndarray, k: int, seed: int = 42,
              max_iter: int = 100) -> np.ndarray:
    """Simple PAM-style k-medoids using precomputed distance matrix."""
    n = len(distances)
    rng = np.random.default_rng(seed)

    # Init medoids randomly
    medoids = rng.choice(n, size=k, replace=False).tolist()

    for _ in range(max_iter):
        # Assign each point to nearest medoid
        labels = np.argmin(distances[:, medoids], axis=1)

        # Update medoids
        new_medoids = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                new_medoids.append(medoids[c])
                continue
            # Pick member with minimum total distance to others in cluster
            sub = distances[np.ix_(members, members)]
            best = members[np.argmin(sub.sum(axis=1))]
            new_medoids.append(int(best))

        if new_medoids == medoids:
            break
        medoids = new_medoids

    return np.argmin(distances[:, medoids], axis=1)


def _hierarchical(distances: np.ndarray, k: int,
                  linkage: str = "average") -> np.ndarray:
    """Agglomerative hierarchical clustering from precomputed distances."""
    n = len(distances)
    # Each point starts as its own cluster
    clusters = [{i} for i in range(n)]
    dist = distances.copy().astype(np.float64)
    np.fill_diagonal(dist, np.inf)
    ids = list(range(n))   # maps cluster index to original label

    while len(clusters) > k:
        # Find closest pair
        best_i, best_j, best_d = 0, 1, np.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                mi = list(clusters[i])
                mj = list(clusters[j])
                sub = distances[np.ix_(mi, mj)]
                if linkage == "single":
                    d = sub.min()
                elif linkage == "complete":
                    d = sub.max()
                else:  # average / ward
                    d = sub.mean()
                if d < best_d:
                    best_d, best_i, best_j = d, i, j

        # Merge
        clusters[best_i] = clusters[best_i] | clusters[best_j]
        clusters.pop(best_j)

    labels = np.zeros(n, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            labels[m] = cid
    return labels


def _compute_rms(distances: np.ndarray, labels: np.ndarray) -> float:
    """Average RMS intra-cluster distance."""
    rms_vals = []
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        if len(members) < 2:
            rms_vals.append(0.0)
            continue
        idx = np.ix_(members, members)
        sub = distances[idx]
        mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
        vals = sub[mask]
        rms_vals.append(float(np.sqrt(np.mean(vals ** 2))))
    return float(np.mean(rms_vals)) if rms_vals else 0.0


def _find_elbow(k_values: List[int], rms: List[float]) -> int:
    """Simple elbow detection via maximum second derivative."""
    if len(k_values) < 3:
        return k_values[np.argmin(rms)]
    diffs2 = [rms[i] - 2 * rms[i+1] + rms[i+2] for i in range(len(rms)-2)]
    best_idx = int(np.argmax(diffs2)) + 1
    return k_values[best_idx]


# ─────────────────────────────────────────────
#  ASCII visualisations
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  Matplotlib / pyplot PNG plots
# ─────────────────────────────────────────────

def _get_plt():
    """Import matplotlib with non-interactive Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor"  : "#161b22",
        "axes.edgecolor"  : "#30363d",
        "axes.labelcolor" : "#c9d1d9",
        "axes.titlecolor" : "#e6edf3",
        "xtick.color"     : "#8b949e",
        "ytick.color"     : "#8b949e",
        "text.color"      : "#c9d1d9",
        "grid.color"      : "#21262d",
        "grid.linestyle"  : "--",
        "grid.alpha"      : 0.6,
        "font.family"     : "monospace",
    })
    return plt


def _plot_elbow(k_values: List[int], mean_rms: List[float],
                std_rms: List[float], best_k: int, out: str):
    """Elbow / RMS-vs-k plot saved as PNG."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(9, 5))

    ks  = np.array(k_values)
    mu  = np.array(mean_rms)
    sig = np.array(std_rms)

    # Shaded std band
    ax.fill_between(ks, mu - sig, mu + sig,
                    color="#58a6ff", alpha=0.18, label="±1 std dev")

    # Main line
    ax.plot(ks, mu, color="#58a6ff", linewidth=2.2,
            marker="o", markersize=6, markerfacecolor="#1f6feb",
            markeredgecolor="#58a6ff", label="Mean RMS")

    # Elbow marker
    if best_k in k_values:
        ei = k_values.index(best_k)
        ax.scatter([best_k], [mu[ei]], s=180, color="#f78166",
                   zorder=5, label=f"Elbow  k = {best_k}")
        ax.axvline(best_k, color="#f78166", linewidth=1.2,
                   linestyle="--", alpha=0.7)
        ax.annotate(f" k = {best_k}",
                    xy=(best_k, mu[ei]),
                    xytext=(best_k + 0.3, mu[ei] + (mu.max() - mu.min()) * 0.05),
                    color="#f78166", fontsize=11)

    ax.set_xlabel("Number of Clusters  k", fontsize=12)
    ax.set_ylabel("Average RMS Intra-cluster Distance", fontsize=12)
    ax.set_title("Elbow Method  —  Optimal k Selection", fontsize=14, fontweight="bold")
    ax.set_xticks(k_values)
    ax.legend(framealpha=0.3, edgecolor="#30363d")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    success(f"Elbow diagram saved → {out}")


# ─────────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Galaxy Morphology Analyzer — Terminal CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = p.add_subparsers(dest="command", help="Command to run")

    # scan
    s = sub.add_parser("scan", help="Scan directory for images")
    s.add_argument("directory", nargs="?", default="data/raw",
                   help="Path to image directory")

    # preprocess
    s = sub.add_parser("preprocess", help="Test preprocessing pipeline")
    s.add_argument("--size",      type=int,   default=128, help="Image size (px)")
    s.add_argument("--sigma",     type=float, default=1.0, help="Gaussian blur sigma")
    s.add_argument("--no-normalize", dest="normalize", action="store_false")
    s.set_defaults(normalize=True)
    s.add_argument("--sample", type=int, default=20, help="How many images to test")

    # compress
    s = sub.add_parser("compress", help="Test NCD on image pairs")
    s.add_argument("paths", nargs="*", help="Image files (optional; uses session if omitted)")
    s.add_argument("--algo", default="zlib", choices=list(COMPRESSORS))
    s.add_argument("--n",    type=int, default=8, help="Number of session images to test")

    # distances
    s = sub.add_parser("distances", help="Compute full NCD distance matrix")
    s.add_argument("--n",      type=int,   default=50,        help="Max images")
    s.add_argument("--algo",   default="zlib", choices=list(COMPRESSORS))
    s.add_argument("--output", default=None,                  help="Output .npy file")

    # neighbors
    s = sub.add_parser("neighbors", help="Find nearest neighbors")
    s.add_argument("--query",    type=int, default=None, help="Query image index")
    s.add_argument("--k",        type=int, default=5,    help="Number of neighbors")
    s.add_argument("--n-queries",type=int, default=3,    help="Auto-query N images")

    # cluster
    s = sub.add_parser("cluster", help="Cluster images")
    s.add_argument("--k",       type=int,   default=4)
    s.add_argument("--algo",    default="kmedoids", choices=["kmedoids","hierarchical"])
    s.add_argument("--linkage", default="average",
                   choices=["ward","complete","average","single"])
    s.add_argument("--seed",    type=int,   default=42)
    s.add_argument("--output",  default=None)

    # rms
    s = sub.add_parser("rms", help="RMS vs k to find optimal clusters")
    s.add_argument("--k-min",  type=int, default=2)
    s.add_argument("--k-max",  type=int, default=10)
    s.add_argument("--runs",   type=int, default=3, help="Runs per k value")
    s.add_argument("--output", default=None)

    # pipeline
    s = sub.add_parser("pipeline", help="Full pipeline end-to-end")
    s.add_argument("directory",        help="Image directory")
    s.add_argument("--n",        type=int, default=50,    help="Max images")
    s.add_argument("--k",        type=int, default=4,     help="Clusters")
    s.add_argument("--algo",     default="zlib",          choices=list(COMPRESSORS))
    s.add_argument("--img-size", type=int, default=128,   dest="img_size")

    # info
    sub.add_parser("info", help="Show session state")

    return p


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

COMMANDS = {
    "scan"      : cmd_scan,
    "preprocess": cmd_preprocess,
    "compress"  : cmd_compress,
    "distances" : cmd_distances,
    "neighbors" : cmd_neighbors,
    "cluster"   : cmd_cluster,
    "rms"       : cmd_rms,
    "pipeline"  : cmd_pipeline,
    "info"      : cmd_info,
}


def main():
    banner()
    parser = build_parser()
    args   = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print(f"""
{CYAN}Quick-start examples:{RESET}
  python galaxy_cli.py scan /path/to/images
  python galaxy_cli.py preprocess --size 128
  python galaxy_cli.py distances --n 200 --algo zlib
  python galaxy_cli.py rms --k-min 2 --k-max 12 --runs 3
  python galaxy_cli.py cluster --k 5
  python galaxy_cli.py pipeline /path/to/images --n 500 --k 6

{CYAN}Large-dataset tips:{RESET}
  • Use --algo zlib  (fastest) for exploration; bz2/lzma for final analysis
  • Use --n to limit images during testing
  • Distances are cached in distances.npy; re-run cluster/rms without recomputing
  • Session state is persisted in galaxy_session.pkl between runs
""")
        return

    fn = COMMANDS.get(args.command)
    if fn is None:
        error(f"Unknown command: {args.command}")
        sys.exit(1)

    try:
        t0 = time.time()
        fn(args)
        elapsed = time.time() - t0
        print(f"\n  {DIM}Finished in {elapsed:.2f}s{RESET}\n")
    except KeyboardInterrupt:
        print(f"\n  {YELLOW}Interrupted by user.{RESET}\n")
        sys.exit(130)
    except Exception as e:
        error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()