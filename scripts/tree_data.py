#!/usr/bin/env -S uvx --with numpy,pandas,joblib,rich python
"""
tree_data_shapes_rich.py

Rich-based tree listing with data-aware summaries.

Features
--------
- Summarize many .npy files with the same array shape in a directory:
    tag: [Σ NPY]
- Summarize many joblib files that are DataFrames with same shape+columns:
    tag: [Σ DF]
- For .csv files:
    tag: [CSV] + shape + columns (DataFrame-style if pandas available)
- For .joblib files that contain a DataFrame:
    tag: [DF]  + shape + columns (with ndarray column shapes)
- For single .npy files:
    tag: [NPY] + array shape
- For DataFrame columns containing numpy.ndarray values, show shapes:
    e.g. data[4x5]
- Group sibling directories that are "pure joblib-DF dirs" with same
  shape + base columns (ignoring ndarray shapes):
    tag: [G DF]
- Group sibling directories that are "pure .npy dirs" with same
  number of files + array shape:
    tag: [G NPY]

Usage
-----
  ./tree_data_shapes_rich.py PATH [--max-depth N] [--follow-symlinks] [--no-progress]
"""

from __future__ import annotations

import argparse
import csv
import os
import stat
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

# -----------------------------------------------------------------------------
# Caches for metadata (to avoid re-reading the same files repeatedly)
# -----------------------------------------------------------------------------
_NPY_CACHE: Dict[str, Optional[Tuple[int, ...]]] = {}
_CSV_CACHE: Dict[str, Optional[Tuple[Tuple[int, int], Tuple[str, ...]]]] = {}
_JOBLIB_DF_CACHE: Dict[str, Optional[Tuple[Tuple[int, int], Tuple[str, ...]]]] = {}


# =============================================================================
# Optional deps
# =============================================================================
def try_import_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except Exception:
        return None


def try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None


def try_import_joblib():
    try:
        import joblib  # type: ignore
        return joblib
    except Exception:
        return None


# =============================================================================
# Filesystem helpers
# =============================================================================
def is_dir(entry: os.DirEntry, follow_symlinks: bool) -> bool:
    try:
        return entry.is_dir(follow_symlinks=follow_symlinks)
    except OSError:
        return False


def is_file(entry: os.DirEntry, follow_symlinks: bool) -> bool:
    try:
        return entry.is_file(follow_symlinks=follow_symlinks)
    except OSError:
        return False


def safe_stat(path: str) -> Optional[os.stat_result]:
    try:
        return os.lstat(path)
    except OSError:
        return None


def scan_sorted(path: str, follow_symlinks: bool) -> List[os.DirEntry]:
    """Return directory entries sorted with dirs first, then files by name."""
    try:
        with os.scandir(path) as it:
            entries = [e for e in it]
    except (PermissionError, FileNotFoundError):
        return []

    def sort_key(e: os.DirEntry):
        try:
            dfirst = 0 if e.is_dir(follow_symlinks=follow_symlinks) else 1
        except OSError:
            dfirst = 1
        return (dfirst, e.name.lower())

    return sorted(entries, key=sort_key)


# =============================================================================
# Data extractors and helpers (cached)
# =============================================================================
def shape_to_str(shape: Tuple[int, ...]) -> str:
    return "x".join(str(d) for d in shape) if shape else "()"


def array_columns_info(df) -> List[str]:
    """
    Build column labels, appending ndarray shapes as e.g. col[4x5].
    Only first row is inspected to decide ndarray-ness.
    """
    np = try_import_numpy()
    info: List[str] = []
    for col in df.columns:
        val = df[col].iloc[0] if len(df[col]) > 0 else None
        if np is not None and isinstance(val, np.ndarray):
            info.append(f"{col}[{shape_to_str(val.shape)}]")
        else:
            info.append(str(col))
    return info


def npy_shape(path: str) -> Optional[Tuple[int, ...]]:
    if path in _NPY_CACHE:
        return _NPY_CACHE[path]
    np = try_import_numpy()
    if np is None:
        _NPY_CACHE[path] = None
        return None
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = tuple(arr.shape)
        _NPY_CACHE[path] = shape
        return shape
    except Exception:
        _NPY_CACHE[path] = None
        return None


def csv_info(path: str) -> Optional[Tuple[Tuple[int, int], Tuple[str, ...]]]:
    if path in _CSV_CACHE:
        return _CSV_CACHE[path]
    pd = try_import_pandas()
    if pd is not None:
        try:
            df = pd.read_csv(path)
            cols = array_columns_info(df)
            result = (tuple(df.shape), tuple(cols))
            _CSV_CACHE[path] = result
            return result
        except Exception:
            _CSV_CACHE[path] = None
            return None
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            n_rows = sum(1 for _ in reader)
            result = ((n_rows, len(header)), tuple(header))
            _CSV_CACHE[path] = result
            return result
    except Exception:
        _CSV_CACHE[path] = None
        return None


def joblib_df_info(path: str) -> Optional[Tuple[Tuple[int, int], Tuple[str, ...]]]:
    if path in _JOBLIB_DF_CACHE:
        return _JOBLIB_DF_CACHE[path]
    joblib = try_import_joblib()
    pd = try_import_pandas()
    if joblib is None or pd is None:
        _JOBLIB_DF_CACHE[path] = None
        return None
    try:
        obj = joblib.load(path)
        if isinstance(obj, pd.DataFrame):
            cols = array_columns_info(obj)
            result = (tuple(obj.shape), tuple(cols))
            _JOBLIB_DF_CACHE[path] = result
            return result
        _JOBLIB_DF_CACHE[path] = None
        return None
    except Exception:
        _JOBLIB_DF_CACHE[path] = None
        return None


def columns_to_str(cols: Sequence[str]) -> str:
    return ", ".join(cols)


def base_col_name(label: str) -> str:
    """
    Strip any ndarray-shape suffix, e.g. "data[4x5]" -> "data".
    """
    i = label.find("[")
    return label[:i] if i != -1 else label


# =============================================================================
# Simple tag helpers for Rich Text
# =============================================================================
def make_tag(label: str, color: str) -> Text:
    t = Text()
    t.append("[", style="dim")
    t.append(label, style=f"{color} bold")
    t.append("]", style="dim")
    return t


# =============================================================================
# Summaries for files in a directory
# =============================================================================
def summarize_npy_entries(
    entries: List[os.DirEntry],
    follow_symlinks: bool,
) -> Tuple[List[os.DirEntry], List[Dict[str, Any]]]:
    """Return entries without summarized npy files and npy summary payloads."""
    npy_files = [
        e
        for e in entries
        if is_file(e, follow_symlinks) and e.name.lower().endswith(".npy")
    ]
    if len(npy_files) < 2:
        return entries, []

    shapes = []
    for e in npy_files:
        s = npy_shape(e.path)
        if s is None:
            return entries, []  # abort on unreadable
        shapes.append(s)

    if len(set(shapes)) != 1:
        return entries, []

    shape = shapes[0]
    remaining = [e for e in entries if e not in npy_files]
    summary = {
        "type": "npy",
        "count": len(npy_files),
        "shape": shape,
    }
    return remaining, [summary]


def summarize_joblib_df_entries(
    entries: List[os.DirEntry],
    follow_symlinks: bool,
) -> Tuple[List[os.DirEntry], List[Dict[str, Any]]]:
    """Return entries without summarized joblib-DF files and DF summary payloads."""
    jl_files = [
        e
        for e in entries
        if is_file(e, follow_symlinks) and e.name.lower().endswith(".joblib")
    ]
    if len(jl_files) < 2:
        return entries, []

    metas: List[Tuple[Tuple[int, int], Tuple[str, ...]]] = []
    for e in jl_files:
        info = joblib_df_info(e.path)
        if info is None:
            return entries, []
        metas.append(info)

    if len(set(metas)) != 1:
        return entries, []

    shape, cols = metas[0]
    remaining = [e for e in entries if e not in jl_files]
    summary = {
        "type": "df",
        "count": len(jl_files),
        "shape": shape,
        "columns": cols,
    }
    return remaining, [summary]


# =============================================================================
# NPY directory grouping
# =============================================================================
def inspect_npy_dir(
    path: str, follow_symlinks: bool
) -> Optional[Tuple[int, Tuple[int, ...]]]:
    """
    Inspect a directory to see if it is a "pure npy dir":
    - Contains only .npy files (>=1).
    - No subdirectories.
    - All .npy files share the same array shape.

    Returns (num_files, shape) or None.
    """
    children = scan_sorted(path, follow_symlinks)
    if not children:
        return None

    files = [e for e in children if is_file(e, follow_symlinks)]
    if not files:
        return None

    npy_files = [e for e in files if e.name.lower().endswith(".npy")]
    if not npy_files:
        return None

    # Only npy files and no subdirs
    if len(npy_files) != len(files):
        return None
    if any(is_dir(e, follow_symlinks) for e in children):
        return None

    shapes: List[Tuple[int, ...]] = []
    for e in npy_files:
        s = npy_shape(e.path)
        if s is None:
            return None
        shapes.append(s)

    if len(set(shapes)) != 1:
        return None

    return len(npy_files), shapes[0]


def build_npy_dir_groups(
    dirs: List[os.DirEntry], follow_symlinks: bool
):
    """
    Group sibling directories that are "pure npy dirs" with the same:
      - number of .npy files
      - array shape
    """
    groups_by_key: Dict[Tuple, Dict[str, Any]] = {}
    for d in dirs:
        sig = inspect_npy_dir(d.path, follow_symlinks)
        if sig is None:
            continue
        count, shape = sig
        key = ("npy_dir", count, shape)
        if key not in groups_by_key:
            groups_by_key[key] = {"key": key, "dirs": [], "meta": (count, shape)}
        groups_by_key[key]["dirs"].append(d)

    groups: List[Dict[str, Any]] = []
    dir_to_group: Dict[str, Dict[str, Any]] = {}
    gid = 0
    for key, g in groups_by_key.items():
        if len(g["dirs"]) < 2:
            continue
        gid += 1
        g["id"] = gid
        groups.append(g)
        for d in g["dirs"]:
            dir_to_group[d.path] = g

    return groups, dir_to_group


# =============================================================================
# Joblib-DataFrame directory grouping
# =============================================================================
def inspect_joblib_df_dir(
    path: str, follow_symlinks: bool
) -> Optional[Tuple[int, Tuple[int, int], Tuple[str, ...]]]:
    """
    Inspect a directory to see if it is a "pure joblib-DF dir":
    - Contains only joblib files (>=1).
    - No subdirectories.
    - All joblib files contain DataFrames with identical (shape, columns).

    Returns (num_files, shape, cols-with-array-shapes) or None.
    """
    children = scan_sorted(path, follow_symlinks)
    if not children:
        return None

    files = [e for e in children if is_file(e, follow_symlinks)]
    if not files:
        return None

    jl_files = [e for e in files if e.name.lower().endswith(".joblib")]
    if not jl_files:
        return None

    # Require only joblib files and no subdirectories for a "pure" data dir
    if len(jl_files) != len(files):
        return None
    if any(is_dir(e, follow_symlinks) for e in children):
        return None

    metas: List[Tuple[Tuple[int, int], Tuple[str, ...]]] = []
    for e in jl_files:
        info = joblib_df_info(e.path)
        if info is None:
            return None
        metas.append(info)

    if len(set(metas)) != 1:
        return None

    shape, cols = metas[0]
    return len(jl_files), shape, cols


def build_joblib_dir_groups(
    dirs: List[os.DirEntry], follow_symlinks: bool
):
    """
    Group sibling directories that are "pure joblib-DF dirs" with the same:
      - number of joblib files
      - DataFrame shape
      - base column names (ignoring ndarray shapes)
    """
    groups_by_key: Dict[Tuple, Dict[str, Any]] = {}
    for d in dirs:
        sig = inspect_joblib_df_dir(d.path, follow_symlinks)
        if sig is None:
            continue
        count, shape, cols = sig
        base_cols = tuple(base_col_name(c) for c in cols)
        key = ("joblib_df_dir", count, shape, base_cols)
        if key not in groups_by_key:
            groups_by_key[key] = {"key": key, "dirs": [], "metas": []}
        groups_by_key[key]["dirs"].append(d)
        groups_by_key[key]["metas"].append((shape, cols))

    groups: List[Dict[str, Any]] = []
    dir_to_group: Dict[str, Dict[str, Any]] = {}
    gid = 0
    for key, g in groups_by_key.items():
        if len(g["dirs"]) < 2:
            continue
        gid += 1
        shape, cols0 = g["metas"][0]
        base_cols = tuple(base_col_name(c) for c in cols0)
        g["id"] = gid
        g["count_per_dir"] = key[1]
        g["shape"] = shape
        g["base_cols"] = base_cols
        groups.append(g)
        for d in g["dirs"]:
            dir_to_group[d.path] = g

    return groups, dir_to_group


# =============================================================================
# Tree rendering with Rich
# =============================================================================
def render_tree(
    console: Console, root: str, max_depth: Optional[int], follow_symlinks: bool
) -> None:
    root_label = Text(root.rstrip(os.sep) + " ", style="bold white")
    root_label.append("(root)", style="dim")
    tree = Tree(root_label)

    total_dirs = 0
    total_files = 0

    def _walk(path: str, depth: int, node: Tree) -> None:
        nonlocal total_dirs, total_files

        entries = scan_sorted(path, follow_symlinks)

        # Summaries within this directory
        entries, npy_summaries = summarize_npy_entries(entries, follow_symlinks)
        entries, df_summaries = summarize_joblib_df_entries(entries, follow_symlinks)
        summaries = npy_summaries + df_summaries

        # Partition dirs/files
        dirs = [e for e in entries if is_dir(e, follow_symlinks)]
        files = [e for e in entries if is_file(e, follow_symlinks)]

        # Group pure joblib-DF directories and pure npy directories
        df_groups, df_dir_to_group = build_joblib_dir_groups(dirs, follow_symlinks)
        npy_groups, npy_dir_to_group = build_npy_dir_groups(dirs, follow_symlinks)
        df_group_ids_seen: set = set()
        npy_group_ids_seen: set = set()

        # 1) Directories (raw + grouped)
        for d in dirs:
            df_g = df_dir_to_group.get(d.path)
            npy_g = npy_dir_to_group.get(d.path)

            # Prefer DF grouping if somehow both would match (should not happen in practice)
            if df_g is not None:
                if df_g["id"] in df_group_ids_seen:
                    continue
                df_group_ids_seen.add(df_g["id"])

                tag = make_tag("G DF", "yellow")
                head = Text()
                head.append_text(tag)
                shape = df_g["shape"]
                count_per_dir = df_g["count_per_dir"]
                base_cols = df_g["base_cols"]
                head.append(
                    f" {len(df_g['dirs'])} dirs; {count_per_dir} joblib DFs each; ",
                    style="yellow",
                )
                head.append(f"frame {shape_to_str(shape)}; ", style="yellow")
                head.append("base columns: ", style="yellow")
                head.append(columns_to_str(base_cols), style="yellow")
                group_node = node.add(head)
                total_dirs += len(df_g["dirs"])

                metas = df_g["metas"]
                _, cols0 = metas[0]
                num_cols = len(cols0)
                varying_positions = set()
                for pos in range(num_cols):
                    labels = {cols[pos] for (_, cols) in metas}
                    if len(labels) > 1:
                        varying_positions.add(pos)

                max_name_len = max(len(dentry.name) for dentry in df_g["dirs"])

                for (shape_j, cols_j), dentry in zip(metas, df_g["dirs"]):
                    name_str = dentry.name + "/"
                    padding = " " * (max_name_len + 1 - len(dentry.name))
                    label = Text(name_str, style="bold cyan")
                    label.append(padding)
                    diffs = [cols_j[pos] for pos in varying_positions]
                    if diffs:
                        label.append("varies: ", style="dim")
                        label.append(", ".join(diffs), style="magenta")
                    else:
                        label.append("(same as group)", style="dim")
                    group_node.add(label)
                continue

            if npy_g is not None:
                if npy_g["id"] in npy_group_ids_seen:
                    continue
                npy_group_ids_seen.add(npy_g["id"])

                tag = make_tag("G NPY", "cyan")
                head = Text()
                head.append_text(tag)
                count_per_dir, shape = npy_g["meta"]
                head.append(
                    f" {len(npy_g['dirs'])} dirs; {count_per_dir} *.npy each; ",
                    style="cyan",
                )
                head.append(f"array shape {shape_to_str(shape)}", style="cyan")
                group_node = node.add(head)
                total_dirs += len(npy_g["dirs"])

                max_name_len = max(len(dentry.name) for dentry in npy_g["dirs"])
                for dentry in npy_g["dirs"]:
                    name_str = dentry.name + "/"
                    padding = " " * (max_name_len + 1 - len(dentry.name))
                    label = Text(name_str, style="bold cyan")
                    label.append(padding)
                    label.append("(same as group)", style="dim")
                    group_node.add(label)
                continue

            # Normal directory
            total_dirs += 1
            d_label = Text(d.name + "/", style="bold cyan")
            child_node = node.add(d_label)
            if max_depth is None or depth < max_depth:
                _walk(d.path, depth + 1, child_node)

        # 2) Summaries (synthetic lines)
        for s in summaries:
            if s["type"] == "npy":
                tag = make_tag("Σ NPY", "cyan")
                t = Text()
                t.append_text(tag)
                t.append(
                    f" {s['count']} *.npy files; array shape {shape_to_str(s['shape'])}",
                    style="cyan",
                )
                node.add(t)
            elif s["type"] == "df":
                tag = make_tag("Σ DF", "green")
                t = Text()
                t.append_text(tag)
                t.append(
                    f" {s['count']} joblib DFs; frame {shape_to_str(s['shape'])}; ",
                    style="green",
                )
                t.append("columns: ", style="dim")
                t.append(columns_to_str(s["columns"]), style="green")
                node.add(t)

        # 3) Files
        for f in files:
            total_files += 1
            name_lower = f.name.lower()
            label = Text(f.name, style="white")

            if name_lower.endswith(".csv"):
                info = csv_info(f.path)
                if info:
                    shape, cols = info
                    tag = make_tag("CSV", "blue")
                    detail = Text()
                    detail.append_text(tag)
                    detail.append(
                        f" shape={shape}; columns: ", style="blue",
                    )
                    detail.append(columns_to_str(cols), style="blue")
                    label.append("  ")
                    label.append_text(detail)

            elif name_lower.endswith(".joblib"):
                info = joblib_df_info(f.path)
                if info:
                    shape, cols = info
                    tag = make_tag("DF", "green")
                    detail = Text()
                    detail.append_text(tag)
                    detail.append(
                        f" shape={shape}; columns: ", style="green",
                    )
                    detail.append(columns_to_str(cols), style="green")
                    label.append("  ")
                    label.append_text(detail)

            elif name_lower.endswith(".npy"):
                shape = npy_shape(f.path)
                if shape is not None:
                    tag = make_tag("NPY", "cyan")
                    detail = Text()
                    detail.append_text(tag)
                    detail.append(f" shape={shape}", style="cyan")
                    label.append("  ")
                    label.append_text(detail)

            node.add(label)

    _walk(root, 1, tree)

    console.print(tree)
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                "[bold]Legend[/bold]\n"
                "  [cyan][Σ NPY][/cyan] summary of many .npy files (same array shape)\n"
                "  [green][Σ DF][/green] summary of many joblib DataFrames (same shape/columns)\n"
                "  [cyan][NPY][/cyan] single .npy file (array shape)\n"
                "  [blue][CSV][/blue] CSV file (shape + columns)\n"
                "  [green][DF][/green] joblib file containing a DataFrame (shape + columns)\n"
                "  [yellow][G DF][/yellow] group of similar DataFrame directories; child dirs "
                "show only differing columns as 'varies: ...'\n"
                "  [cyan][G NPY][/cyan] group of similar .npy-only directories (same count/shape)\n"
                "  Column names like 'data[4x5]' mean that column stores numpy arrays of shape 4x5."
            ),
            title="Data-aware tree",
            border_style="magenta",
        )
    )
    console.print(f"[bold]{total_dirs}[/bold] directories, [bold]{total_files}[/bold] files")


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Rich tree-like listing with .npy/.csv/.joblib awareness, ndarray column shapes, "
            "and collapsed joblib-DF/.npy directories"
        )
    )
    ap.add_argument("path", help="root path")
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--follow-symlinks", action="store_true")
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="disable Rich spinner while scanning",
    )
    args = ap.parse_args()

    root = args.path
    st = safe_stat(root)
    if st is None:
        print(f"error: cannot access '{root}'", file=sys.stderr)
        sys.exit(1)
    if not stat.S_ISDIR(st.st_mode):
        print(root)
        print("└── (not a directory)")
        print("\n0 directories, 1 files")
        return

    console = Console()

    if args.no_progress:
        render_tree(console, root, args.max_depth, args.follow_symlinks)
    else:
        with console.status("Scanning directories and data files...", spinner="dots"):
            render_tree(console, root, args.max_depth, args.follow_symlinks)


if __name__ == "__main__":
    main()
