"""Create a FASTA file with one entry per UGT_ID from a CSV file.

Reads a CSV, extracts columns for UGT_ID and prot_seq (configurable), and
writes one FASTA entry per UGT_ID. If two different UGT_IDs have identical
protein sequences, both IDs will be written separately (no merging).

Usage:
    python scripts/create_unique_fasta.py -i input.csv -o output.fasta

Defaults:
    id column: UGT_ID
    sequence column: prot_seq

This script attempts to use pandas if available, otherwise falls back to the
standard csv module. It gracefully handles missing values and prints a short
summary.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import re
import sys
from typing import Dict, Iterable, List, Set, Tuple

try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except Exception:  # pragma: no cover - optional dependency
    import csv

    _HAS_PANDAS = False


LOG_FORMAT = "%(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


AA_RE = re.compile(r"^[A-Z]+$")


def read_csv_rows(path: str, id_col: str, seq_col: str) -> Iterable[Tuple[str, str]]:
    """Yield (id, sequence) tuples from CSV file.

    Supports gzip (.gz) CSVs as well.
    """
    opener = gzip.open if path.endswith(".gz") else open

    if _HAS_PANDAS:
        logger.debug("Reading CSV with pandas: %s", path)
        df = pd.read_csv(path, dtype=str)
        for idx, row in df.iterrows():
            yield (str(row.get(id_col, "")).strip(), str(row.get(seq_col, "")).strip())
    else:
        logger.debug("Reading CSV with csv.DictReader: %s", path)
        with opener(path, "rt", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield (str(row.get(id_col, "")).strip(), str(row.get(seq_col, "")).strip())


def normalize_sequence(seq: str) -> str:
    """Normalize sequence: remove whitespace and non-letter characters, uppercase."""
    if seq is None:
        return ""
    seq = re.sub(r"\s+", "", seq).upper()
    # remove unusual characters like '-' or numbers
    seq = re.sub(r"[^A-Z]", "", seq)
    return seq


def write_fasta(mapping: Dict[str, str], out_path: str, line_width: int = 60) -> None:
    """Write FASTA with one record per UGT_ID (mapping: id -> sequence)."""
    with open(out_path, "w", encoding="utf-8") as out:
        for uid, seq in mapping.items():
            out.write(f">{uid}\n")
            for i in range(0, len(seq), line_width):
                out.write(seq[i : i + line_width] + "\n")


def collect_unique_sequences(path: str, id_col: str, seq_col: str) -> Tuple[Dict[str, str], int, int]:
    """Return mapping id -> sequence, total_rows, skipped_rows.

    Each UGT_ID is returned once. If a UGT_ID appears multiple times with
    different sequences, the first occurrence is kept and a warning is emitted.
    """
    id_to_seq: Dict[str, str] = {}
    total = 0
    skipped = 0

    for uid, raw_seq in read_csv_rows(path, id_col, seq_col):
        total += 1
        if not uid or not raw_seq:
            skipped += 1
            logger.debug("Skipping row with missing id or sequence (id=%r)", uid)
            continue
        seq = normalize_sequence(raw_seq)
        if not seq:
            skipped += 1
            logger.debug("Skipping row because sequence empty after normalization (id=%r)", uid)
            continue
        if not AA_RE.match(seq):
            # still accept but warn
            logger.warning("Sequence for id %s contains unexpected characters; cleaned to %r", uid, seq)
        if uid in id_to_seq:
            if id_to_seq[uid] != seq:
                logger.warning("UGT_ID %s appears multiple times with different sequences; keeping the first occurrence", uid)
            continue
        id_to_seq[uid] = seq

    return id_to_seq, total, skipped


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a FASTA of unique protein sequences from a CSV")
    p.add_argument("-i", "--input", required=True, help="Path to input CSV file")
    p.add_argument("-o", "--output", help="Path to output FASTA. Defaults to <input>.unique.fasta")
    p.add_argument("--id-col", default="UGT_ID", help="Column name for the sequence identifier (default: UGT_ID)")
    p.add_argument("--seq-col", default="prot_seq", help="Column name for protein sequence (default: prot_seq)")
    p.add_argument("--min-len", type=int, default=0, help="Minimum sequence length to keep (after cleaning)")
    p.add_argument("--quiet", action="store_true", help="Reduce logging output")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quiet:
        logger.setLevel(logging.WARNING)

    in_path = args.input
    out_path = args.output or os.path.splitext(in_path)[0] + ".unique.fasta"

    logger.info("Reading %s", in_path)
    mapping, total, skipped = collect_unique_sequences(in_path, args.id_col, args.seq_col)

    # apply length filter
    if args.min_len > 0:
        before = len(mapping)
        mapping = {s: ids for s, ids in mapping.items() if len(s) >= args.min_len}
        after = len(mapping)
        logger.info("Filtered sequences shorter than %d: %d -> %d", args.min_len, before, after)

    if not mapping:
        logger.warning("No sequences to write after processing (total rows=%d, skipped=%d)", total, skipped)
        return 1

    write_fasta(mapping, out_path)

    logger.info("Wrote %d unique sequences to %s (total rows=%d, skipped=%d)", len(mapping), out_path, total, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
