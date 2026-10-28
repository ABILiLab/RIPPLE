"""Dataset readers for the RNA foundation benchmark.

Each loader returns plain dict rows. Sequence tasks use ``seq`` plus ``label``
or ``target``. Pair tasks use ``seq`` plus ``pairs`` or ``contact``.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

DATA_ROOT = Path("/root/autodl-tmp/RIPPLE/benchmark")
CONTACT_DEV_VAL = 43
SPLICE_SPECIES = ("Danio", "Fly", "Worm", "Thaliana")


def clean_seq(seq: str) -> str:
    return seq.upper().replace("T", "U")


def cap_rows(rows: Sequence[dict], limit: int, key=None) -> List[dict]:
    rows = list(rows)
    if limit <= 0 or len(rows) <= limit:
        return rows
    if key is None:
        return rows[:limit]
    buckets: Dict[object, List[dict]] = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    chosen: List[dict] = []
    while len(chosen) < limit:
        progressed = False
        for label in list(buckets):
            if buckets[label] and len(chosen) < limit:
                chosen.append(buckets[label].pop(0))
                progressed = True
        if not progressed:
            break
    return chosen


def parse_pairs(struct: str) -> List[Tuple[int, int]]:
    stacks: Dict[str, List[int]] = {"(": [], "[": [], "{": [], "<": []}
    close = {")": "(", "]": "[", "}": "{", ">": "<"}
    pairs: List[Tuple[int, int]] = []
    for index, char in enumerate(struct):
        if char in stacks:
            stacks[char].append(index)
        elif char in close and stacks[close[char]]:
            partner = stacks[close[char]].pop()
            left, right = (partner, index) if partner < index else (index, partner)
            pairs.append((left, right))
    return pairs


def pairs_to_matrix(length: int, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((length, length), dtype=np.float32)
    for left, right in pairs:
        if 0 <= left < length and 0 <= right < length and left != right:
            matrix[left, right] = 1.0
            matrix[right, left] = 1.0
    return matrix


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    header = None
    chunks: List[str] = []
    records: List[Tuple[str, str]] = []

    def flush() -> None:
        if header is not None:
            records.append((header, clean_seq("".join(chunks))))

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    flush()
    return records


def archive_family(name: str) -> str:
    return name.split("-", 1)[0]


def read_fold(path: Path, max_len: int) -> Tuple[List[dict], int]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    rows = []
    skipped = 0
    for offset in range(0, len(lines) - 2, 3):
        name = lines[offset][1:] if lines[offset].startswith(">") else lines[offset]
        seq = clean_seq(lines[offset + 1])
        struct = lines[offset + 2].split()[0]
        # Keep IUPAC ambiguity codes. Encoders map them to an unknown token.
        if len(seq) != len(struct) or len(seq) < 8 or any(not base.isalpha() for base in seq):
            skipped += 1
            continue
        if len(seq) > max_len:
            seq = seq[:max_len]
            struct = struct[:max_len]
        rows.append({"name": name, "seq": seq, "pairs": parse_pairs(struct), "family": archive_family(name)})
    return rows, skipped


def load_ncrna(root: Path, limit: int) -> Dict[str, object]:
    """ncrna-deep. Official sizes: 105,864 / 17,324 / 25,342 across 88 families."""
    folder = root / "ncRNAFamilyClassification" / "dataset"
    label_to_id: Dict[str, int] = {}
    train_records = read_fasta(folder / "x_train.fasta")
    for label, _seq in train_records:
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)
    splits: Dict[str, object] = {}
    for split, filename in (("train", "x_train.fasta"), ("val", "x_val.fasta"), ("test", "x_test.fasta")):
        records = train_records if split == "train" else read_fasta(folder / filename)
        rows = []
        for label, seq in records:
            if label not in label_to_id or not (8 <= len(seq) <= 1024):
                continue
            if any(base not in "AUGCN" for base in seq):
                continue
            rows.append({"seq": seq, "label": label_to_id[label], "name": label})
        splits[split] = cap_rows(rows, limit, key=lambda row: row["label"])
    splits["n_classes"] = len(label_to_id)
    splits["label_to_id"] = label_to_id
    return splits


def load_mrl_fasta(root: Path, limit: int) -> List[dict]:
    """GSM4084997, 106,530 sequences. Used by the 5-fold Ridge probe."""
    path = root / "MeanRibosomeLoading" / "dataset" / "GSM4084997_varying_length_25to100.fasta"
    rows = []
    for header, seq in read_fasta(path):
        parts = header.split("_")
        if len(parts) < 4 or not parts[-1].startswith("rl"):
            continue
        if any(base not in "AUGC" for base in seq):
            continue
        rows.append({"name": header, "seq": seq, "target": float(parts[-1][2:])})
    return cap_rows(rows, limit)


def load_mrl_optimus(root: Path, limit: int) -> Dict[str, List[dict]]:
    """RiNALMo MRL split: random train / random7600 / human7600."""
    folder = root / "MeanRibosomeLoading" / "dataset" / "Optimus"
    mapping = {
        "train": "train.tsv",
        "val": "validation_random7600.tsv",
        "test": "test_human7600.tsv",
    }
    splits = {}
    for split, filename in mapping.items():
        rows = []
        with (folder / filename).open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for record in reader:
                seq = clean_seq(record["utr"])
                if any(base not in "AUGC" for base in seq):
                    continue
                rows.append({"name": record.get("index", str(len(rows))), "seq": seq, "target": float(record["rl"])})
        splits[split] = cap_rows(rows, limit)
    return splits


def load_ss(root: Path, dataset: str, max_len: int, limit: int) -> Dict[str, object]:
    """Secondary-structure fold files.

    dataset is one of BpRNA-Spot_0 (TR0/VL0/TS0), RIVAS, ArchiveII1024, BpRNA-New.
    ArchiveII1024 only has a test file; leave-one-family-out splits it later.
    """
    folder = root / "SecondaryStructurePrediction" / "dataset" / dataset
    loaded: Dict[str, object] = {}
    skipped = {}
    for split, filename in (("train", "train.fold"), ("val", "validation.fold"), ("test", "test.fold")):
        path = folder / filename
        if not path.is_file():
            continue
        rows, n_skipped = read_fold(path, max_len)
        loaded[split] = cap_rows(rows, limit, key=lambda row: row["family"] if dataset.startswith("Archive") else None)
        skipped[split] = n_skipped
    loaded["skipped"] = skipped
    return loaded


def _read_splice(path: Path) -> List[dict]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for record in reader:
            seq = clean_seq(record["sequences"])
            if len(seq) < 8 or any(base not in "AUGCN" for base in seq):
                continue
            rows.append({"seq": seq, "label": int(record["labels"])})
    return rows


def load_splice(root: Path, limit: int) -> Dict[str, Dict[str, List[dict]]]:
    """Spliceator GS_1 donor and acceptor, plus four external species."""
    folder = root / "Multi-SpeciesSpliceSite" / "GenBio_AI_Splice_Site"
    output = {}
    for kind in ("donor", "acceptor"):
        splits: Dict[str, List[dict]] = {}
        for split, filename in (("train", "train.tsv"), ("val", "valid.tsv")):
            rows = _read_splice(folder / kind / filename)
            splits[split] = cap_rows(rows, limit, key=lambda row: row["label"])
        for name in SPLICE_SPECIES:
            rows = _read_splice(folder / kind / f"test_{name}.tsv")
            splits[name] = cap_rows(rows, limit, key=lambda row: row["label"])
        output[kind] = splits
    return output


def pdb_residue_coords(path: Path) -> List[np.ndarray]:
    best: Dict[Tuple[str, str, str], Tuple[float, int, np.ndarray]] = {}
    order: Dict[Tuple[str, str], int] = {}
    next_order = 0
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            atom = line[12:16]
            residue_key = (line[22:26], line[26])
            try:
                occ = float(line[54:60]) if line[54:60].strip() else 1.0
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
            except ValueError:
                continue
            if residue_key not in order:
                order[residue_key] = next_order
                next_order += 1
            key = (residue_key[0], residue_key[1], atom)
            previous = best.get(key)
            if previous is None or occ > previous[0]:
                best[key] = (occ, order[residue_key], xyz)
    grouped: List[List[np.ndarray]] = [[] for _ in range(next_order)]
    for _occ, residue_index, xyz in best.values():
        grouped[residue_index].append(xyz)
    return [np.stack(items, axis=0) for items in grouped if items]


def contact_matrix(coords: Sequence[np.ndarray], cutoff: float = 8.0) -> np.ndarray:
    """Heavy-atom minimum distance <= cutoff. Symmetric, zero diagonal."""
    length = len(coords)
    contact = np.zeros((length, length), dtype=np.uint8)
    for i in range(length):
        left = coords[i]
        for j in range(i + 1, length):
            delta = left[:, None, :] - coords[j][None, :, :]
            if float(np.sqrt((delta * delta).sum(-1).min())) <= cutoff:
                contact[i, j] = contact[j, i] = 1
    return contact


def load_contact(root: Path, max_len: int, limit: int, seed: int) -> Dict[str, object]:
    """RNAContact: 211 development structures split 168/43, plus 80 test.

    The 43 validation structures are a seed-42 shuffle of the development set.
    Contacts are cached next to the tertiary-structure files.
    """
    base = root / "TertiaryStructurePrediction"
    cache = base / "contact8A_cache.npz"
    meta_path = base / "contact8A_meta.json"
    if cache.is_file() and meta_path.is_file() and limit <= 0:
        blob = json.loads(meta_path.read_text())
        matrices = np.load(cache)
        rows = []
        skipped = list(blob.get("skipped", []))
        for item in blob["rows"]:
            row = dict(item)
            # Older caches labeled the development set as "train".
            stored_split = row["split"]
            if stored_split == "train":
                row["split"] = "dev"
            key = f"{stored_split}::{row['name']}"
            contact = matrices[key]
            seq = clean_seq(row["seq"])
            if not (8 <= len(seq) <= max_len) or contact.shape[0] != len(seq):
                skipped.append({"id": row["name"], "split": row["split"], "fasta": len(seq), "pdb": int(contact.shape[0]), "reason": "cache_filter"})
                continue
            row["seq"] = seq
            row["contact"] = contact
            rows.append(row)
    else:
        rows = []
        skipped = []
        jobs = (
            ("dev", base / "train_set" / "train_fasta", base / "train_set" / "train_pdb"),
            ("test", base / "test_set" / "test_fasta", base / "test_set" / "test_pdb"),
        )
        wanted = {"dev": limit, "test": limit}
        seen = {"dev": 0, "test": 0}
        for split, fasta_dir, pdb_dir in jobs:
            for fasta in sorted(fasta_dir.glob("*.fasta")):
                if limit > 0 and seen[split] >= wanted[split]:
                    break
                seq = clean_seq(
                    "".join(
                        line.strip()
                        for line in fasta.read_text().splitlines()
                        if line.strip() and not line.startswith(">")
                    )
                )
                coords = pdb_residue_coords(pdb_dir / f"{fasta.stem}.pdb")
                if len(coords) != len(seq) or not (8 <= len(seq) <= max_len) or any(base not in "AUGCN" for base in seq):
                    skipped.append({"id": fasta.stem, "split": split, "fasta": len(seq), "pdb": len(coords)})
                    continue
                rows.append({"name": fasta.stem, "split": split, "seq": seq, "contact": contact_matrix(coords)})
                seen[split] += 1
                print(f"[contact] {split} {seen[split]} {fasta.stem} L={len(seq)}", flush=True)
        if limit <= 0:
            np.savez_compressed(cache, **{f"{row['split']}::{row['name']}": row["contact"] for row in rows})
            meta = [{"name": row["name"], "split": row["split"], "seq": row["seq"]} for row in rows]
            meta_path.write_text(json.dumps({"rows": meta, "skipped": skipped}))
    dev = [row for row in rows if row["split"] == "dev"]
    test = [row for row in rows if row["split"] == "test"]
    val_n = CONTACT_DEV_VAL if len(dev) == 211 else max(1, int(round(0.2 * len(dev)))) if len(dev) > 1 else 0
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dev))
    val_index = set(order[:val_n].tolist())
    val = [dev[i] for i in range(len(dev)) if i in val_index]
    train = [dev[i] for i in range(len(dev)) if i not in val_index]
    return {"train": train, "val": val, "test": test, "skipped": skipped, "val_n": val_n}
