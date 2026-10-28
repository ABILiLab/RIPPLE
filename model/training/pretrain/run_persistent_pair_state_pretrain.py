#!/usr/bin/env python3
"""Full-scale sequence-only MLM + persistent pair-state pretraining.

The training objective has three components:

* token/span masked-language modelling on the large RNAcentral corpus;
* a 16-way self-supervised joint endpoint objective on every sequence;
* gain-distilled candidate retrieval without structure labels.

At scheduled pretraining validations, the same in-memory frozen encoder is
evaluated on small, fixed monitors for repository understanding tasks. Tertiary
structure is deliberately excluded. Monitor test scores are diagnostic only and
never select checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

# Some downstream metadata CSVs contain serialized fields above Python's
# conservative 128 KiB default. Raising only the parser limit preserves the
# leakage-audit predicate and prevents a pre-training false failure.
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.core.models import RNAModelConfig
from model.core.persistent_pair_state_mlm import (
    PersistentPairStateEncoderAdapter,
    PersistentPairStateForMaskedLM,
    PersistentPairStateFoundationView,
)
from model.core.pair_composite import (
    fixed_marginal_joint_log_probs,
    independent_joint_log_probs,
    nll_from_joint_logp,
)
from model.core.tokenizer import CLS_ID, MASK_ID, PAD_ID, SEP_ID, RNATokenizer


BRACKETS = {"(": ")", "[": "]", "{": "}", "<": ">"}
BASE_TO_ID = {"A": 4, "U": 5, "C": 6, "G": 7}
BASE_ORDER = ("A", "U", "C", "G")


@dataclass(frozen=True)
class FoldRecord:
    name: str
    sequence: str
    structure: str


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def fold_offset_index_path(path: str, max_bases: int) -> str:
    return f"{path}.offsets.mb{int(max_bases)}.u64"


def build_fold_offset_index(path: str, max_bases: int, out_path: str) -> int:
    """Byte offsets of records accepted by the same filters as iter_fold."""
    temporary = out_path + ".tmp"
    count = 0
    with open(path, "rb") as handle, open(temporary, "wb") as output:
        while True:
            position = handle.tell()
            name = handle.readline()
            if not name:
                break
            if not name.startswith(b">"):
                continue
            sequence_bytes = handle.readline()
            structure_bytes = handle.readline()
            if not sequence_bytes or not structure_bytes:
                break
            sequence = sequence_bytes.decode("utf-8").strip().upper().replace("T", "U")
            structure = structure_bytes.decode("utf-8").strip().split()[0]
            if len(sequence) != len(structure) or not 8 <= len(sequence) <= max_bases:
                continue
            if any(base not in "AUCGN" for base in sequence):
                continue
            output.write(int(position).to_bytes(8, "little"))
            count += 1
    if count <= 0:
        raise RuntimeError(f"no indexable fold records in {path}")
    os.replace(temporary, out_path)
    return count


def read_fold_at(handle, offset: int, max_bases: int) -> Optional[FoldRecord]:
    handle.seek(int(offset))
    name = handle.readline()
    if not name.startswith(b">"):
        return None
    sequence_bytes = handle.readline()
    structure_bytes = handle.readline()
    if not sequence_bytes or not structure_bytes:
        return None
    sequence = sequence_bytes.decode("utf-8").strip().upper().replace("T", "U")
    structure = structure_bytes.decode("utf-8").strip().split()[0]
    if len(sequence) != len(structure) or not 8 <= len(sequence) <= max_bases:
        return None
    if any(base not in "AUCGN" for base in sequence):
        return None
    return FoldRecord(name.decode("utf-8").strip(), sequence, structure)


def iter_fold(path: str, max_bases: int) -> Iterator[FoldRecord]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as handle:
        while True:
            name = handle.readline()
            if not name:
                break
            if not name.startswith(">"):
                continue
            sequence = handle.readline().strip().upper().replace("T", "U")
            structure_line = handle.readline().strip()
            if not sequence or not structure_line:
                continue
            structure = structure_line.split()[0]
            if len(sequence) != len(structure) or not 8 <= len(sequence) <= max_bases:
                continue
            if any(base not in "AUCGN" for base in sequence):
                continue
            yield FoldRecord(name.strip(), sequence, structure)


def normalize_identifier(value: str) -> str:
    """Conservative cross-file identifier normalization for leakage checks."""
    token = value.strip().lstrip(">").split()[0].upper()
    return token.rsplit(".", 1)[0] if token.rsplit(".", 1)[-1].isdigit() else token


def normalize_sequence(value: str) -> str:
    return "".join(value.upper().replace("T", "U").split())


@dataclass
class LeakageExclusions:
    identifiers: Set[str]
    sequences: Set[str]
    source_counts: Dict[str, int]

    def matches(self, record: FoldRecord) -> Tuple[bool, bool]:
        return (
            normalize_identifier(record.name) in self.identifiers,
            normalize_sequence(record.sequence) in self.sequences,
        )


def should_exclude_record(record: FoldRecord, exclusions: LeakageExclusions) -> bool:
    by_id, by_sequence = exclusions.matches(record)
    return by_id or by_sequence


def _read_fasta_records(path: str) -> Iterator[Tuple[str, str]]:
    name = ""
    chunks: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if chunks:
                    yield name, normalize_sequence("".join(chunks))
                name, chunks = line, []
            elif line:
                chunks.append(line)
    if chunks:
        yield name, normalize_sequence("".join(chunks))


def build_leakage_exclusions(
    sequence_valid_path: str,
    bprna_monitor_paths: Sequence[str],
    downstream_data_root: str,
    max_bases: int,
) -> LeakageExclusions:
    """Collect every fixed validation/monitor sequence used by this trainer."""
    identifiers: Set[str] = set()
    sequences: Set[str] = set()
    counts: Dict[str, int] = {}

    def add(source: str, identifier: str, sequence: str) -> None:
        normalized = normalize_sequence(sequence)
        if not normalized:
            return
        if identifier:
            identifiers.add(normalize_identifier(identifier))
        sequences.add(normalized)
        counts[source] = counts.get(source, 0) + 1

    for record in iter_fold(sequence_valid_path, max_bases):
        add("sequence_validation", record.name, record.sequence)
    for path in bprna_monitor_paths:
        if not path or not os.path.isfile(path):
            continue
        source = "bprna:" + os.path.basename(path)
        for record in iter_fold(path, max_bases):
            add(source, record.name, record.sequence)

    nc_root = os.path.join(downstream_data_root, "ncFamilyClassification", "ncRNA-deep")
    for name in ("x_train.fasta", "x_val.fasta", "x_test.fasta"):
        path = os.path.join(nc_root, name)
        if os.path.isfile(path):
            for identifier, sequence in _read_fasta_records(path):
                add("downstream_nc_family", identifier, sequence)

    csv_specs = (
        (os.path.join(downstream_data_root, "UTR_MRL", "GSM4084997_varying_length_25to100.csv"), ("utr",), "downstream_mrl"),
        (os.path.join(downstream_data_root, "MirTar", "MirTarRAW.csv"), ("a_seq", "b_seq"), "downstream_mirtar"),
        (os.path.join(downstream_data_root, "RNA-RNA_interaction", "testset_200x200.csv"), ("rna1_cdna", "rna2_cdna"), "downstream_rime_test"),
    )
    for path, fields, source in csv_specs:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for field in fields:
                    add(source, "", row.get(field, ""))

    rime_root = os.path.join(downstream_data_root, "RNA-RNA_interaction")
    metadata_path = os.path.join(rime_root, "rna_metadata.csv")
    pairs_path = os.path.join(rime_root, "transcript_pairs.csv")
    if os.path.isfile(metadata_path) and os.path.isfile(pairs_path):
        metadata: Dict[str, str] = {}
        with open(metadata_path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                metadata[row["gene_ensembl_id"]] = normalize_sequence(row["sequence_cdna"])
        with open(pairs_path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for prefix in ("rna1", "rna2"):
                    sequence = metadata.get(row[f"{prefix}_id"], "")
                    if not sequence:
                        continue
                    start = int(float(row[f"region{prefix[-1]}_start"]))
                    length = int(float(row[f"region{prefix[-1]}_length"]))
                    left = max(0, start - 200)
                    right = min(len(sequence), start + length + 200)
                    add("downstream_rime_train_valid", row[f"{prefix}_id"], sequence[left:right])

    secondary_root = os.path.join(downstream_data_root, "SecondaryStructurePrediction")
    secondary_names = (
        "BpRNA-Spot_train_lt1024.fold",
        "BpRNA-Spot_validation_lt1024.fold",
        "BpRNA-Spot_test_lt1024.fold",
        "BpRNA-Spot_train4Archivell.fold",
        "BpRNA-Spot_valid4Archivell.fold",
        "ArchiveII_lt1024_3950_noleak80.fold",
    )
    for name in secondary_names:
        path = os.path.join(secondary_root, name)
        if not os.path.isfile(path):
            continue
        for record in iter_fold(path, max_bases):
            add("downstream_secondary", record.name, record.sequence)

    return LeakageExclusions(identifiers, sequences, counts)


def audit_training_source(
    path: str,
    exclusions: LeakageExclusions,
    max_bases: int,
) -> Dict[str, object]:
    total = id_overlap = sequence_overlap = excluded = post_filter_overlap = 0
    for record in iter_fold(path, max_bases):
        total += 1
        by_id, by_sequence = exclusions.matches(record)
        id_overlap += int(by_id)
        sequence_overlap += int(by_sequence)
        should_exclude = should_exclude_record(record, exclusions)
        excluded += int(should_exclude)
        retained = not should_exclude
        post_filter_overlap += int(retained and (by_id or by_sequence))
    return {
        "training_source": path,
        "training_records_scanned": total,
        "source_overlap_by_id": id_overlap,
        "source_overlap_by_sequence": sequence_overlap,
        "training_records_excluded": excluded,
        "training_records_kept": total - excluded,
        "post_filter_overlap": post_filter_overlap,
        "filter_policy": "exclude_if_id_or_exact_sequence_matches",
        "filter_enforced_during_streaming": True,
        "exclusion_identifier_count": len(exclusions.identifiers),
        "exclusion_sequence_count": len(exclusions.sequences),
        "monitor_source_counts": exclusions.source_counts,
    }


def parse_pairs(structure: str, min_distance: int) -> List[Tuple[int, int]]:
    stacks = {opening: [] for opening in BRACKETS}
    closing = {value: key for key, value in BRACKETS.items()}
    pairs: List[Tuple[int, int]] = []
    for index, symbol in enumerate(structure):
        if symbol in stacks:
            stacks[symbol].append(index)
        elif symbol in closing and stacks[closing[symbol]]:
            left = stacks[closing[symbol]].pop()
            if index - left >= min_distance:
                pairs.append((left, index))
    return pairs


def distance_bucket(distance: int) -> int:
    return min(15, max(0, int(math.log2(max(1, distance)))))


class StreamingSequenceDataset(IterableDataset):
    """Worker-sharded deterministic stream with replay-safe resume skipping."""

    def __init__(
        self,
        path: str,
        max_bases: int,
        seed: int,
        shuffle_buffer: int,
        exclusions: Optional[LeakageExclusions] = None,
        max_records: int = 0,
        start_examples: int = 0,
        rank: int = 0,
        world_size: int = 1,
        global_shuffle: bool = False,
        offset_index_path: str = "",
    ) -> None:
        self.path = path
        self.max_bases = max_bases
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.exclusions = exclusions
        self.max_records = max_records
        self.start_examples = int(start_examples)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.global_shuffle = bool(global_shuffle)
        self.offset_index_path = offset_index_path

    def __iter__(self):
        if self.global_shuffle:
            yield from self._iter_global_shuffle()
            return
        worker = get_worker_info()
        local_worker_id = worker.id if worker else 0
        local_workers = worker.num_workers if worker else 1
        worker_id = self.rank * local_workers + local_worker_id
        workers = self.world_size * local_workers
        skip_remaining = self.start_examples // workers + int(
            worker_id < self.start_examples % workers
        )
        cycle = 0
        while True:
            rng = random.Random(self.seed + worker_id + cycle * 1_000_003)
            buffer: List[Dict[str, str]] = []
            accepted = 0
            for record_index, record in enumerate(iter_fold(self.path, self.max_bases)):
                if record_index % workers != worker_id:
                    continue
                if self.max_records and accepted >= self.max_records:
                    break
                if self.exclusions and should_exclude_record(record, self.exclusions):
                    continue
                accepted += 1
                item = {
                    "name": record.name,
                    "sequence": record.sequence,
                    "proposal_structure": record.structure,
                    "kind": "sequence",
                }
                if self.shuffle_buffer <= 1:
                    candidate = item
                elif len(buffer) < self.shuffle_buffer:
                    buffer.append(item)
                    continue
                else:
                    slot = rng.randrange(len(buffer))
                    candidate = buffer[slot]
                    buffer[slot] = item
                if skip_remaining:
                    skip_remaining -= 1
                else:
                    yield candidate
            rng.shuffle(buffer)
            for candidate in buffer:
                if skip_remaining:
                    skip_remaining -= 1
                else:
                    yield candidate
            if accepted == 0:
                raise RuntimeError(f"no sequence records found in {self.path}")
            cycle += 1

    def _iter_global_shuffle(self):
        """One full permutation per epoch, sharded across workers.

        The permutation covers the whole offset index, so length blocks stored
        contiguously in the fold file are mixed inside every epoch. The epoch
        seed changes the permutation only; mask randomness stays in the collator.
        """
        worker = get_worker_info()
        local_worker_id = worker.id if worker else 0
        local_workers = worker.num_workers if worker else 1
        worker_id = self.rank * local_workers + local_worker_id
        workers = self.world_size * local_workers
        offsets = np.memmap(self.offset_index_path, dtype="<u8", mode="r")
        record_count = int(offsets.shape[0])
        if self.max_records:
            record_count = min(record_count, self.max_records * workers)
        local_positions = np.arange(worker_id, record_count, workers, dtype=np.int64)
        if local_positions.size == 0:
            raise RuntimeError(f"worker {worker_id} received no shuffled records from {self.path}")
        skip_remaining = self.start_examples // workers + int(
            worker_id < self.start_examples % workers
        )
        cycle = 0
        with open(self.path, "rb") as handle:
            while True:
                order = local_positions.copy()
                rng = np.random.RandomState(
                    (int(self.seed) + worker_id + cycle * 1_000_003) % (2**32)
                )
                rng.shuffle(order)
                yielded = False
                for position in order:
                    record = read_fold_at(handle, int(offsets[int(position)]), self.max_bases)
                    if record is None:
                        continue
                    if self.exclusions and should_exclude_record(record, self.exclusions):
                        continue
                    if skip_remaining:
                        skip_remaining -= 1
                        continue
                    yielded = True
                    yield {
                        "name": record.name,
                        "sequence": record.sequence,
                        "proposal_structure": record.structure,
                        "kind": "sequence",
                    }
                if not yielded and skip_remaining == 0:
                    raise RuntimeError(f"shuffled epoch produced no records from {self.path}")
                cycle += 1


class FiniteSequenceDataset(Dataset):
    def __init__(self, path: str, max_bases: int, limit: int) -> None:
        self.items = [
            {
                "name": record.name,
                "sequence": record.sequence,
                "proposal_structure": record.structure,
                "kind": "sequence",
            }
            for record in iter_fold(path, max_bases)
        ][:limit]
        if not self.items:
            raise RuntimeError(f"no validation sequences found in {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class StructureDataset(Dataset):
    def __init__(self, path: str, max_bases: int, min_distance: int, limit: int = 0) -> None:
        self.items: List[Dict[str, str]] = []
        for record in iter_fold(path, max_bases):
            if parse_pairs(record.structure, min_distance):
                self.items.append(
                    {
                        "name": record.name,
                        "sequence": record.sequence,
                        "structure": record.structure,
                        "kind": "structure",
                    }
                )
            if limit and len(self.items) >= limit:
                break
        if not self.items:
            raise RuntimeError(f"no structure records found in {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def objective_edges_for_length(length: int, schedule: str, default_edges: int) -> int:
    """Length bins keep short sequences at 16 and give longer sequences more edges."""
    if schedule != "by_length":
        return int(default_edges)
    if length < 256:
        return 16
    if length < 384:
        return 24
    if length < 512:
        return 32
    return 40


class PersistentPairCollator:
    def __init__(
        self,
        tokenizer: RNATokenizer,
        token_mask_probability: float,
        span_batch_probability: float,
        span_mask_probability: float,
        mean_span_length: float,
        edges_per_sequence: int,
        forced_slots: int,
        min_distance: int,
        retrieval_partners_per_anchor: Optional[int] = None,
        proposal_probability_ref=None,
        endpoint_fraction_ref=None,
        deterministic_seed: Optional[int] = None,
        pad_multiple: int = 8,
        endpoint_budget_scale: float = 0.0,
        length_edge_schedule: str = "",
    ) -> None:
        self.tokenizer = tokenizer
        self.token_mask_probability = token_mask_probability
        self.span_batch_probability = span_batch_probability
        self.span_mask_probability = span_mask_probability
        self.mean_span_length = mean_span_length
        self.edges_per_sequence = edges_per_sequence
        self.forced_slots = forced_slots
        self.retrieval_partners_per_anchor = (
            forced_slots
            if retrieval_partners_per_anchor is None
            else retrieval_partners_per_anchor
        )
        self.min_distance = min_distance
        self.proposal_probability_ref = proposal_probability_ref
        self.endpoint_fraction_ref = endpoint_fraction_ref
        self.deterministic_seed = deterministic_seed
        self.pad_multiple = pad_multiple
        self.endpoint_budget_scale = float(endpoint_budget_scale)
        self.length_edge_schedule = length_edge_schedule

    def _rng(self, item: Dict[str, str]) -> random.Random:
        if self.deterministic_seed is None:
            return random
        identity = normalize_identifier(item.get("name", ""))
        payload = f"{self.deterministic_seed}|{identity}|{normalize_sequence(item['sequence'])}"
        seed = int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "little")
        return random.Random(seed)

    def _mask_positions(
        self,
        length: int,
        target: int,
        span_mode: bool,
        excluded: Set[int],
        rng: random.Random,
    ) -> List[int]:
        candidates = list(range(1, length + 1))
        candidates = [position for position in candidates if position not in excluded]
        target = min(target, len(candidates))
        if target <= 0:
            return []
        if not span_mode:
            return rng.sample(candidates, target)
        selected = set()
        attempts = 0
        while len(selected) < target and attempts < target * 20:
            attempts += 1
            start = rng.choice(candidates)
            span = max(1, int(rng.expovariate(1.0 / self.mean_span_length)))
            selected.update(
                position
                for position in range(start, min(length + 1, start + span))
                if position not in excluded
            )
        if len(selected) < target:
            remaining = [position for position in candidates if position not in selected]
            selected.update(rng.sample(remaining, min(target - len(selected), len(remaining))))
        return sorted(selected)[:target]

    def _proposal_probability(self) -> float:
        if self.deterministic_seed is not None or self.proposal_probability_ref is None:
            return 0.0
        return float(self.proposal_probability_ref.value)

    def _objective_endpoint_fraction(self) -> float:
        ref = self.endpoint_fraction_ref
        if ref is None:
            return 0.0
        return float(ref.value)

    @staticmethod
    def _sequence_only_anchor_order(
        positions: Sequence[int], rng: random.Random
    ) -> List[int]:
        anchors = list(positions)
        rng.shuffle(anchors)
        return anchors

    def _sample_edges(
        self,
        sequence: str,
        proposal_structure: str,
        endpoint_budget: int,
        rng: random.Random,
        edge_limit: Optional[int] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """Return directed ranking records: (anchor, partner, group, is_fold)."""
        positions = [index for index, value in enumerate(sequence) if value in BASE_TO_ID]
        if len(positions) < 2 or endpoint_budget < 2:
            return []
        limit = self.edges_per_sequence if edge_limit is None else int(edge_limit)
        proposal_probability = self._proposal_probability()
        proposal_map: Dict[int, List[int]] = {}
        if proposal_probability > 0:
            for left, right in parse_pairs(proposal_structure, self.min_distance):
                if sequence[left] not in BASE_TO_ID or sequence[right] not in BASE_TO_ID:
                    continue
                proposal_map.setdefault(left, []).append(right)
                proposal_map.setdefault(right, []).append(left)
        # Anchor identity and order are sequence-only. RNAfold may only suggest
        # one partner after an anchor has already been selected.
        anchors = self._sequence_only_anchor_order(positions, rng)
        degree = [0] * len(sequence)
        selected: Set[Tuple[int, int]] = set()
        endpoints: Set[int] = set()
        output: List[Tuple[int, int, int, int]] = []
        group = 0
        for anchor in anchors:
            if len(output) >= limit:
                break
            partners: List[int] = []

            def add_partner(partner: int, is_fold: bool) -> bool:
                canonical = (min(anchor, partner), max(anchor, partner))
                if (
                    partner == anchor
                    or abs(partner - anchor) < self.min_distance
                    or canonical in selected
                    or degree[anchor] >= self.retrieval_partners_per_anchor
                    or degree[partner] >= self.forced_slots
                ):
                    return False
                new_endpoints = endpoints | {anchor, partner}
                if len(new_endpoints) > endpoint_budget:
                    return False
                selected.add(canonical)
                endpoints.update((anchor, partner))
                degree[anchor] += 1
                degree[partner] += 1
                partners.append(partner)
                output.append((anchor, partner, group, int(is_fold)))
                return True

            proposed = list(proposal_map.get(anchor, ()))
            rng.shuffle(proposed)
            if proposed and rng.random() < proposal_probability:
                for partner in proposed:
                    if add_partner(partner, True):
                        break

            while (
                len(partners) < self.retrieval_partners_per_anchor
                and len(output) < limit
            ):
                choices = [
                    value for value in positions
                    if value != anchor
                    and abs(value - anchor) >= self.min_distance
                    and degree[value] < self.forced_slots
                    and (min(anchor, value), max(anchor, value)) not in selected
                ]
                if not choices:
                    break
                partner = rng.choice(choices)
                if not add_partner(partner, False):
                    break
            if partners:
                group += 1
        return output

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor | bool | str]:
        encoded = [self.tokenizer.encode(item["sequence"]) for item in features]
        width = max(map(len, encoded))
        width = math.ceil(width / self.pad_multiple) * self.pad_multiple
        batch_size = len(features)
        clean = torch.full((batch_size, width), PAD_ID, dtype=torch.long)
        inputs = torch.full_like(clean, PAD_ID)
        attention = torch.zeros_like(clean)
        labels = torch.full_like(clean, -100)
        objective_row: List[int] = []
        objective_i: List[int] = []
        objective_j: List[int] = []
        objective_target: List[int] = []
        objective_is_rnafold: List[int] = []
        retrieval_row: List[int] = []
        retrieval_i: List[int] = []
        retrieval_j: List[int] = []
        retrieval_objective_index: List[int] = []
        retrieval_group: List[int] = []
        retrieval_is_rnafold: List[int] = []
        corruption_modes: List[str] = []
        unique_endpoint_count = 0
        requested_edges = 0

        for row, (item, ids) in enumerate(zip(features, encoded)):
            rng = self._rng(item)
            span_mode = rng.random() < self.span_batch_probability
            corruption_modes.append("span" if span_mode else "token")
            length = len(ids)
            clean[row, :length] = torch.as_tensor(ids)
            inputs[row, :length] = torch.as_tensor(ids)
            attention[row, :length] = 1
            endpoints = set()
            mask_probability = (
                self.span_mask_probability if span_mode else self.token_mask_probability
            )
            sequence_length = len(item["sequence"])
            edge_limit = objective_edges_for_length(
                sequence_length, self.length_edge_schedule, self.edges_per_sequence
            )
            hard_max = max(
                1,
                math.floor(sequence_length * 0.20),
            )
            endpoint_fraction = self._objective_endpoint_fraction()
            scaled_budget = 0
            if endpoint_fraction > 0.0:
                # Target unique endpoints, and do not shrink that set to keep
                # the 15% mask. Degree ≤ 2, so an edge cap of the same size
                # can fill the budget. If the budget exceeds 20% of L, the
                # mask ceiling lifts and the total mask fraction may rise.
                endpoint_budget = max(2, round(sequence_length * endpoint_fraction))
                if endpoint_budget > hard_max:
                    hard_max = min(endpoint_budget, max(2, sequence_length - 2))
                edge_limit = max(edge_limit, endpoint_budget)
                endpoint_budget = min(hard_max, endpoint_budget)
            elif self.endpoint_budget_scale > 0:
                scaled_budget = max(
                    0, round(edge_limit * self.endpoint_budget_scale)
                )
                # The 20% ceiling cannot hold 1.5*K endpoints on short sequences,
                # and those sequences then miss the requested edge budget. Lift
                # the ceiling only to the endpoint budget, never past L-2.
                if scaled_budget > hard_max:
                    hard_max = min(scaled_budget, max(2, sequence_length - 2))
            requested_edges += edge_limit
            target_masks = max(
                1,
                min(
                    hard_max,
                    round(sequence_length * mask_probability),
                ),
            )
            if endpoint_fraction > 0.0:
                pass
            elif hard_max < 2:
                endpoint_budget = 0
            elif self.endpoint_budget_scale > 0:
                endpoint_budget = min(hard_max, scaled_budget)
            else:
                endpoint_budget = min(
                    hard_max,
                    max(2, round(target_masks * 0.60)),
                )
            edges = self._sample_edges(
                item["sequence"],
                item.get("proposal_structure", "." * len(item["sequence"])),
                endpoint_budget,
                rng,
                edge_limit,
            )
            canonical_to_index: Dict[Tuple[int, int], int] = {}
            for anchor, partner, group, is_rnafold in edges:
                left, right = sorted((anchor + 1, partner + 1))
                canonical = (left, right)
                if canonical not in canonical_to_index:
                    canonical_to_index[canonical] = len(objective_row)
                    objective_row.append(row)
                    objective_i.append(left)
                    objective_j.append(right)
                    objective_target.append(
                        (ids[left] - 4) * 4 + (ids[right] - 4)
                    )
                    objective_is_rnafold.append(is_rnafold)
                objective_index = canonical_to_index[canonical]
                retrieval_row.append(row)
                retrieval_i.append(anchor + 1)
                retrieval_j.append(partner + 1)
                retrieval_objective_index.append(objective_index)
                retrieval_group.append(group)
                retrieval_is_rnafold.append(is_rnafold)
                endpoints.update(canonical)
            unique_endpoint_count += len(endpoints)

            ordinary_target = max(0, target_masks - len(endpoints))
            selected = set(endpoints)
            selected.update(self._mask_positions(
                len(item["sequence"]), ordinary_target, span_mode, endpoints, rng
            ))
            if len(selected) > hard_max:
                raise RuntimeError("mask budget exceeded")
            for position in selected:
                labels[row, position] = ids[position]
                if position in endpoints:
                    inputs[row, position] = MASK_ID
                else:
                    draw = rng.random()
                    if draw < 0.8:
                        inputs[row, position] = MASK_ID
                    elif draw < 0.9:
                        inputs[row, position] = rng.randint(4, 7)

        tensor = lambda value: torch.as_tensor(value, dtype=torch.long)
        return {
            "input_ids": inputs,
            "clean_input_ids": clean,
            "attention_mask": attention,
            "labels": labels,
            "objective_edge_row": tensor(objective_row),
            "objective_edge_i": tensor(objective_i),
            "objective_edge_j": tensor(objective_j),
            "objective_edge_target": tensor(objective_target),
            "objective_edge_is_rnafold": tensor(objective_is_rnafold),
            "retrieval_edge_row": tensor(retrieval_row),
            "retrieval_edge_i": tensor(retrieval_i),
            "retrieval_edge_j": tensor(retrieval_j),
            "retrieval_objective_index": tensor(retrieval_objective_index),
            "retrieval_group": tensor(retrieval_group),
            "retrieval_edge_is_rnafold": tensor(retrieval_is_rnafold),
            "has_edges": bool(objective_row),
            "objective_unique_endpoint_count": unique_endpoint_count,
            "objective_edges_requested_mean": requested_edges / max(1, batch_size),
            "mask_fraction": float(labels.ne(-100).sum()) / max(1, int(attention.sum()) - 2 * batch_size),
            "objective_endpoint_mask_fraction": unique_endpoint_count / max(1, int(attention.sum()) - 2 * batch_size),
            "corruption": ",".join(corruption_modes),
        }


class BPRNADiagnosticCollator:
    """Deterministic true-pair/matched-nonpair batches used only under inference mode."""

    def __init__(self, tokenizer, forced_slots: int, min_distance: int, edges: int, seed: int):
        self.tokenizer = tokenizer
        self.forced_slots = forced_slots
        self.min_distance = min_distance
        self.edges = edges
        self.seed = seed

    def _edges(self, item: Dict[str, str]) -> List[Tuple[int, int, int]]:
        sequence = item["sequence"]
        positives = {
            pair for pair in parse_pairs(item["structure"], self.min_distance)
            if sequence[pair[0]] in BASE_TO_ID and sequence[pair[1]] in BASE_TO_ID
        }
        payload = f"{self.seed}|{normalize_identifier(item['name'])}|{sequence}"
        rng = random.Random(int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "little"))
        chosen = list(positives)
        rng.shuffle(chosen)
        chosen = chosen[: max(1, self.edges // 2)]
        output: List[Tuple[int, int, int]] = []
        occupied: Set[Tuple[int, int]] = set()
        degree = [0] * len(sequence)
        for left, right in chosen:
            if degree[left] >= self.forced_slots or degree[right] >= self.forced_slots:
                continue
            output.append((left, right, 1))
            occupied.add((left, right))
            degree[left] += 1
            degree[right] += 1
            left_positions = [i for i, base in enumerate(sequence) if base == sequence[left]]
            right_positions = [i for i, base in enumerate(sequence) if base == sequence[right]]
            candidates = [
                (i, j) for i in left_positions for j in right_positions
                if i < j
                and j - i >= self.min_distance
                and distance_bucket(j - i) == distance_bucket(right - left)
                and (i, j) not in positives
                and (i, j) not in occupied
                and degree[i] < self.forced_slots
                and degree[j] < self.forced_slots
            ]
            if candidates and len(output) < self.edges:
                negative = rng.choice(candidates)
                output.append((negative[0], negative[1], 0))
                occupied.add(negative)
                degree[negative[0]] += 1
                degree[negative[1]] += 1
        return output

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        encoded = [self.tokenizer.encode(item["sequence"]) for item in features]
        width = math.ceil(max(map(len, encoded)) / 8) * 8
        batch = len(features)
        clean = torch.full((batch, width), PAD_ID, dtype=torch.long)
        inputs = torch.full_like(clean, PAD_ID)
        attention = torch.zeros_like(clean)
        rows: List[int] = []
        lefts: List[int] = []
        rights: List[int] = []
        targets: List[int] = []
        labels: List[int] = []
        for row, (item, ids) in enumerate(zip(features, encoded)):
            clean[row, : len(ids)] = torch.as_tensor(ids)
            inputs[row, : len(ids)] = torch.as_tensor(ids)
            attention[row, : len(ids)] = 1
            for left, right, label in self._edges(item):
                left += 1
                right += 1
                inputs[row, left] = MASK_ID
                inputs[row, right] = MASK_ID
                rows.append(row)
                lefts.append(left)
                rights.append(right)
                targets.append((ids[left] - 4) * 4 + (ids[right] - 4))
                labels.append(label)
        tensor = lambda value: torch.as_tensor(value, dtype=torch.long)
        return {
            "input_ids": inputs,
            "clean_input_ids": clean,
            "attention_mask": attention,
            "objective_edge_row": tensor(rows),
            "objective_edge_i": tensor(lefts),
            "objective_edge_j": tensor(rights),
            "objective_edge_target": tensor(targets),
            "structure_label": tensor(labels),
        }


@torch.inference_mode()
def run_bprna_pair_diagnostics(model, loaders, device, residual_cap: float, amp_context) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    forced_labels: List[np.ndarray] = []
    forced_gains: List[np.ndarray] = []
    native_labels: List[np.ndarray] = []
    native_scores: List[np.ndarray] = []
    reciprocal_ranks: List[float] = []
    recall_at_k: List[float] = []
    recall_at_1: List[float] = []
    for loader in loaders:
        for batch in loader:
            batch = move_batch(batch, device)
            if batch["objective_edge_row"].numel():
                with amp_context():
                    output = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        objective_edge_row=batch["objective_edge_row"],
                        objective_edge_i=batch["objective_edge_i"],
                        objective_edge_j=batch["objective_edge_j"],
                        residual_cap=residual_cap,
                    )
                rows = batch["objective_edge_row"]
                left = batch["objective_edge_i"]
                right = batch["objective_edge_j"]
                logp_i = F.log_softmax(output["logits"][rows, left, 4:8].float(), -1)
                logp_j = F.log_softmax(output["logits"][rows, right, 4:8].float(), -1)
                independent = independent_joint_log_probs(logp_i, logp_j)
                joint = fixed_marginal_joint_log_probs(logp_i, logp_j, output["coupling"])
                gain = nll_from_joint_logp(independent, batch["objective_edge_target"]) - nll_from_joint_logp(joint, batch["objective_edge_target"])
                forced_labels.append(batch["structure_label"].cpu().numpy())
                forced_gains.append(gain.cpu().numpy())

            with amp_context():
                native = model(
                    input_ids=batch["clean_input_ids"],
                    attention_mask=batch["attention_mask"],
                    residual_cap=residual_cap,
                )
            q, k = native["retrieval_q"].float(), native["retrieval_k"].float()
            dense = torch.matmul(q, k.transpose(-1, -2)) * math.sqrt(q.shape[-1])
            for row in range(batch["input_ids"].shape[0]):
                true_pairs = {
                    (int(i), int(j)) for r, i, j, label in zip(
                        batch["objective_edge_row"].tolist(), batch["objective_edge_i"].tolist(),
                        batch["objective_edge_j"].tolist(), batch["structure_label"].tolist()
                    ) if r == row and label == 1
                }
                candidates = native["candidate_indices"][row]
                for left, right in true_pairs:
                    ranked = candidates[left].tolist()
                    rank = ranked.index(right) + 1 if right in ranked else 0
                    reciprocal_ranks.append(1.0 / rank if rank else 0.0)
                    recall_at_k.append(float(rank > 0))
                    recall_at_1.append(float(rank == 1))
                    valid = torch.arange(dense.shape[-1], device=device)
                    valid_tokens = (
                        batch["clean_input_ids"][row].ge(4)
                        & batch["clean_input_ids"][row].le(7)
                        & batch["attention_mask"][row].bool()
                    )
                    eligible = valid_tokens & ((valid - left).abs() >= model.min_distance)
                    scores = dense[row, left, eligible].cpu().numpy()
                    partners = valid[eligible].cpu().numpy()
                    labels = (partners == right).astype(np.int64)
                    native_scores.append(scores)
                    native_labels.append(labels)

    result: Dict[str, float] = {}
    if forced_labels:
        labels = np.concatenate(forced_labels)
        gains = np.concatenate(forced_gains)
        result.update({
            "forced_auprc": float(average_precision_score(labels, gains)),
            "forced_auroc": float(roc_auc_score(labels, gains)),
            "forced_true_gain": float(gains[labels == 1].mean()),
            "forced_nonpair_gain": float(gains[labels == 0].mean()),
            "forced_gain_gap": float(gains[labels == 1].mean() - gains[labels == 0].mean()),
        })
    if native_labels:
        labels = np.concatenate(native_labels)
        scores = np.concatenate(native_scores)
        result.update({
            "native_candidate_recall_at_k": float(np.mean(recall_at_k)),
            "native_r_at_1": float(np.mean(recall_at_1)),
            "native_mrr": float(np.mean(reciprocal_ranks)),
            "native_auprc": float(average_precision_score(labels, scores)),
        })
    result["diagnostic_only"] = True
    result["used_for_checkpoint_selection"] = False
    return result


def cycle_loader(loader: DataLoader):
    while True:
        yield from loader


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def pair_weight_at(step: int, args: argparse.Namespace) -> float:
    if step < args.pair_start_step:
        return 0.0
    progress = min(
        1.0,
        (step - args.pair_start_step) / max(1, args.pair_ramp_end_step - args.pair_start_step),
    )
    return args.pair_loss_weight * progress


def retrieval_weight_at(step: int, args: argparse.Namespace) -> float:
    if step < args.retrieval_start_step:
        return 0.0
    progress = min(
        1.0,
        (step - args.retrieval_start_step)
        / max(1, args.retrieval_ramp_end_step - args.retrieval_start_step),
    )
    return args.retrieval_loss_weight * progress


OBJECTIVE_ENDPOINT_FRACTION = 0.10


def rnafold_proposal_probability_at(args: argparse.Namespace) -> float:
    fixed = float(getattr(args, "rnafold_fixed_probability", -1.0))
    if fixed >= 0.0:
        return fixed
    return float(args.rnafold_proposal_probability)


def residual_cap_at(step: int, args: argparse.Namespace) -> float:
    if step < args.pair_start_step:
        return 0.0
    if step < args.residual_ramp_end_step:
        progress = (step - args.pair_start_step) / max(
            1, args.residual_ramp_end_step - args.pair_start_step
        )
        return args.training_residual_cap * progress
    return args.training_residual_cap


def gain_margin_at(step: int, args: argparse.Namespace) -> float:
    del step
    return float(args.retrieval_gain_margin)


def retrieval_qk_lr(args: argparse.Namespace | None) -> float:
    if args is None:
        return 1.0e-6
    return float(args.retrieval_qk_lr)


def apply_qk_learning_rates(optimizer, step: int, args: argparse.Namespace) -> None:
    del step
    learning_rate = retrieval_qk_lr(args)
    for group in optimizer.param_groups:
        name = group.get("name")
        if name in ("retrieval_q", "retrieval_k"):
            group["lr"] = learning_rate


def named_group_lr(optimizer, name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    return float("nan")


def support_aware_anchor_confidence(
    teacher_gain: torch.Tensor,
    margin: float,
    scale: float,
    support_floor: float,
    support_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_gain = teacher_gain.clamp_min(0.0)
    top = positive_gain.topk(min(2, positive_gain.numel())).values
    best = top[0]
    second = top[1] if top.numel() > 1 else torch.zeros_like(best)
    top2_mean = top.mean()
    best_strength = ((best - margin) / scale).clamp(0.0, 1.0)
    support_strength = (second / support_scale).clamp(0.0, 1.0)
    confidence = best_strength * (
        support_floor + (1.0 - support_floor) * support_strength
    )
    return confidence, best, second, top2_mean


def gain_distilled_retrieval_group_loss(
    group_gain: torch.Tensor,
    group_logits: torch.Tensor,
    group_overlap: torch.Tensor,
    temperature: float,
    margin: float,
    scale: float,
    support_floor: float,
    support_scale: float,
):
    """Return discovery loss/statistics, or None for an all-native group.

    Native logits are constant score references. Confidence is determined only
    by missed candidates, while the existing all-candidate teacher distribution
    is intentionally preserved.
    """
    if bool(group_overlap.all()):
        return None
    discovery_gain = group_gain[~group_overlap]
    confidence, best, second, top2_mean = support_aware_anchor_confidence(
        discovery_gain, margin, scale, support_floor, support_scale
    )
    target_distribution = F.softmax(group_gain / temperature, dim=0)
    student_logits = torch.where(group_overlap, group_logits.detach(), group_logits)
    retrieval_logp = F.log_softmax(student_logits / temperature, dim=0)
    loss = confidence * -(target_distribution * retrieval_logp).sum()
    return loss, confidence, best, second, top2_mean


def retrieval_group_coverage_metrics(
    total_groups: int,
    discovery_groups: int,
    all_native_groups: int,
    active_discovery_groups: int,
    confidence_values: Optional[torch.Tensor],
    zero: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    denominator = max(1, int(total_groups))
    active_all = zero.new_tensor(float(active_discovery_groups) / denominator)
    discovery = zero.new_tensor(float(discovery_groups) / denominator)
    all_native = zero.new_tensor(float(all_native_groups) / denominator)
    active_given_discovery = (
        confidence_values.gt(0).float().mean()
        if confidence_values is not None and confidence_values.numel()
        else zero
    )
    return active_all, discovery, all_native, active_given_discovery


def raw_overlap_gain_metrics(
    retrieval_raw_gain: torch.Tensor,
    native_overlap: torch.Tensor,
    zero: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    overlap_gain = (
        retrieval_raw_gain[native_overlap].mean()
        if bool(native_overlap.any()) else zero
    )
    nonoverlap = ~native_overlap
    nonoverlap_gain = (
        retrieval_raw_gain[nonoverlap].mean()
        if bool(nonoverlap.any()) else zero
    )
    return overlap_gain, nonoverlap_gain, nonoverlap_gain - overlap_gain


def teacher_student_alignment_group_metrics(
    objective_gain: torch.Tensor,
    retrieval_score: torch.Tensor,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    """Measure detached teacher quality and teacher/student rank alignment."""
    if objective_gain.numel() < 2 or retrieval_score.numel() != objective_gain.numel():
        raise ValueError("alignment metrics require at least two paired gain/score values")
    gain = objective_gain.detach().float()
    score = retrieval_score.detach().float()
    top = gain.topk(2).values
    teacher_probability = F.softmax(gain / temperature, dim=0)
    teacher_entropy = -(
        teacher_probability * teacher_probability.clamp_min(1e-12).log()
    ).sum()
    teacher_top_index = gain.argmax()
    other = torch.arange(gain.numel(), device=gain.device).ne(teacher_top_index)
    gain_rank = gain.argsort().argsort().float()
    score_rank = score.argsort().argsort().float()
    gain_rank = gain_rank - gain_rank.mean()
    score_rank = score_rank - score_rank.mean()
    rank_denominator = gain_rank.norm() * score_rank.norm()
    spearman = (
        (gain_rank * score_rank).sum() / rank_denominator
        if bool(rank_denominator.gt(0))
        else gain.new_tensor(float("nan"))
    )
    return {
        "gain_top1": top[0],
        "gain_top2": top[1],
        "gain_gap": top[0] - top[1],
        "teacher_entropy": teacher_entropy,
        "top1_agreement": teacher_top_index.eq(score.argmax()).float(),
        "spearman_gain_score": spearman,
        "score_gap_by_gain_rank": score[teacher_top_index] - score[other].mean(),
    }


def warmup_complete(step: int, args: argparse.Namespace) -> bool:
    return step >= max(
        args.pair_ramp_end_step,
        args.residual_ramp_end_step,
        args.retrieval_ramp_end_step,
    )


def valid_anchor_routing_mean(
    routed: List[Dict[str, torch.Tensor]],
    key: str,
    valid_native_anchor: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    """Mean a [B,L,H] routing diagnostic over routable anchors only."""
    if not bool(valid_native_anchor.any()):
        return zero
    layer_means = [item[key].mean(-1)[valid_native_anchor].mean() for item in routed]
    return torch.stack(layer_means).mean() if layer_means else zero


def candidate_graph_churn(previous, current):
    """Compare fixed-probe candidate sets; return retention, churn, anchor count."""
    retained_values = []
    for (previous_indices, previous_mask), (current_indices, current_mask) in zip(
        previous, current
    ):
        same = current_indices[..., :, None].eq(previous_indices[..., None, :])
        matched_current = (
            same
            & current_mask[..., :, None]
            & previous_mask[..., None, :]
        ).any(-1)
        eligible = current_mask.any(-1) & previous_mask.any(-1)
        retained = matched_current.sum(-1).float() / current_mask.sum(-1).clamp_min(1).float()
        if bool(eligible.any()):
            retained_values.append(retained[eligible])
    if not retained_values:
        return float("nan"), float("nan"), 0
    retention = float(torch.cat(retained_values).mean())
    return retention, 1.0 - retention, sum(value.numel() for value in retained_values)


LENGTH_GATE_MAX_BASES = 0


def _unwrap_module(model):
    return model.module if hasattr(model, "module") else model


def _sequence_content_length(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(-1).clamp_min(2) - 2


def _install_length_gate_residual_detach(model) -> None:
    """Detach pair-to-hidden residuals on long rows so MLM cannot train pair modules there."""
    if getattr(model, "_length_gate_residual_hooked", False):
        return
    model._length_gate_residual_hooked = True
    for step in model.steps:
        original = step.apply_pair_to_hidden

        def _wrap(orig):
            def apply_pair_to_hidden(hidden, state, candidate_indices, candidate_mask):
                result = orig(hidden, state, candidate_indices, candidate_mask)
                short = getattr(model, "_length_gate_short_mask", None)
                if short is None:
                    return result
                update = result["hidden_states"] - hidden
                keep = short.to(dtype=update.dtype).view(-1, 1, 1)
                result["hidden_states"] = (
                    hidden + keep * update + (1.0 - keep) * update.detach()
                )
                return result

            return apply_pair_to_hidden

        step.apply_pair_to_hidden = _wrap(original)


def compute_objective(
    model: PersistentPairStateForMaskedLM,
    batch: Dict,
    residual_cap: float,
    pair_weight: float,
    retrieval_weight: float,
    retrieval_temperature: float,
    retrieval_gain_clip: float,
    retrieval_gain_margin: float,
    retrieval_gain_scale: float,
    retrieval_support_floor: float,
    retrieval_support_scale: float,
    native_low_utilization_threshold: float = 1e-4,
    return_graph_diagnostics: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    core = _unwrap_module(model)
    gate_max = int(LENGTH_GATE_MAX_BASES)
    short_mask = None
    gate_active = bool(gate_max > 0 and core.training)
    if gate_max > 0:
        short_mask = _sequence_content_length(batch["attention_mask"]) < gate_max
        _install_length_gate_residual_detach(core)
        core._length_gate_short_mask = short_mask if gate_active else None
    elif hasattr(core, "_length_gate_short_mask"):
        core._length_gate_short_mask = None
    has_edges = bool(batch["objective_edge_row"].numel())
    edge_kwargs = {}
    if has_edges:
        edge_kwargs = {
            "objective_edge_row": batch["objective_edge_row"],
            "objective_edge_i": batch["objective_edge_i"],
            "objective_edge_j": batch["objective_edge_j"],
            "retrieval_edge_row": batch["retrieval_edge_row"],
            "retrieval_edge_i": batch["retrieval_edge_i"],
            "retrieval_edge_j": batch["retrieval_edge_j"],
        }
    try:
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            residual_cap=residual_cap,
            **edge_kwargs,
        )
    finally:
        if hasattr(core, "_length_gate_short_mask"):
            core._length_gate_short_mask = None
    token_loss = F.cross_entropy(
        output["logits"].reshape(-1, output["logits"].shape[-1]),
        batch["labels"].reshape(-1),
        ignore_index=-100,
    )
    masked = batch["labels"].ne(-100)
    token_accuracy = (
        output["logits"].argmax(-1)[masked].eq(batch["labels"][masked]).float().mean()
    )
    zero = token_loss.new_zeros(())
    pair_nll = independent_nll = retrieval_loss = zero
    pair_nll_ungated = independent_nll_ungated = retrieval_loss_ungated = zero
    length_gate_pair_edge_keep_fraction = zero
    length_gate_retrieval_group_keep_fraction = zero
    pair_accuracy = zero
    gain_positive_fraction = zero
    retrieval_anchor_confidence = zero
    retrieval_active_anchor_fraction = zero
    retrieval_discovery_group_fraction = zero
    retrieval_all_native_group_fraction = zero
    retrieval_active_given_discovery_fraction = zero
    retrieval_anchor_max_gain = zero
    retrieval_anchor_second_positive_gain = zero
    retrieval_anchor_top2_positive_gain = zero
    retrieval_anchor_robust_gain = zero
    objective_anchor_count = zero
    objective_edges_per_anchor = zero
    anchors_with_rnafold_proposal_fraction = zero
    native_objective_overlap_fraction = zero
    gain_native_overlap = zero
    gain_native_nonoverlap = zero
    gain_nonoverlap_minus_overlap = zero
    gain_top1 = zero
    gain_top2 = zero
    gain_gap = zero
    teacher_entropy = zero
    top1_agreement = zero
    spearman_gain_score = zero
    score_gap_by_gain_rank = zero
    alignment_group_count = zero
    spearman_defined_fraction = zero
    diagnostics: Dict[str, torch.Tensor] = {}
    if has_edges:
        rows = batch["objective_edge_row"]
        left = batch["objective_edge_i"]
        right = batch["objective_edge_j"]
        logp_i = F.log_softmax(output["logits"][rows, left, 4:8].float(), dim=-1)
        logp_j = F.log_softmax(output["logits"][rows, right, 4:8].float(), dim=-1)
        independent = independent_joint_log_probs(logp_i, logp_j)
        joint = fixed_marginal_joint_log_probs(logp_i, logp_j, output["coupling"])
        independent_per_edge = nll_from_joint_logp(
            independent, batch["objective_edge_target"]
        )
        pair_per_edge = nll_from_joint_logp(
            joint, batch["objective_edge_target"]
        )
        independent_nll_ungated = independent_per_edge.mean()
        pair_nll_ungated = pair_per_edge.mean()
        independent_nll = independent_nll_ungated
        pair_nll = pair_nll_ungated
        if gate_active:
            edge_keep = short_mask[rows]
            length_gate_pair_edge_keep_fraction = edge_keep.float().mean()
            if bool(edge_keep.any()):
                independent_nll = independent_per_edge[edge_keep].mean()
                pair_nll = pair_per_edge[edge_keep].mean()
            else:
                independent_nll = zero
                pair_nll = zero
        gain = (independent_per_edge - pair_per_edge).detach()
        teacher_gain = gain.clamp(-retrieval_gain_clip, retrieval_gain_clip)
        gain_positive_fraction = gain.gt(0).float().mean()
        pair_accuracy = joint.argmax(-1).eq(
            batch["objective_edge_target"]
        ).float().mean()
        ranking_losses = []
        ranking_losses_ungated = []
        anchor_confidences = []
        anchor_max_gains = []
        anchor_second_positive_gains = []
        anchor_top2_positive_gains = []
        anchor_robust_gains = []
        retrieval_rows = batch["retrieval_edge_row"]
        group_key = (
            retrieval_rows * batch["input_ids"].shape[1]
            + batch["retrieval_edge_i"]
        )
        unique_groups = group_key.unique()
        discovery_group_count = 0
        all_native_group_count = 0
        active_discovery_group_count = 0
        active_rnafold_outside_group_count = 0
        alignment_values: Dict[str, List[torch.Tensor]] = {
            "gain_top1": [],
            "gain_top2": [],
            "gain_gap": [],
            "teacher_entropy": [],
            "top1_agreement": [],
            "spearman_gain_score": [],
            "score_gap_by_gain_rank": [],
        }
        alignment_group_total = 0
        spearman_defined_count = 0
        objective_anchor_count = zero.new_tensor(float(unique_groups.numel()))
        objective_edges_per_anchor = zero.new_tensor(
            float(group_key.numel()) / max(1, unique_groups.numel())
        )
        fold_by_group = []
        for key in unique_groups:
            selected_group = group_key.eq(key)
            fold_by_group.append(
                batch["retrieval_edge_is_rnafold"][selected_group].any().float()
            )
        if fold_by_group:
            anchors_with_rnafold_proposal_fraction = torch.stack(fold_by_group).mean()
        native_for_edges = output["candidate_indices"][
            retrieval_rows, batch["retrieval_edge_i"]
        ]
        native_mask_for_edges = output["candidate_mask"][
            retrieval_rows, batch["retrieval_edge_i"]
        ]
        native_overlap = (
            native_for_edges.eq(batch["retrieval_edge_j"][:, None])
            & native_mask_for_edges
        ).any(-1)
        native_objective_overlap_fraction = native_overlap.float().mean()
        # Training uses clipped teacher_gain; overlap diagnostics intentionally
        # retain the full observed detached residual gain.
        retrieval_raw_gain = gain[batch["retrieval_objective_index"]]
        (
            gain_native_overlap,
            gain_native_nonoverlap,
            gain_nonoverlap_minus_overlap,
        ) = raw_overlap_gain_metrics(retrieval_raw_gain, native_overlap, zero)
        for key in unique_groups:
            selected = group_key.eq(key)
            group_overlap = native_overlap[selected]
            if int(selected.sum()) >= 2:
                selected_raw_gain = retrieval_raw_gain[selected]
                selected_score = output["retrieval_logit"][selected]
                group_alignment = teacher_student_alignment_group_metrics(
                    selected_raw_gain,
                    selected_score,
                    retrieval_temperature,
                )
                alignment_group_total += 1
                for name, value in group_alignment.items():
                    if name == "spearman_gain_score" and not bool(torch.isfinite(value)):
                        continue
                    alignment_values[name].append(value)
                    if name == "spearman_gain_score":
                        spearman_defined_count += 1
            if bool(group_overlap.all()):
                all_native_group_count += 1
                continue
            discovery_group_count += 1
            if selected.sum() < 2:
                continue
            selected_teacher_gain = teacher_gain[
                batch["retrieval_objective_index"][selected]
            ]
            group_result = gain_distilled_retrieval_group_loss(
                selected_teacher_gain,
                output["retrieval_logit"][selected],
                group_overlap,
                retrieval_temperature,
                retrieval_gain_margin,
                retrieval_gain_scale,
                retrieval_support_floor,
                retrieval_support_scale,
            )
            if group_result is None:
                continue
            group_loss, confidence, max_gain, second_gain, top2_positive_gain = group_result
            if bool(confidence.gt(0)):
                active_discovery_group_count += 1
                if bool(batch["retrieval_edge_is_rnafold"][selected].any()):
                    active_rnafold_outside_group_count += 1
            robust_gain = 0.5 * (max_gain + second_gain)
            ranking_losses_ungated.append(group_loss.detach())
            group_is_short = True
            if gate_active:
                group_is_short = bool(short_mask[retrieval_rows[selected][0]])
            if group_is_short:
                ranking_losses.append(group_loss)
                anchor_confidences.append(confidence)
                anchor_max_gains.append(max_gain)
                anchor_second_positive_gains.append(second_gain)
                anchor_top2_positive_gains.append(top2_positive_gain)
                anchor_robust_gains.append(robust_gain)
        retrieval_loss_ungated = (
            torch.stack(ranking_losses_ungated).mean()
            if ranking_losses_ungated
            else zero
        )
        length_gate_retrieval_group_keep_fraction = zero.new_tensor(
            float(len(ranking_losses)) / max(1, len(ranking_losses_ungated))
        )
        if ranking_losses:
            retrieval_loss = torch.stack(ranking_losses).mean()
            confidence_values = torch.stack(anchor_confidences)
            retrieval_anchor_confidence = confidence_values.mean()
            retrieval_anchor_max_gain = torch.stack(anchor_max_gains).mean()
            retrieval_anchor_second_positive_gain = torch.stack(
                anchor_second_positive_gains
            ).mean()
            retrieval_anchor_top2_positive_gain = torch.stack(
                anchor_top2_positive_gains
            ).mean()
            retrieval_anchor_robust_gain = torch.stack(anchor_robust_gains).mean()
        if alignment_group_total:
            alignment_group_count = zero.new_tensor(float(alignment_group_total))
            gain_top1 = torch.stack(alignment_values["gain_top1"]).mean()
            gain_top2 = torch.stack(alignment_values["gain_top2"]).mean()
            gain_gap = torch.stack(alignment_values["gain_gap"]).mean()
            teacher_entropy = torch.stack(alignment_values["teacher_entropy"]).mean()
            top1_agreement = torch.stack(alignment_values["top1_agreement"]).mean()
            score_gap_by_gain_rank = torch.stack(
                alignment_values["score_gap_by_gain_rank"]
            ).mean()
            spearman_defined_fraction = zero.new_tensor(
                float(spearman_defined_count) / alignment_group_total
            )
            if alignment_values["spearman_gain_score"]:
                spearman_gain_score = torch.stack(
                    alignment_values["spearman_gain_score"]
                ).mean()
        (
            retrieval_active_anchor_fraction,
            retrieval_discovery_group_fraction,
            retrieval_all_native_group_fraction,
            retrieval_active_given_discovery_fraction,
        ) = retrieval_group_coverage_metrics(
            unique_groups.numel(),
            discovery_group_count,
            all_native_group_count,
            active_discovery_group_count,
            torch.stack(anchor_confidences) if anchor_confidences else None,
            zero,
        )
        diagnostics = {
            "edge_gain": gain,
            "edge_is_rnafold": batch["objective_edge_is_rnafold"].detach(),
            "retrieval_logit": output["retrieval_logit"].detach(),
            "retrieval_teacher_gain": retrieval_raw_gain.detach(),
            "retrieval_group_key": group_key.detach(),
            "retrieval_row": retrieval_rows.detach(),
            "retrieval_anchor": batch["retrieval_edge_i"].detach(),
            "retrieval_partner": batch["retrieval_edge_j"].detach(),
            "retrieval_native_overlap": native_overlap.detach(),
            "retrieval_is_rnafold": batch["retrieval_edge_is_rnafold"].detach(),
        }
    if return_graph_diagnostics:
        diagnostics["candidate_indices"] = output["candidate_indices"].detach()
        diagnostics["candidate_mask"] = output["candidate_mask"].detach()
    total = token_loss + pair_weight * pair_nll + retrieval_weight * retrieval_loss
    routed = output["layer_metrics"]
    valid_native_anchor = output["candidate_mask"].any(-1)
    valid_token_anchor = (
        batch["attention_mask"].bool()
        & batch["input_ids"].ge(3)
        & batch["input_ids"].le(7)
    )
    slot_attention_values = [
        item["slot_attention"][output["candidate_mask"]] for item in routed
    ]
    slot_utilization_values = [
        item["slot_utilization"][output["candidate_mask"]] for item in routed
    ]
    valid_slot_attention = (
        torch.cat(slot_attention_values) if slot_attention_values else zero.new_empty(0)
    )
    valid_slot_utilization = (
        torch.cat(slot_utilization_values) if slot_utilization_values else zero.new_empty(0)
    )
    valid_native_slot = output["candidate_mask"]
    pair_state_norm = (
        output["pair_state"].float().norm(dim=-1)[valid_native_slot].mean()
        if bool(valid_native_slot.any())
        else zero
    )
    native_candidate_count_per_anchor = (
        output["candidate_mask"].sum(-1).float()[valid_native_anchor].mean()
        if bool(valid_native_anchor.any())
        else zero
    )
    metrics = {
        "loss": total,
        "mlm_loss": token_loss,
        "mlm_accuracy": token_accuracy,
        "masked_tokens": masked.sum().float(),
        "pair_nll": pair_nll,
        "independent_nll": independent_nll,
        "pair_gain": independent_nll - pair_nll,
        "joint16_accuracy": pair_accuracy,
        "gain_positive_fraction": gain_positive_fraction,
        "retrieval_loss": retrieval_loss,
        "retrieval_anchor_confidence": retrieval_anchor_confidence,
        "retrieval_active_anchor_fraction": retrieval_active_anchor_fraction,
        "retrieval_discovery_group_fraction": retrieval_discovery_group_fraction,
        "retrieval_all_native_group_fraction": retrieval_all_native_group_fraction,
        "retrieval_active_given_discovery_fraction": retrieval_active_given_discovery_fraction,
        "retrieval_anchor_max_gain": retrieval_anchor_max_gain,
        "retrieval_anchor_second_positive_gain": retrieval_anchor_second_positive_gain,
        "retrieval_anchor_top2_positive_gain": retrieval_anchor_top2_positive_gain,
        "retrieval_anchor_robust_gain": retrieval_anchor_robust_gain,
        "objective_edge_count": zero.new_tensor(float(batch["objective_edge_row"].numel())),
        "objective_edges_requested_mean": zero.new_tensor(
            float(batch.get("objective_edges_requested_mean", 0.0))
        ),
        "objective_edges_actual_mean": zero.new_tensor(
            float(batch["retrieval_edge_row"].numel()) / max(1, batch["input_ids"].shape[0])
        ),
        "pair_endpoint_count_mean": zero.new_tensor(
            float(batch["objective_unique_endpoint_count"]) / max(1, batch["input_ids"].shape[0])
        ),
        "rnafold_edges_per_sequence": (
            batch["retrieval_edge_is_rnafold"].float().sum() / max(1, batch["input_ids"].shape[0])
            if batch["retrieval_edge_is_rnafold"].numel()
            else zero
        ),
        "rnafold_fraction": (
            batch["retrieval_edge_is_rnafold"].float().mean()
            if batch["retrieval_edge_is_rnafold"].numel()
            else zero
        ),
        "active_groups_per_sequence": zero.new_tensor(
            float(active_discovery_group_count) / max(1, batch["input_ids"].shape[0])
        ),
        "active_outside_groups_per_sequence": zero.new_tensor(
            float(active_discovery_group_count) / max(1, batch["input_ids"].shape[0])
        ),
        "active_rnafold_outside_groups_per_sequence": zero.new_tensor(
            float(active_rnafold_outside_group_count) / max(1, batch["input_ids"].shape[0])
        ),
        "objective_anchor_count": objective_anchor_count,
        "objective_edges_per_anchor": objective_edges_per_anchor,
        "anchors_with_rnafold_proposal_fraction": anchors_with_rnafold_proposal_fraction,
        "native_objective_overlap_fraction": native_objective_overlap_fraction,
        "gain_native_overlap": gain_native_overlap,
        "gain_native_nonoverlap": gain_native_nonoverlap,
        "gain_nonoverlap_minus_overlap": gain_nonoverlap_minus_overlap,
        "gain_top1": gain_top1,
        "gain_top2": gain_top2,
        "gain_gap": gain_gap,
        "teacher_entropy": teacher_entropy,
        "top1_agreement": top1_agreement,
        "spearman_gain_score": spearman_gain_score,
        "score_gap_by_gain_rank": score_gap_by_gain_rank,
        "alignment_group_count": alignment_group_count,
        "spearman_defined_fraction": spearman_defined_fraction,
        "objective_unique_endpoint_count": zero.new_tensor(float(batch["objective_unique_endpoint_count"])),
        "mask_fraction": zero.new_tensor(float(batch["mask_fraction"])),
        "objective_endpoint_mask_fraction": zero.new_tensor(float(batch["objective_endpoint_mask_fraction"])),
        "rnafold_objective_edge_fraction": (
            batch["retrieval_edge_is_rnafold"].float().mean()
            if batch["retrieval_edge_is_rnafold"].numel() else zero
        ),
        "pair_state_norm": pair_state_norm,
        "gate": torch.stack([item["gate_mean"] for item in routed]).mean(),
        "native_candidate_count_per_anchor": native_candidate_count_per_anchor,
        "native_routable_anchor_fraction": (
            valid_native_anchor[valid_token_anchor].float().mean()
            if bool(valid_token_anchor.any()) else zero
        ),
        # Null/mass now describe only anchors with at least one valid native edge.
        "effective_pair_attention_mass": valid_anchor_routing_mean(
            routed, "effective_pair_attention_mass", valid_native_anchor, zero
        ),
        "true_null_probability": valid_anchor_routing_mean(
            routed, "true_null_probability", valid_native_anchor, zero
        ),
        "native_slot_attention_mean": (
            valid_slot_attention.mean() if valid_slot_attention.numel() else zero
        ),
        "native_slot_utilization_mean": (
            valid_slot_utilization.mean() if valid_slot_utilization.numel() else zero
        ),
        "native_low_utilization_fraction": (
            valid_slot_utilization.lt(native_low_utilization_threshold).float().mean()
            if valid_slot_utilization.numel() else zero
        ),
        "update_ratio": torch.stack([item["update_ratio"] for item in routed]).mean(),
    }
    if gate_max > 0:
        metrics.update(
            {
                "length_gate_max_bases": zero.new_tensor(float(gate_max)),
                "length_gate_active": zero.new_tensor(float(gate_active)),
                "length_gate_short_fraction": (
                    short_mask.float().mean() if short_mask is not None else zero
                ),
                "length_gate_long_fraction": (
                    1.0 - (short_mask.float().mean() if short_mask is not None else zero)
                ),
                "pair_nll_ungated": pair_nll_ungated.detach(),
                "independent_nll_ungated": independent_nll_ungated.detach(),
                "retrieval_loss_ungated": retrieval_loss_ungated.detach(),
                "length_gate_pair_edge_keep_fraction": length_gate_pair_edge_keep_fraction.detach(),
                "length_gate_retrieval_group_keep_fraction": length_gate_retrieval_group_keep_fraction.detach(),
            }
        )
    return total, metrics, diagnostics


def binary_diagnostics(targets: List[np.ndarray], logits: List[np.ndarray]) -> Dict[str, float]:
    if not targets:
        return {}
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    target = np.concatenate(targets).astype(np.int64)
    score = np.concatenate(logits).astype(np.float64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
    result = {
        "auprc": float(average_precision_score(target, probability)),
        "f1_at_0.5": float(f1_score(target, probability >= 0.5, zero_division=0)),
    }
    result["auroc"] = float(roc_auc_score(target, probability)) if len(np.unique(target)) > 1 else float("nan")
    return result


class InProcessDownstreamRuntime:
    """Minimal runtime interface used by run_full_downstream_benchmark tasks."""

    def __init__(self, model, tokenizer, device, args):
        self.device = device
        self.tokenizer = tokenizer
        self.hidden_size = int(model.config.hidden_size)
        self.model = PersistentPairStateFoundationView(model)
        self.args = args
        self.amp_enabled = device.type == "cuda"
        self.amp_dtype = torch.bfloat16

    def autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)

    @torch.inference_mode()
    def encode(
        self,
        sequences: Sequence[str],
        struct_dots: Optional[Sequence[Optional[str]]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        del struct_dots
        if not sequences:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        max_raw = self.args.max_seq_length - 2
        lengths = [min(len(sequence), max_raw) for sequence in sequences]
        order = sorted(range(len(sequences)), key=lambda index: lengths[index])
        output = np.empty((len(sequences), self.hidden_size), dtype=np.float32)
        nominal = int(batch_size or self.args.embedding_batch_size)
        done = 0
        self.model.eval()
        while done < len(order):
            current = max(1, lengths[order[done]])
            adaptive = max(1, min(nominal, int(2_000_000 / (current * current))))
            chosen = order[done : done + adaptive]
            encoded = [
                self.tokenizer.encode(
                    sequences[index].upper().replace("T", "U")[:max_raw]
                )
                for index in chosen
            ]
            width = max(map(len, encoded))
            ids = torch.full(
                (len(chosen), width), PAD_ID, dtype=torch.long, device=self.device
            )
            attention = torch.zeros_like(ids)
            for row, values in enumerate(encoded):
                ids[row, : len(values)] = torch.as_tensor(values, device=self.device)
                attention[row, : len(values)] = 1
            with self.autocast():
                hidden = self.model.encoder(
                    input_ids=ids,
                    struct_input_ids=None,
                    attention_mask=attention,
                )
                mask = attention.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            values = pooled.float().cpu().numpy()
            for row, index in enumerate(chosen):
                output[index] = values[row]
            done += len(chosen)
        return output


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=True)
    os.replace(temporary, path)


def append_event(path: Path, kind: str, payload: Dict) -> None:
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, **payload}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")


def downstream_tasks_for_step(args, step: int) -> List[str]:
    """Select diagnostic downstream tasks without changing core validation cadence."""
    if args.disable_downstream_monitor:
        return []
    configured = [value.strip() for value in args.downstream_tasks.split(",") if value.strip()]
    full_steps = args.downstream_monitor_steps or args.valid_steps
    if step % full_steps == 0:
        return configured
    if (
        args.downstream_secondary_steps
        and step % args.downstream_secondary_steps == 0
        and "secondary" in configured
    ):
        return ["secondary"]
    return []


def run_downstream_monitor(
    model,
    tokenizer,
    device,
    args,
    step: int,
    task_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    from downstream import run_full_downstream_benchmark as benchmark

    configured = [value.strip() for value in args.downstream_tasks.split(",") if value.strip()]
    task_names = configured if task_names is None else list(task_names)
    unconfigured = set(task_names) - set(configured)
    if unconfigured:
        raise ValueError(f"downstream tasks are not configured: {sorted(unconfigured)}")
    unsupported = set(task_names) - set(benchmark.TASKS)
    if unsupported:
        raise ValueError(f"unsupported downstream tasks: {sorted(unsupported)}")
    if "tertiary" in task_names:
        raise ValueError("tertiary structure monitoring is intentionally disabled")
    monitor_dir = Path(args.output_dir) / "downstream_monitor" / f"step-{step}"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    bench_args = SimpleNamespace(
        data_root=args.downstream_data_root,
        device=str(device),
        amp_dtype="bf16",
        max_seq_length=args.max_seq_length,
        embedding_batch_size=args.downstream_embedding_batch_size,
        head_batch_size=args.downstream_head_batch_size,
        head_epochs=args.downstream_head_epochs,
        head_patience=args.downstream_head_patience,
        secondary_epochs=args.downstream_secondary_epochs,
        secondary_batch_size=1,
        secondary_eval_batch_size=1,
        secondary_patience=1,
        secondary_min_delta=1e-4,
        secondary_protocol="archiveii_noleak80",
        secondary_input_mode="base_only",
        seed=args.downstream_seed,
        smoke_limit=args.downstream_monitor_records,
    )
    runtime = InProcessDownstreamRuntime(model, tokenizer, device, bench_args)
    summary: Dict[str, object] = {
        "step": step,
        "diagnostic_only": True,
        "used_for_checkpoint_selection": False,
        "encoder_pair_residual_cap": float(runtime.model.encoder.residual_cap),
        "tasks": {},
    }
    was_training = model.training
    original_trainable = [parameter.requires_grad for parameter in model.parameters()]
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        for task_index, task_name in enumerate(task_names):
            result_path = monitor_dir / f"{task_name}_metrics.json"
            if result_path.exists():
                with open(result_path, encoding="utf-8") as handle:
                    result = json.load(handle)
            else:
                seed_all(args.downstream_seed + task_index * 1009)
                try:
                    result = benchmark.TASKS[task_name](runtime, bench_args)
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    if args.downstream_fail_fast:
                        raise
                atomic_json(result_path, result)
            summary["tasks"][task_name] = result
            print(
                "[downstream] "
                + json.dumps({"step": step, "task": task_name, "result": result}, allow_nan=True),
                flush=True,
            )
    finally:
        for parameter, enabled in zip(model.parameters(), original_trainable):
            parameter.requires_grad_(enabled)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if was_training:
            model.train()
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(monitor_dir / "metrics_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="500M persistent pair-state RNA pretraining")
    parser.add_argument("--sequence_train_path", required=True)
    parser.add_argument("--sequence_valid_path", required=True)
    parser.add_argument(
        "--bprna_monitor_paths",
        default="",
        help="comma-separated real-structure fold files used only for diagnostics/leakage exclusion",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_from", default="")
    parser.add_argument(
        "--init_from",
        default="",
        help="model-only warm start; unlike --resume_from this does not require trainer_state.pt",
    )
    parser.add_argument(
        "--init_step",
        type=int,
        default=0,
        help="optimizer step assigned to a model-only warm start",
    )
    parser.add_argument(
        "--init_examples_consumed",
        type=int,
        default=-1,
        help="stream offset for --init_from; negative derives it from init_step and effective batch",
    )
    parser.add_argument(
        "--run_steps",
        type=int,
        default=0,
        help="stop after this many optimizer updates; zero runs through max_steps",
    )
    parser.add_argument(
        "--stability_guard_start_step",
        type=int,
        default=0,
        help="logical step at which optional replay guards become active",
    )
    parser.add_argument(
        "--max_train_mlm_loss",
        type=float,
        default=0.0,
        help="stop cleanly when a logging-window MLM loss exceeds this; zero disables",
    )
    parser.add_argument(
        "--max_train_pair_state_norm",
        type=float,
        default=0.0,
        help="stop cleanly when a logging-window pair-state norm exceeds this; zero disables",
    )

    parser.add_argument("--num_hidden_layers", type=int, default=26)
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--num_attention_heads", type=int, default=20)
    parser.add_argument("--intermediate_size", type=int, default=5120)
    parser.add_argument("--max_seq_length", type=int, default=1026)
    parser.add_argument("--attention_backend", default="auto")
    parser.add_argument("--pair_layer_start", type=int, default=18)
    parser.add_argument("--pair_dim", type=int, default=128)
    parser.add_argument("--pair_heads", type=int, default=4)
    parser.add_argument("--max_candidates", type=int, default=8)
    parser.add_argument("--forced_slots", type=int, default=2)
    parser.add_argument("--retrieval_partners_per_anchor", type=int, default=2)
    parser.add_argument("--pair_min_distance", type=int, default=4)
    parser.add_argument("--pair_mass_cap", type=float, default=0.25)
    parser.add_argument("--training_residual_cap", type=float, default=0.006)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")

    parser.add_argument("--batch_size", type=int, default=18)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--pair_lr_multiplier", type=float, default=1.5)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--pair_loss_weight", type=float, default=0.1)
    parser.add_argument("--retrieval_loss_weight", type=float, default=0.02)
    parser.add_argument("--retrieval_temperature", type=float, default=0.25)
    parser.add_argument("--retrieval_gain_clip", type=float, default=0.25)
    parser.add_argument("--retrieval_gain_margin", type=float, default=0.02)
    parser.add_argument("--retrieval_gain_scale", type=float, default=0.10)
    parser.add_argument("--retrieval_support_floor", type=float, default=0.20)
    parser.add_argument("--retrieval_support_scale", type=float, default=0.10)
    parser.add_argument(
        "--native_low_utilization_threshold", type=float, default=1e-4,
        help="diagnostic-only threshold for native slot utilization",
    )
    parser.add_argument(
        "--graph_probe_batches", type=int, default=2,
        help="leading deterministic validation batches retained for candidate churn",
    )
    parser.add_argument("--pair_start_step", type=int, default=5_000)
    parser.add_argument("--pair_ramp_end_step", type=int, default=25_000)
    parser.add_argument("--retrieval_start_step", type=int, default=25_000)
    parser.add_argument("--retrieval_ramp_end_step", type=int, default=50_000)
    parser.add_argument("--residual_ramp_end_step", type=int, default=25_000)
    parser.add_argument("--rnafold_proposal_probability", type=float, default=1.0)
    parser.add_argument("--fused_adam", action="store_true")

    parser.add_argument(
        "--length_gate_max_bases",
        type=int,
        default=0,
        help="if >0, training pair/retrieval grads skip sequences with content length >= this",
    )
    parser.add_argument("--token_mask_probability", type=float, default=0.15)
    parser.add_argument("--span_batch_probability", type=float, default=0.2)
    parser.add_argument("--span_mask_probability", type=float, default=0.15)
    parser.add_argument("--mean_span_length", type=float, default=4.0)
    parser.add_argument("--edges_per_sequence", type=int, default=12)
    parser.add_argument(
        "--length_edge_schedule",
        default="",
        help="by_length uses 16/24/32/40 edges below 256/384/512 and at or above 512",
    )
    parser.add_argument(
        "--endpoint_budget_scale",
        type=float,
        default=0.0,
        help="if >0, objective endpoint budget is scale * edges, lifting the 20%% ceiling when it cannot hold that budget",
    )
    parser.add_argument(
        "--rnafold_fixed_probability",
        type=float,
        default=-1.0,
        help="if >=0, use this RNAfold proposal probability instead of --rnafold_proposal_probability",
    )
    parser.add_argument("--retrieval_qk_lr", type=float, default=1.0e-6)
    parser.add_argument("--checkpoint_weights_only", action="store_true")
    parser.add_argument("--checkpoint_overflow_dir", default="")
    parser.add_argument(
        "--native_membership_paths",
        default="",
        help="comma-separated folds for full directed-pair Top-8 membership",
    )
    parser.add_argument(
        "--structure_monitor_records",
        type=int,
        default=-1,
        help="structure diagnostic limit; negative uses valid_records; zero keeps every record",
    )
    parser.add_argument("--shuffle_buffer", type=int, default=20_000)
    parser.add_argument(
        "--global_shuffle",
        action="store_true",
        help="permute the full fold offset index each epoch instead of a small streaming buffer",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_sequence_train_records", type=int, default=0)
    parser.add_argument("--valid_records", type=int, default=2_048)
    parser.add_argument("--valid_batches", type=int, default=32)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--valid_steps", type=int, default=10_000)
    parser.add_argument("--save_steps", type=int, default=25_000)
    parser.add_argument("--keep_last_checkpoints", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-3)

    parser.add_argument("--disable_downstream_monitor", action="store_true")
    parser.add_argument(
        "--downstream_tasks",
        default="nc_family,mrl,mirtar,rime,secondary",
    )
    parser.add_argument(
        "--downstream_data_root",
        default="/root/autodl-tmp/RIPPLE/downstream",
    )
    parser.add_argument("--downstream_monitor_records", type=int, default=256)
    parser.add_argument(
        "--downstream_monitor_steps",
        type=int,
        default=0,
        help="full downstream interval in global steps; 0 follows every validation",
    )
    parser.add_argument(
        "--downstream_secondary_steps",
        type=int,
        default=0,
        help="secondary-only interval between full downstream monitors; 0 disables it",
    )
    parser.add_argument("--downstream_embedding_batch_size", type=int, default=8)
    parser.add_argument("--downstream_head_batch_size", type=int, default=512)
    parser.add_argument("--downstream_head_epochs", type=int, default=6)
    parser.add_argument("--downstream_head_patience", type=int, default=2)
    parser.add_argument("--downstream_secondary_epochs", type=int, default=2)
    parser.add_argument("--downstream_seed", type=int, default=942)
    parser.add_argument("--eval_seed", type=int, default=20260908)
    parser.add_argument("--downstream_fail_fast", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_downstream", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.resume_from and args.init_from:
        raise ValueError("--resume_from and --init_from are mutually exclusive")
    if (
        args.init_step < 0
        or args.run_steps < 0
        or args.init_examples_consumed < -1
        or args.stability_guard_start_step < 0
        or args.max_train_mlm_loss < 0
        or args.max_train_pair_state_norm < 0
    ):
        raise ValueError("init_step/run_steps must be non-negative and init_examples_consumed >= -1")
    if args.init_step and not args.init_from:
        raise ValueError("--init_step requires --init_from")
    if args.init_step >= args.max_steps:
        raise ValueError("init_step must be less than max_steps")
    if not 0 <= args.training_residual_cap <= 0.02:
        raise ValueError("training_residual_cap must be in [0, 0.02]")
    if not 0 <= args.rnafold_proposal_probability <= 1:
        raise ValueError("rnafold_proposal_probability must be in [0, 1]")
    if args.rnafold_fixed_probability > 1:
        raise ValueError("rnafold_fixed_probability must be <= 1")
    if args.retrieval_qk_lr < 0:
        raise ValueError("retrieval_qk_lr must be non-negative")
    if args.retrieval_temperature <= 0:
        raise ValueError("retrieval_temperature must be positive")
    if args.retrieval_partners_per_anchor < 1:
        raise ValueError("retrieval_partners_per_anchor must be positive")
    if args.retrieval_gain_clip <= 0:
        raise ValueError("retrieval_gain_clip must be positive")
    if args.retrieval_gain_margin < 0:
        raise ValueError("retrieval_gain_margin must be non-negative")
    if args.retrieval_gain_scale <= 0:
        raise ValueError("retrieval_gain_scale must be positive")
    if not 0 <= args.retrieval_support_floor <= 1:
        raise ValueError("retrieval_support_floor must be in [0, 1]")
    if args.retrieval_support_scale <= 0:
        raise ValueError("retrieval_support_scale must be positive")
    if args.native_low_utilization_threshold < 0:
        raise ValueError("native_low_utilization_threshold must be non-negative")
    if args.graph_probe_batches < 0:
        raise ValueError("graph_probe_batches must be non-negative")
    if args.valid_steps <= 0:
        raise ValueError("valid_steps must be positive because downstream monitoring follows validation")
    for name in ("downstream_monitor_steps", "downstream_secondary_steps"):
        interval = getattr(args, name)
        if interval < 0:
            raise ValueError(f"{name} must be non-negative")
        if interval and interval % args.valid_steps:
            raise ValueError(f"{name} must be a multiple of valid_steps")
    if args.save_steps <= 0 or args.logging_steps <= 0:
        raise ValueError("logging_steps and save_steps must be positive")
    if args.keep_last_checkpoints < 1:
        raise ValueError("keep_last_checkpoints must be at least one")
    if args.pair_layer_start >= args.num_hidden_layers:
        raise ValueError("pair_layer_start must be less than num_hidden_layers")
    if args.length_edge_schedule not in ("", "by_length"):
        raise ValueError("length_edge_schedule must be empty or by_length")


def parameter_groups(model, args):
    pair_ids = {id(parameter) for module in model.pair_modules for parameter in module.parameters()}
    q_params = list(model.retriever.q_proj.parameters())
    k_params = list(model.retriever.k_proj.parameters())
    excluded = {id(parameter) for parameter in q_params + k_params}
    groups: Dict[Tuple[bool, bool], List[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if id(parameter) in excluded:
            continue
        is_pair = id(parameter) in pair_ids
        no_decay = parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower()
        groups.setdefault((is_pair, no_decay), []).append(parameter)
    built = [
        {
            "params": parameters,
            "lr": args.lr * (args.pair_lr_multiplier if is_pair else 1.0),
            "weight_decay": 0.0 if no_decay else args.weight_decay,
        }
        for (is_pair, no_decay), parameters in groups.items()
    ]
    learning_rate = retrieval_qk_lr(args)
    built.append({
        "name": "retrieval_q",
        "params": q_params,
        "lr": learning_rate,
        "weight_decay": args.weight_decay,
    })
    built.append({
        "name": "retrieval_k",
        "params": k_params,
        "lr": learning_rate,
        "weight_decay": args.weight_decay,
    })
    return built


def _checkpoint_root(args) -> Path:
    primary = Path(args.output_dir)
    overflow = Path(getattr(args, "checkpoint_overflow_dir", "") or "")
    free = shutil.disk_usage(primary).free
    if overflow and free < 8 * 1024**3:
        overflow.mkdir(parents=True, exist_ok=True)
        return overflow
    return primary


def _drop_other_trainer_states(args, keep: Path) -> None:
    roots = [Path(args.output_dir)]
    overflow = getattr(args, "checkpoint_overflow_dir", "")
    if overflow:
        roots.append(Path(overflow))
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.glob("checkpoint-*"):
            if not path.is_dir() or path.resolve() == keep.resolve():
                continue
            state = path / "trainer_state.pt"
            if state.is_file():
                state.unlink()


def save_checkpoint(
    model, tokenizer, optimizer, scheduler, args, step, best_score, stale,
    examples_consumed,
):
    root = _checkpoint_root(args) if getattr(args, "checkpoint_weights_only", False) else Path(args.output_dir)
    target = root / f"checkpoint-{step}"
    temporary = root / f"checkpoint-{step}.incomplete"
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(str(temporary))
    tokenizer.save_pretrained(str(temporary))
    state = {
        "global_step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_score": best_score,
        "stale_validations": stale,
        "examples_consumed": int(examples_consumed),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "world_size": int(getattr(args, "world_size", 1)),
        "args": vars(args),
    }
    torch.save(state, temporary / "trainer_state.pt")
    os.replace(temporary, target)
    if getattr(args, "checkpoint_weights_only", False):
        _drop_other_trainer_states(args, target)
        pointer = Path(args.output_dir) / "latest_checkpoint.txt"
        pointer.write_text(str(target) + "\n", encoding="utf-8")
    roots = [Path(args.output_dir)]
    overflow = getattr(args, "checkpoint_overflow_dir", "") or ""
    if overflow:
        roots.append(Path(overflow))
    elif not getattr(args, "checkpoint_weights_only", False):
        roots = [root]
    checkpoints = []
    for checkpoint_root in roots:
        if not checkpoint_root.is_dir():
            continue
        checkpoints.extend(
            path
            for path in checkpoint_root.glob("checkpoint-*")
            if path.is_dir() and path.name.split("-")[-1].isdigit()
        )
    checkpoints.sort(key=lambda path: int(path.name.split("-")[-1]))
    for old in checkpoints[: -args.keep_last_checkpoints]:
        if old.resolve() == target.resolve():
            continue
        shutil.rmtree(old)
    print(f"[checkpoint] {target}", flush=True)


def main() -> None:
    args = parse_args()
    global LENGTH_GATE_MAX_BASES
    LENGTH_GATE_MAX_BASES = int(getattr(args, "length_gate_max_bases", 0))
    if LENGTH_GATE_MAX_BASES < 0:
        raise ValueError("length_gate_max_bases must be non-negative")
    if args.smoke:
        args.num_hidden_layers = 4
        args.hidden_size = 256
        args.num_attention_heads = 4
        args.intermediate_size = 768
        args.max_seq_length = min(args.max_seq_length, 512)
        args.pair_layer_start = 2
        args.pair_dim = 64
        args.pair_heads = 4
        args.max_candidates = 4
        args.batch_size = 2
        args.gradient_accumulation_steps = 1
        args.max_steps = 3
        args.pair_start_step = 0
        args.pair_ramp_end_step = 1
        args.retrieval_start_step = 1
        args.retrieval_ramp_end_step = 2
        args.residual_ramp_end_step = 1
        args.max_sequence_train_records = 32
        args.valid_records = 8
        args.valid_batches = 2
        args.logging_steps = 1
        args.valid_steps = 2
        args.save_steps = 3
        args.num_workers = 0
        args.downstream_monitor_records = 32
        if not args.smoke_downstream:
            args.disable_downstream_monitor = True
    validate_args(args)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(hours=6)
        )
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    is_main = rank == 0
    args.world_size = world_size
    args.global_effective_batch_size = (
        args.batch_size * args.gradient_accumulation_steps * world_size
    )
    resume_state = None
    if args.resume_from:
        resume_state = torch.load(
            os.path.join(args.resume_from, "trainer_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
    if args.init_from and not os.path.isfile(
        os.path.join(args.init_from, "pytorch_model.bin")
    ):
        raise FileNotFoundError(f"missing model-only warm start: {args.init_from}")
    initial_examples_consumed = 0
    if resume_state:
        initial_examples_consumed = int(resume_state.get("examples_consumed", 0))
    elif args.init_from:
        initial_examples_consumed = (
            args.init_examples_consumed
            if args.init_examples_consumed >= 0
            else args.init_step * args.global_effective_batch_size
        )
    seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    metrics_path = output_dir / "metrics.jsonl"
    tokenizer = RNATokenizer()
    max_bases = args.max_seq_length - 2
    bprna_monitor_paths = [
        value.strip() for value in args.bprna_monitor_paths.split(",") if value.strip()
    ]
    exclusions = build_leakage_exclusions(
        args.sequence_valid_path,
        bprna_monitor_paths,
        args.downstream_data_root,
        max_bases,
    )
    if is_main:
        leakage_audit = audit_training_source(
            args.sequence_train_path, exclusions, max_bases
        )
        if (
            leakage_audit["post_filter_overlap"] != 0
            or not leakage_audit["filter_enforced_during_streaming"]
        ):
            raise RuntimeError("leakage audit failed")
        atomic_json(output_dir / "leakage_audit.json", leakage_audit)
    if distributed:
        dist.barrier()

    offset_index_path = ""
    if args.global_shuffle:
        offset_index_path = fold_offset_index_path(args.sequence_train_path, max_bases)
        if is_main and not os.path.isfile(offset_index_path):
            indexed = build_fold_offset_index(
                args.sequence_train_path, max_bases, offset_index_path
            )
            print(
                f"[shuffle-index] records={indexed} path={offset_index_path}",
                flush=True,
            )
        if distributed:
            dist.barrier()
        if not os.path.isfile(offset_index_path):
            raise FileNotFoundError(offset_index_path)
    proposal_probability_ref = mp.Value(
        "d", rnafold_proposal_probability_at(args), lock=False
    )
    endpoint_fraction_ref = mp.Value("d", OBJECTIVE_ENDPOINT_FRACTION, lock=False)
    collator = PersistentPairCollator(
        tokenizer,
        args.token_mask_probability,
        args.span_batch_probability,
        args.span_mask_probability,
        args.mean_span_length,
        args.edges_per_sequence,
        args.forced_slots,
        args.pair_min_distance,
        retrieval_partners_per_anchor=args.retrieval_partners_per_anchor,
        proposal_probability_ref=proposal_probability_ref,
        endpoint_fraction_ref=endpoint_fraction_ref,
        endpoint_budget_scale=args.endpoint_budget_scale,
        length_edge_schedule=args.length_edge_schedule,
    )
    eval_collator = PersistentPairCollator(
        tokenizer,
        args.token_mask_probability,
        args.span_batch_probability,
        args.span_mask_probability,
        args.mean_span_length,
        args.edges_per_sequence,
        args.forced_slots,
        args.pair_min_distance,
        retrieval_partners_per_anchor=args.retrieval_partners_per_anchor,
        deterministic_seed=args.eval_seed,
        endpoint_budget_scale=args.endpoint_budget_scale,
        length_edge_schedule=args.length_edge_schedule,
    )
    sequence_train = StreamingSequenceDataset(
        args.sequence_train_path,
        max_bases,
        args.seed,
        args.shuffle_buffer,
        exclusions=exclusions,
        max_records=args.max_sequence_train_records,
        start_examples=initial_examples_consumed,
        rank=rank,
        world_size=world_size,
        global_shuffle=args.global_shuffle,
        offset_index_path=offset_index_path,
    )
    sequence_valid = FiniteSequenceDataset(
        args.sequence_valid_path, max_bases, args.valid_records
    )
    diagnostic_collator = BPRNADiagnosticCollator(
        tokenizer,
        args.forced_slots,
        args.pair_min_distance,
        args.edges_per_sequence,
        args.eval_seed,
    )
    structure_limit = (
        args.valid_records
        if args.structure_monitor_records < 0
        else args.structure_monitor_records
    )
    bprna_loaders = [
        DataLoader(
            StructureDataset(path, max_bases, args.pair_min_distance, structure_limit),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=diagnostic_collator,
        )
        for path in bprna_monitor_paths
    ] if is_main else []
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader_kwargs = {
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker if args.num_workers else None,
        "generator": generator,
    }
    sequence_loader = DataLoader(sequence_train, batch_size=args.batch_size, **loader_kwargs)
    sequence_valid_loader = DataLoader(
        sequence_valid, batch_size=args.batch_size, collate_fn=eval_collator
    )

    base_config = RNAModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_seq_length,
        attention_backend=args.attention_backend,
    )
    model_source = args.resume_from or args.init_from
    if model_source:
        model = PersistentPairStateForMaskedLM.from_pretrained(model_source)
        expected_architecture = {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "intermediate_size": args.intermediate_size,
            "max_position_embeddings": args.max_seq_length,
            "pair_layer_start": args.pair_layer_start,
            "pair_dim": args.pair_dim,
            "pair_heads": args.pair_heads,
            "max_candidates": args.max_candidates,
            "forced_slots": args.forced_slots,
            "pair_min_distance": args.pair_min_distance,
        }
        found = model.checkpoint_config()
        if any(found.get(key) != value for key, value in expected_architecture.items()):
            raise ValueError("resume checkpoint architecture does not match requested configuration")
    else:
        model = PersistentPairStateForMaskedLM(
            base_config,
            pair_layer_start=args.pair_layer_start,
            pair_dim=args.pair_dim,
            pair_heads=args.pair_heads,
            max_candidates=args.max_candidates,
            forced_slots=args.forced_slots,
            min_distance=args.pair_min_distance,
            residual_cap=args.training_residual_cap,
            pair_mass_cap=args.pair_mass_cap,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    pair_parameters = sum(
        parameter.numel() for module in model.pair_modules for parameter in module.parameters()
    )
    if not 400_000_000 <= total_parameters <= 600_000_000 and not args.smoke:
        raise RuntimeError(f"model has {total_parameters:,} parameters; expected 400M-600M")
    if is_main:
        print("[config] " + json.dumps(vars(args), sort_keys=True), flush=True)
        if LENGTH_GATE_MAX_BASES > 0:
            print(
                "[length-gate-fallback-config] "
                + json.dumps(
                    {
                        "length_gate_max_bases": LENGTH_GATE_MAX_BASES,
                        "content_length": "attention_mask_sum_minus_cls_sep",
                        "long_sequence_pair_nll": False,
                        "long_sequence_retrieval_loss": False,
                        "long_sequence_pair_residual_detached": True,
                        "long_sequence_mlm": True,
                        "validation_losses_ungated": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        print(
            "[model] "
            + json.dumps(
                {
                    "total_parameters": total_parameters,
                    "pair_parameters": pair_parameters,
                    "backbone_parameters": total_parameters - pair_parameters,
                    "configuration": model.checkpoint_config(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        atomic_json(
            output_dir / "run_config.json",
            {
                **vars(args),
                "total_parameters": total_parameters,
                "pair_parameters": pair_parameters,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
    if args.dry_run:
        if is_main:
            print("[done] dry_run", flush=True)
        if distributed:
            dist.destroy_process_group()
        return

    model.to(device)
    if args.compile:
        model = torch.compile(model)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    train_model = model
    if distributed:
        train_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=100,
        )
    adam_kwargs = {"fused": True} if args.fused_adam and device.type == "cuda" else {}
    optimizer = AdamW(parameter_groups(raw_model, args), **adam_kwargs)
    warmup_steps = max(1, round(args.max_steps * args.warmup_ratio))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler_last_epoch = -1
    if args.init_from and args.init_step:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        scheduler_last_epoch = args.init_step - 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_factor, last_epoch=scheduler_last_epoch
    )
    global_step = args.init_step if args.init_from else 0
    apply_qk_learning_rates(optimizer, global_step, args)
    examples_consumed = initial_examples_consumed
    best_score = float("inf")
    stale = 0
    if args.resume_from:
        state = resume_state
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["global_step"])
        best_score = float(state["best_score"])
        stale = int(state["stale_validations"])
        examples_consumed = int(state.get("examples_consumed", 0))
        apply_qk_learning_rates(optimizer, global_step, args)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None and is_main:
            cuda_rng = state["cuda_rng"]
            if isinstance(cuda_rng, (list, tuple)):
                cuda_rng = cuda_rng[0]
            torch.cuda.set_rng_state(cuda_rng, device=device)
        if rank:
            seed_all(args.seed + rank + global_step)
        if is_main:
            print(f"[resume] step={global_step} path={args.resume_from}", flush=True)
    elif args.init_from:
        seed_all(args.seed + rank + global_step)
        if is_main:
            print(
                f"[init] step={global_step} examples={examples_consumed} path={args.init_from}",
                flush=True,
            )
    else:
        seed_all(args.seed + rank)

    def amp_context():
        return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    previous_graph_probe = None

    @torch.inference_mode()
    def evaluate(step: int) -> Dict[str, float]:
        nonlocal previous_graph_probe
        raw_model.eval()
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        current_graph_probe = []
        alignment_records = {
            name: []
            for name in (
                "batch_index",
                "group_key",
                "row",
                "anchor",
                "partner",
                "gain",
                "score",
                "native_overlap",
                "is_rnafold",
            )
        }
        cap = args.training_residual_cap
        for batch_index, batch in enumerate(sequence_valid_loader):
            if batch_index >= args.valid_batches:
                break
            batch = move_batch(batch, device)
            with amp_context():
                _, metrics, diagnostics = compute_objective(
                    raw_model,
                    batch,
                    cap,
                    0.1,
                    0.0,
                    args.retrieval_temperature,
                    args.retrieval_gain_clip,
                    gain_margin_at(step, args),
                    args.retrieval_gain_scale,
                    args.retrieval_support_floor,
                    args.retrieval_support_scale,
                    args.native_low_utilization_threshold,
                    batch_index < min(args.graph_probe_batches, args.valid_batches),
                )
            if "candidate_indices" in diagnostics:
                current_graph_probe.append((
                    diagnostics["candidate_indices"].cpu(),
                    diagnostics["candidate_mask"].cpu(),
                ))
            if "retrieval_teacher_gain" in diagnostics:
                record_count = diagnostics["retrieval_teacher_gain"].numel()
                alignment_records["batch_index"].append(
                    torch.full((record_count,), batch_index, dtype=torch.long)
                )
                for output_name, diagnostic_name in (
                    ("group_key", "retrieval_group_key"),
                    ("row", "retrieval_row"),
                    ("anchor", "retrieval_anchor"),
                    ("partner", "retrieval_partner"),
                    ("gain", "retrieval_teacher_gain"),
                    ("score", "retrieval_logit"),
                    ("native_overlap", "retrieval_native_overlap"),
                    ("is_rnafold", "retrieval_is_rnafold"),
                ):
                    alignment_records[output_name].append(
                        diagnostics[diagnostic_name].cpu()
                    )
            for key, value in metrics.items():
                name = f"sequence_{key}"
                totals[name] = totals.get(name, 0.0) + float(value.detach())
                counts[name] = counts.get(name, 0) + 1
        result = {key: totals[key] / counts[key] for key in totals}
        if alignment_records["gain"]:
            alignment_arrays = {
                name: torch.cat(values).numpy()
                for name, values in alignment_records.items()
            }
            alignment_path = output_dir / f"alignment_step-{step}.npz"
            temporary_alignment_path = output_dir / f"alignment_step-{step}.npz.incomplete"
            with temporary_alignment_path.open("wb") as handle:
                np.savez_compressed(handle, **alignment_arrays)
            os.replace(temporary_alignment_path, alignment_path)
            result["alignment_record_count"] = int(alignment_arrays["gain"].size)
            result["alignment_artifact"] = str(alignment_path)
        if "sequence_mlm_loss" in result:
            result["sequence_mlm_perplexity"] = math.exp(min(20.0, result["sequence_mlm_loss"]))
        result["selection_objective"] = (
            result.get("sequence_mlm_loss", float("inf"))
            + 0.1 * result.get("sequence_pair_nll", float("inf"))
        )
        result["router_cap"] = cap
        result["pair_weight"] = 0.1
        result["retrieval_weight"] = 0.0
        result["rnafold_proposal_probability"] = 0.0
        current_probe_anchor_count = sum(
            int(mask.any(-1).sum()) for _, mask in current_graph_probe
        )
        if previous_graph_probe is None:
            result["native_candidate_retention"] = float("nan")
            result["native_candidate_churn"] = float("nan")
            result["graph_probe_anchor_count"] = current_probe_anchor_count
        else:
            retention, churn, anchor_count = candidate_graph_churn(
                previous_graph_probe, current_graph_probe
            )
            result["native_candidate_retention"] = retention
            result["native_candidate_churn"] = churn
            result["graph_probe_anchor_count"] = anchor_count
        previous_graph_probe = current_graph_probe
        raw_model.train()
        return result

    proposal_probability_ref.value = rnafold_proposal_probability_at(args)
    endpoint_fraction_ref.value = OBJECTIVE_ENDPOINT_FRACTION
    sequence_iterator = iter(sequence_loader)
    optimizer.zero_grad(set_to_none=True)
    window: Dict[str, float] = {}
    window_counts: Dict[str, int] = {}
    window_tokens = 0
    window_started = time.time()
    started = time.time()
    micro_step = 0
    target_step = (
        min(args.max_steps, global_step + args.run_steps)
        if args.run_steps
        else args.max_steps
    )
    stop_reason = "run_steps_complete" if target_step < args.max_steps else "max_steps"
    def _keep_h18(module, inputs):
        hidden = inputs[0]
        if torch.is_tensor(hidden) and hidden.requires_grad:
            hidden.retain_grad()
            module._coverage_h18 = hidden
    raw_model.retriever.register_forward_pre_hook(_keep_h18)
    grad_trace = {"q": [], "k": [], "h18": []}
    schedule_trace = {"margin": [], "q_lr": [], "k_lr": []}
    raw_model.train()
    while global_step < target_step:
        proposal_probability_ref.value = rnafold_proposal_probability_at(args)
        endpoint_fraction_ref.value = OBJECTIVE_ENDPOINT_FRACTION
        batch = next(sequence_iterator)
        examples_consumed += int(batch["input_ids"].shape[0]) * world_size
        batch = move_batch(batch, device)
        cap = residual_cap_at(global_step, args)
        current_pair_weight = pair_weight_at(global_step, args)
        current_retrieval_weight = retrieval_weight_at(global_step, args)
        sync_gradients = (micro_step + 1) % args.gradient_accumulation_steps == 0
        sync_context = (
            train_model.no_sync()
            if distributed and not sync_gradients
            else nullcontext()
        )
        with sync_context:
            with amp_context():
                loss, metrics, _ = compute_objective(
                    train_model,
                    batch,
                    cap,
                    current_pair_weight,
                    current_retrieval_weight,
                    args.retrieval_temperature,
                    args.retrieval_gain_clip,
                    gain_margin_at(global_step, args),
                    args.retrieval_gain_scale,
                    args.retrieval_support_floor,
                    args.retrieval_support_scale,
                    args.native_low_utilization_threshold,
                )
                scaled_loss = loss / args.gradient_accumulation_steps
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            scaled_loss.backward()
        micro_step += 1
        extra_metrics = getattr(raw_model, "graph_freeze_grad_metrics", None)
        if callable(extra_metrics):
            metrics.update(extra_metrics())
        for key, value in metrics.items():
            window[key] = window.get(key, 0.0) + float(value.detach())
            window_counts[key] = window_counts.get(key, 0) + 1
        window_tokens += int(batch["attention_mask"].sum())
        if micro_step % args.gradient_accumulation_steps:
            continue

        def _parameter_grad_norm(parameter) -> float:
            grad = parameter.grad
            if grad is None:
                return 0.0
            return float(grad.detach().float().norm())

        q_grad_norm = _parameter_grad_norm(raw_model.retriever.q_proj.weight)
        k_grad_norm = _parameter_grad_norm(raw_model.retriever.k_proj.weight)
        h18_tensor = getattr(raw_model.retriever, "_coverage_h18", None)
        if h18_tensor is not None and h18_tensor.grad is not None:
            h18_grad_norm = float(h18_tensor.grad.detach().float().norm())
        else:
            h18_grad_norm = float("nan")
        if h18_tensor is not None:
            h18_tensor.grad = None
        raw_model.retriever._coverage_h18 = None
        grad_trace["q"].append(q_grad_norm)
        grad_trace["k"].append(k_grad_norm)
        grad_trace["h18"].append(h18_grad_norm)
        scheduled_step = global_step
        apply_qk_learning_rates(optimizer, scheduled_step, args)
        schedule_trace["margin"].append(gain_margin_at(scheduled_step, args))
        schedule_trace["q_lr"].append(named_group_lr(optimizer, "retrieval_q"))
        schedule_trace["k_lr"].append(named_group_lr(optimizer, "retrieval_k"))
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        apply_qk_learning_rates(optimizer, scheduled_step, args)
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % args.logging_steps == 0:
            elapsed = max(1e-6, time.time() - window_started)
            metric_keys = sorted(window)
            packed = torch.tensor(
                [
                    value
                    for key in metric_keys
                    for value in (window[key], float(window_counts.get(key, 0)))
                ],
                dtype=torch.float64,
                device=device,
            )
            token_count = torch.tensor(float(window_tokens), dtype=torch.float64, device=device)
            peak_memory = torch.tensor(
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0,
                dtype=torch.float64,
                device=device,
            )
            if distributed:
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            event = {
                key: packed[index * 2].item()
                / max(1.0, packed[index * 2 + 1].item())
                for index, key in enumerate(metric_keys)
            }
            event.update(
                {
                    "step": global_step,
                    "lr": scheduler.get_last_lr()[0],
                    "pair_lr": max(scheduler.get_last_lr()),
                    "grad_norm": float(grad_norm),
                    "router_cap": cap,
                    "pair_weight": current_pair_weight,
                    "retrieval_weight": current_retrieval_weight,
                    "rnafold_proposal_probability": proposal_probability_ref.value,
                    "tokens_per_second": token_count.item() / elapsed,
                      "elapsed_sec": time.time() - started,
                      "examples_consumed": examples_consumed,
                    "cuda_peak_allocated_gb": peak_memory.item(),
                    "global_step": global_step,
                    "objective_edges_requested": event.get(
                        "objective_edges_requested_mean", args.edges_per_sequence
                    ),
                    "gain_margin": (
                        sum(schedule_trace["margin"]) / len(schedule_trace["margin"])
                        if schedule_trace["margin"] else gain_margin_at(global_step, args)
                    ),
                    "q_lr": (
                        sum(schedule_trace["q_lr"]) / len(schedule_trace["q_lr"])
                        if schedule_trace["q_lr"] else named_group_lr(optimizer, "retrieval_q")
                    ),
                    "k_lr": (
                        sum(schedule_trace["k_lr"]) / len(schedule_trace["k_lr"])
                        if schedule_trace["k_lr"] else named_group_lr(optimizer, "retrieval_k")
                    ),
                    "q_grad_norm": (
                        sum(grad_trace["q"]) / len(grad_trace["q"]) if grad_trace["q"] else float("nan")
                    ),
                    "k_grad_norm": (
                        sum(grad_trace["k"]) / len(grad_trace["k"]) if grad_trace["k"] else float("nan")
                    ),
                    "h18_retrieval_grad_norm": (
                        sum(value for value in grad_trace["h18"] if math.isfinite(value))
                        / max(1, sum(math.isfinite(value) for value in grad_trace["h18"]))
                        if any(math.isfinite(value) for value in grad_trace["h18"])
                        else float("nan")
                    ),
                    "pair_loss": None,
                    "total_loss": None,
                }
            )
            if "pair_nll" in event:
                event["pair_loss"] = event["pair_nll"]
            if "loss" in event:
                event["total_loss"] = event["loss"]
            grad_trace = {"q": [], "k": [], "h18": []}
            schedule_trace = {"margin": [], "q_lr": [], "k_lr": []}
            if is_main:
                print("[train] " + json.dumps(event, sort_keys=True, allow_nan=True), flush=True)
                append_event(metrics_path, "train", event)
            window = {}
            window_counts = {}
            window_tokens = 0
            window_started = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            violations = []
            if global_step >= args.stability_guard_start_step:
                if (
                    args.max_train_mlm_loss
                    and event.get("mlm_loss", 0.0) > args.max_train_mlm_loss
                ):
                    violations.append("mlm_loss")
                if (
                    args.max_train_pair_state_norm
                    and event.get("pair_state_norm", 0.0)
                    > args.max_train_pair_state_norm
                ):
                    violations.append("pair_state_norm")
            if violations:
                stop_reason = "stability_guard:" + ",".join(violations)
                guard_event = {
                    "step": global_step,
                    "reason": stop_reason,
                    "mlm_loss": event.get("mlm_loss"),
                    "pair_state_norm": event.get("pair_state_norm"),
                }
                if is_main:
                    print("[stability-abort] " + json.dumps(guard_event, sort_keys=True), flush=True)
                    append_event(metrics_path, "stability_abort", guard_event)
                break

        if global_step % args.valid_steps == 0:
            validation_stop = False
            if is_main:
                validation = evaluate(global_step)
                validation["step"] = global_step
                print("[valid] " + json.dumps(validation, sort_keys=True, allow_nan=True), flush=True)
                append_event(metrics_path, "valid", validation)
                score = validation["selection_objective"]
                # Intermediate validations are diagnostic. Numbered checkpoints
                # and the best-directory copy are written only on save steps.
                if global_step % args.save_steps == 0:
                    if score < best_score - args.early_stopping_min_delta:
                        best_score = score
                        stale = 0
                        best_dir = output_dir / "best"
                        temporary = output_dir / "best.incomplete"
                        if temporary.exists():
                            shutil.rmtree(temporary)
                        raw_model.save_pretrained(str(temporary))
                        tokenizer.save_pretrained(str(temporary))
                        if best_dir.exists():
                            shutil.rmtree(best_dir)
                        os.replace(temporary, best_dir)
                    else:
                        stale += 1
                if bprna_loaders:
                    raw_model.eval()
                    bprna_diagnostics = run_bprna_pair_diagnostics(
                        raw_model,
                        bprna_loaders,
                        device,
                        args.training_residual_cap,
                        amp_context,
                    )
                    bprna_diagnostics["step"] = global_step
                    print(
                        "[bprna-diagnostic] "
                        + json.dumps(bprna_diagnostics, sort_keys=True, allow_nan=True),
                        flush=True,
                    )
                    append_event(metrics_path, "bprna_diagnostic", bprna_diagnostics)
                    raw_model.train()
                membership_paths = [
                    value.strip()
                    for value in getattr(args, "native_membership_paths", "").split(",")
                    if value.strip()
                ]
                if membership_paths:
                    from model.training.pretrain.native_membership_eval import (
                        evaluate_native_membership,
                    )

                    raw_model.eval()
                    membership = evaluate_native_membership(
                        raw_model,
                        membership_paths,
                        tokenizer,
                        device,
                        amp_context,
                        args.pair_min_distance,
                        args.training_residual_cap,
                        output_dir / f"membership_step-{global_step}.npz",
                        batch_size=max(1, min(4, args.batch_size)),
                    )
                    membership["step"] = global_step
                    if bprna_diagnostics if bprna_loaders else None:
                        membership["forced_auprc"] = bprna_diagnostics.get("forced_auprc")
                    print(
                        "[native-membership] "
                        + json.dumps(membership, sort_keys=True, allow_nan=True),
                        flush=True,
                    )
                    append_event(metrics_path, "native_membership", membership)
                    raw_model.train()
                downstream_task_names = downstream_tasks_for_step(args, global_step)
                if downstream_task_names:
                    optimizer.zero_grad(set_to_none=True)
                    downstream = run_downstream_monitor(
                        raw_model,
                        tokenizer,
                        device,
                        args,
                        global_step,
                        task_names=downstream_task_names,
                    )
                    append_event(
                        metrics_path,
                        "downstream",
                        {"step": global_step, "summary_path": str(output_dir / "downstream_monitor" / f"step-{global_step}" / "metrics_summary.json")},
                    )
                    del downstream
                else:
                    print(
                        "[downstream-skip] "
                        + json.dumps(
                            {
                                "step": global_step,
                                "full_interval": args.downstream_monitor_steps
                                or args.valid_steps,
                                "secondary_interval": args.downstream_secondary_steps,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                schedule_complete = warmup_complete(global_step, args)
                validation_stop = bool(
                    schedule_complete
                    and args.early_stopping_patience
                    and stale >= args.early_stopping_patience
                )
            if distributed:
                validation_state = [best_score, stale, validation_stop]
                dist.broadcast_object_list(validation_state, src=0)
                best_score = float(validation_state[0])
                stale = int(validation_state[1])
                validation_stop = bool(validation_state[2])
            if validation_stop:
                stop_reason = "validation_converged"
                break
            window_started = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        if global_step % args.save_steps == 0:
            if is_main:
                checkpoint_root = (
                    _checkpoint_root(args)
                    if getattr(args, "checkpoint_weights_only", False)
                    else Path(args.output_dir)
                )
                if shutil.disk_usage(checkpoint_root).free < 8 * 1024**3:
                    stop_reason = "disk_guard"
                    print(
                        "[disk-guard] "
                        + json.dumps({"step": global_step, "free_bytes": shutil.disk_usage(checkpoint_root).free, "root": str(checkpoint_root)}),
                        flush=True,
                    )
                else:
                    save_checkpoint(
                        raw_model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        args,
                        global_step,
                        best_score,
                        stale,
                        examples_consumed,
                    )
            if distributed:
                stop_state = [stop_reason]
                dist.broadcast_object_list(stop_state, src=0)
                stop_reason = str(stop_state[0])
                dist.barrier()
            if stop_reason == "disk_guard":
                break

    if is_main:
        final_dir = output_dir / "final"
        raw_model.set_residual_cap(args.training_residual_cap)
        raw_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        summary = {
            "step": global_step,
            "stop_reason": stop_reason,
            "best_validation_objective": best_score,
            "final_inference_residual_cap": args.training_residual_cap,
            "final_rnafold_proposal_probability": rnafold_proposal_probability_at(args),
            "examples_consumed": examples_consumed,
            "elapsed_sec": time.time() - started,
            "world_size": world_size,
        }
        atomic_json(output_dir / "training_summary.json", summary)
        print("[done] " + json.dumps(summary, sort_keys=True), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
