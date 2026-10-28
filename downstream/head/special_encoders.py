"""Encoders whose official stacks are not plain PyTorch BERT/fairseq ESM.

RNAErnie
    PaddleNLP ErnieModel. The released file is ``model_state.pdparams``.
    Official extraction is ``ErnieModel.from_pretrained(...)(input_ids)[0]``,
    then the nucleotide states between ``[CLS]`` and ``[SEP]``. Paddle is not
    installed here, so the same post-norm Ernie stack is loaded in PyTorch.
    1-mer vocab: [CLS]=2, [SEP]=3, [PAD]=0, A/T/C/G = 35/36/37/38. U is T.
    Max nucleotides: 510 (position table is 513, official max_seq_len is 512).

RNABERT
    ``MLM_SFP.py --data_embedding``. 1-mer ids are PAD=0, MASK=1, A=2, U=3,
    G=4, C=5. There is no CLS. Positions are capped at 439 because
    ``max_position_embeddings`` is 440. The checkpoint is
    ``DataParallel(BertForMaskedLM)``; only ``module.bert.*`` is the encoder.
    Non-ACGU bases use the MASK id, which is the only non-base token.

RNA-MSM
    ``RNA_MSM_Inference.py`` reads ``*.a2m_msa2`` and returns layer-10 row-0
    embeddings of shape ``(L, 768)``. This benchmark has no homolog MSA, so
    each sequence is passed as a depth-1 MSA. That is a defined forward of the
    same MSA Transformer, not the homolog-search setting in the paper.

HydraRNA
    ``examples/extract_HydraAttRNA12_5UTRMRL.py``. fairseq task
    ``masked_lm_span``, dictionary ``HydraRNA-main/dict``, then
    ``model.encoder.extract_features``. Tokens are ``<s>`` + bases + ``</s>``.
    U is written as T. The architecture imports flash-attn and mamba-ssm.

BiRNA-BERT
    Local Mosaic BERT (``bert_layers.BertForMaskedLM``). The MLM head is
    replaced by ``Identity`` after the checkpoint loads, as in the repository
    README, so ``logits`` are the token states. Nucleotide mode is a
    space-separated string. ``from_pretrained`` is not used: current
    transformers builds the module on the meta device, and this code computes
    the ALiBi tensor during ``__init__``.

RNAret
    Official ``pretrain_1mer.pth`` RetNet backbone from ``RNAret-main``.
    Embeddings are ``aux['inner_states'][-1]``. With k=1 the token stream is
    one id per nucleotide (no CLS), so nucleotide states are taken as-is.
    Requires fairscale / torchscale; ``timm`` is stubbed if unavailable.

Uni-RNA
    HuggingFace-compatible ``unirna_tf`` (``AutoModel`` / ``UniRNAModels``).
    Released dirs: ``unirna_L8`` / ``L12`` / ``L16``. Tokens are
    ``<cls>`` + nucleotides + ``<eos>``. Sequences must be uppercase; T is
    mapped to U to match the official RNA examples. Max nucleotides follow
    ``max_position_embeddings - 2`` (1024 for L16).
"""

from __future__ import annotations

import importlib.util
import math
import pickle
import sys
import types
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from datasets import clean_seq

REFERENCE_ROOT = Path("/root/RIPPLE/references")


def strip_special(representation: torch.Tensor, lengths: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    width = max(lengths) if lengths else 1
    hidden = representation.shape[-1]
    embeddings = representation.new_zeros(len(lengths), width, hidden)
    padding = torch.ones(len(lengths), width, dtype=torch.bool, device=representation.device)
    for row, length in enumerate(lengths):
        if length <= 0:
            continue
        embeddings[row, :length] = representation[row, 1 : 1 + length]
        padding[row, :length] = False
    return embeddings, padding


def nucleotide_states(representation: torch.Tensor, lengths: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep the first ``length`` positions. Used when the model has no CLS."""
    width = max(lengths) if lengths else 1
    hidden = representation.shape[-1]
    embeddings = representation.new_zeros(len(lengths), width, hidden)
    padding = torch.ones(len(lengths), width, dtype=torch.bool, device=representation.device)
    for row, length in enumerate(lengths):
        if length <= 0:
            continue
        embeddings[row, :length] = representation[row, :length]
        padding[row, :length] = False
    return embeddings, padding


class _ErnieAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(768, 768)
        self.k_proj = nn.Linear(768, 768)
        self.v_proj = nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, 768)
        self.heads = 12
        self.head_dim = 64

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = self._split(self.q_proj(hidden))
        key = self._split(self.k_proj(hidden))
        value = self._split(self.v_proj(hidden))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + mask
        context = torch.matmul(torch.softmax(scores, dim=-1), value)
        return self.out_proj(self._merge(context))

    def _split(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, _channels = hidden.shape
        return hidden.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def _merge(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, _heads, length, _dim = hidden.shape
        return hidden.transpose(1, 2).contiguous().view(batch, length, self.heads * self.head_dim)


class _ErnieLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _ErnieAttention()
        self.linear1 = nn.Linear(768, 3072)
        self.linear2 = nn.Linear(3072, 768)
        self.norm1 = nn.LayerNorm(768, eps=1e-5)
        self.norm2 = nn.LayerNorm(768, eps=1e-5)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(hidden + self.self_attn(hidden, mask))
        feed = self.linear2(F.relu(self.linear1(hidden)))
        return self.norm2(hidden + feed)


class _RNAErnieBackbone(nn.Module):
    """PaddleNLP ErnieModel body. Paddle linear weights are stored as (in, out)."""

    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(39, 768, padding_idx=0)
        self.position_embeddings = nn.Embedding(513, 768)
        self.token_type_embeddings = nn.Embedding(2, 768)
        self.layer_norm = nn.LayerNorm(768, eps=1e-5)
        self.layers = nn.ModuleList(_ErnieLayer() for _ in range(12))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        token_type = torch.zeros_like(tokens)
        hidden = self.layer_norm(
            self.word_embeddings(tokens)
            + self.position_embeddings(positions)
            + self.token_type_embeddings(token_type)
        )
        mask = (tokens == 0).unsqueeze(1).unsqueeze(2).to(hidden.dtype) * -1e4
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return hidden

    def load_paddle(self, path: str) -> None:
        with open(path, "rb") as handle:
            state = pickle.load(handle)

        def ndarray(key: str) -> torch.Tensor:
            value = state[key]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{key} is {type(value).__name__}, expected ndarray")
            return torch.from_numpy(np.ascontiguousarray(value)).float()

        def copy_linear(module: nn.Linear, prefix: str) -> None:
            weight = ndarray(prefix + ".weight")
            module.weight.data.copy_(weight.t())
            module.bias.data.copy_(ndarray(prefix + ".bias"))

        def copy_norm(module: nn.LayerNorm, prefix: str) -> None:
            module.weight.data.copy_(ndarray(prefix + ".weight"))
            module.bias.data.copy_(ndarray(prefix + ".bias"))

        with torch.no_grad():
            self.word_embeddings.weight.copy_(ndarray("ernie.embeddings.word_embeddings.weight"))
            self.position_embeddings.weight.copy_(ndarray("ernie.embeddings.position_embeddings.weight"))
            self.token_type_embeddings.weight.copy_(ndarray("ernie.embeddings.token_type_embeddings.weight"))
            copy_norm(self.layer_norm, "ernie.embeddings.layer_norm")
            for index, layer in enumerate(self.layers):
                prefix = f"ernie.encoder.layers.{index}"
                copy_linear(layer.self_attn.q_proj, prefix + ".self_attn.q_proj")
                copy_linear(layer.self_attn.k_proj, prefix + ".self_attn.k_proj")
                copy_linear(layer.self_attn.v_proj, prefix + ".self_attn.v_proj")
                copy_linear(layer.self_attn.out_proj, prefix + ".self_attn.out_proj")
                copy_linear(layer.linear1, prefix + ".linear1")
                copy_linear(layer.linear2, prefix + ".linear2")
                copy_norm(layer.norm1, prefix + ".norm1")
                copy_norm(layer.norm2, prefix + ".norm2")


class RNAErnieEncoder(nn.Module):
    name = "rnaernie"
    _BASE = {"A": 35, "T": 36, "C": 37, "G": 38, "U": 36}

    def __init__(self, weights: str, device: str):
        super().__init__()
        model = _RNAErnieBackbone()
        model.load_paddle(weights)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model.to(device)
        self.hidden_size = 768
        self.max_raw = 510
        self.device = device

    def embed(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = [clean_seq(seq)[: self.max_raw] for seq in sequences]
        lengths = [len(seq) for seq in sequences]
        width = (max(lengths) if lengths else 0) + 2
        tokens = torch.zeros(len(sequences), width, dtype=torch.long, device=self.device)
        for row, seq in enumerate(sequences):
            ids = [2] + [self._BASE.get(base, 1) for base in seq] + [3]
            tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            hidden = self.model(tokens)
        embeddings, padding = strip_special(hidden, lengths)
        return embeddings.detach(), padding


class RNABertEncoder(nn.Module):
    name = "rnabert"
    _BASE = {"A": 2, "U": 3, "G": 4, "C": 5}

    def __init__(self, weights: str, device: str):
        super().__init__()
        package = REFERENCE_ROOT / "RNABERT-master"
        sys.path.insert(0, str(package))
        _install_attrdict()
        from utils.bert import BertModel, get_config

        config = get_config(str(package / "RNA_bert_config.json"))
        config.hidden_size = config.num_attention_heads * config.multiple
        model = BertModel(config)
        raw = torch.load(weights, map_location="cpu")
        state = {key[len("module.bert.") :]: value for key, value in raw.items() if key.startswith("module.bert.")}
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model.to(device)
        self.hidden_size = int(config.hidden_size)
        self.max_raw = int(config.max_position_embeddings) - 1
        self.width = int(config.max_position_embeddings)
        self.device = device

    def embed(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = [clean_seq(seq)[: self.max_raw] for seq in sequences]
        lengths = [len(seq) for seq in sequences]
        tokens = torch.zeros(len(sequences), self.width, dtype=torch.long, device=self.device)
        for row, seq in enumerate(sequences):
            ids = [self._BASE.get(base, 1) for base in seq]
            if ids:
                tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            encoded, _pooled = self.model(tokens, output_all_encoded_layers=False)
        embeddings, padding = nucleotide_states(encoded.float(), lengths)
        return embeddings.detach(), padding


class RNAMSMEncoder(nn.Module):
    name = "rnamsm"

    def __init__(self, weights: str, device: str):
        super().__init__()
        root = REFERENCE_ROOT / "RNA-MSM-master"
        _stub_rnamsm_imports()
        sys.path.insert(0, str(root))
        import __main__
        from model import MSATransformer, OptimizerConfig, PKMLayerConfig, TransformerConfig, TransformerLayerConfig
        from utils.tokenization import Vocab
        import msm

        for cls in (OptimizerConfig, PKMLayerConfig, TransformerConfig, TransformerLayerConfig, MSATransformer):
            setattr(__main__, cls.__name__, cls)

        alphabet = msm.data.Alphabet.from_architecture("rna language")
        vocab = Vocab.from_esm_alphabet(alphabet)
        model = MSATransformer(
            vocab,
            embed_dim=768,
            num_attention_heads=12,
            num_layers=10,
            embed_positions_msa=True,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            max_tokens_per_msa=16384,
            max_seqlen=1024,
        )
        checkpoint = _load_checkpoint(weights)
        state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model.to(device)
        self.vocab = vocab
        self.hidden_size = 768
        self.max_raw = 1023
        self.device = device
        self._allowed = set("AGCUXN")

    def embed(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = ["".join(base if base in self._allowed else "N" for base in clean_seq(seq)[: self.max_raw]) for seq in sequences]
        lengths = [len(seq) for seq in sequences]
        if not sequences:
            empty = torch.zeros(0, 1, self.hidden_size, device=self.device)
            return empty, torch.ones(0, 1, dtype=torch.bool, device=self.device)
        encoded = self.vocab.encode_batched_sequences(sequences)
        tokens = torch.from_numpy(encoded).long().unsqueeze(1).to(self.device)
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[10], need_head_weights=False)
        hidden = results["representations"][10][:, 0].float()
        embeddings, padding = strip_special(hidden, lengths)
        return embeddings.detach(), padding


class HydraRNAEncoder(nn.Module):
    name = "hydrarna"

    def __init__(self, weights: str, device: str):
        super().__init__()
        root = REFERENCE_ROOT / "HydraRNA-main"
        fairseq_root = str(root / "fairseq")
        sys.path.insert(0, fairseq_root)
        try:
            from fairseq import checkpoint_utils, options, tasks
            import fairseq
        except ImportError as exc:
            raise ImportError(
                "HydraRNA needs its own fairseq plus flash-attn and mamba-ssm. "
                "Official environment: torch 2.3.1+cu118, mamba-ssm[causal-conv1d]==2.2.2, flash-attn."
            ) from exc
        if "HydraRNA" not in str(fairseq.__file__):
            raise ImportError(f"Imported {fairseq.__file__} instead of HydraRNA fairseq. Put {fairseq_root} first on PYTHONPATH.")
        parser = options.get_generation_parser(default_task="masked_lm_span")
        args = options.parse_args_and_arch(parser, [str(root / "dict")])
        task = tasks.setup_task(args)
        models, _model_args = checkpoint_utils.load_model_ensemble([weights], task=task)
        model = models[0]
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model.to(device).half()
        self.task = task
        self.hidden_size = 1024
        self.max_raw = 10240
        self.device = device

    def embed(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        from fairseq import data

        sequences = [clean_seq(seq).replace("U", "T")[: self.max_raw] for seq in sequences]
        lengths = [len(seq) for seq in sequences]
        encoded = []
        for seq in sequences:
            text = "<s> " + " ".join(seq)
            encoded.append(self.task.source_dictionary.encode_line(text, add_if_not_exist=False, append_eos=True))
        batch = data.monolingual_dataset.collate(
            samples=[{"id": -1, "source": tokens, "target": tokens} for tokens in encoded],
            pad_idx=self.task.source_dictionary.pad(),
            eos_idx=self.task.source_dictionary.eos(),
        )
        source = batch["net_input"]["src_tokens"].to(self.device)
        with torch.no_grad():
            features, _extra = self.model.encoder.extract_features(source)
        embeddings, padding = strip_special(features.float(), lengths)
        return embeddings.detach(), padding


def _load_birna_package(model_dir: Path):
    """Import the local Mosaic BERT files as a package so relative imports work."""
    pkg_name = "birna_local"
    if pkg_name not in sys.modules:
        package = types.ModuleType(pkg_name)
        package.__path__ = [str(model_dir)]
        sys.modules[pkg_name] = package
    loaded = {}
    for name in ("bert_padding", "configuration_bert", "bert_layers"):
        module_name = f"{pkg_name}.{name}"
        if module_name in sys.modules:
            loaded[name] = sys.modules[module_name]
            continue
        spec = importlib.util.spec_from_file_location(module_name, model_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["bert_layers"], loaded["configuration_bert"]


class BiRNABERTEncoder(nn.Module):
    name = "birnabert"

    def __init__(self, weights: str, device: str):
        super().__init__()
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("BiRNA-BERT needs transformers. pip install transformers tokenizers") from exc
        root = Path(weights)
        model_dir = root / "model" if (root / "model" / "config.json").is_file() else root
        tokenizer_dir = root / "tokenizer" if (root / "tokenizer").is_dir() else model_dir
        weight_file = model_dir / "pytorch_model.bin"
        if not (model_dir / "config.json").is_file() or not weight_file.is_file():
            raise FileNotFoundError(f"BiRNA-BERT config or pytorch_model.bin is missing in {root}")
        layers, config_module = _load_birna_package(model_dir)
        config = config_module.BertConfig.from_pretrained(str(model_dir), local_files_only=True)
        if hasattr(config, "alibi_starting_size"):
            config.alibi_starting_size = max(int(config.alibi_starting_size), 2048)
        model = layers.BertForMaskedLM(config)
        state = torch.load(weight_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.cls = nn.Identity()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        self.hidden_size = int(getattr(config, "hidden_size", 768))
        self.max_raw = 2046
        self.tokenizer.model_max_length = self.max_raw + 2
        self.device = device

    def embed(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = [clean_seq(seq).replace("U", "T")[: self.max_raw] for seq in sequences]
        lengths = [len(seq) for seq in sequences]
        texts = [" ".join(seq) if seq else "N" for seq in sequences]
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_raw + 2)
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.no_grad():
            output = self.model(**tokens)
        hidden = output.logits if hasattr(output, "logits") else output[0]
        if hidden.shape[-1] != self.hidden_size and hasattr(self.model, "bert"):
            hidden = self.model.bert(**tokens).last_hidden_state
        if any(hidden.shape[1] < length + 2 for length in lengths):
            raise RuntimeError(f"BiRNA-BERT tokenizer merged nucleotides: token width {hidden.shape[1]}, lengths {lengths}")
        embeddings, padding = strip_special(hidden.float(), lengths)
        return embeddings.detach(), padding


def _stub_timm_drop_path() -> None:
    """torchscale imports timm.drop_path; avoid pulling torchvision."""
    if "timm.models.layers" in sys.modules:
        return
    try:
        from timm.models.layers import drop_path  # noqa: F401

        return
    except Exception:
        pass
    for name in ("timm", "timm.models", "timm.models.layers"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    layers = sys.modules["timm.models.layers"]

    def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        return x

    layers.drop_path = drop_path


class RNAretEncoder(nn.Module):
    """RNAret frozen RetNet encoder, 1-mer tokenization."""

    name = "rnaret"

    def __init__(self, weights: str, device: str, k: int = 1):
        super().__init__()
        if k != 1:
            raise ValueError("Benchmark RNAret encoder currently supports k=1 only")
        _stub_timm_drop_path()
        package = REFERENCE_ROOT / "RNAret-main"
        if not package.is_dir():
            raise FileNotFoundError(f"Missing RNAret source at {package}")
        sys.path.insert(0, str(package))
        from retnet import RetNet, RetNetConfig

        weight_path = Path(weights)
        if weight_path.is_dir():
            weight_path = weight_path / "model" / "pretrain" / "pretrain_1mer.pth"
            if not weight_path.is_file():
                weight_path = Path(weights) / "pretrain_1mer.pth"
        if not weight_path.is_file():
            raise FileNotFoundError(f"Missing RNAret weights: {weights}")
        config = RetNetConfig(
            vocab_size=4**k + 6,
            retnet_embed_dim=384,
            retnet_value_embed_dim=512,
            retnet_ffn_embed_dim=512,
            retnet_layers=8,
            retnet_retention_heads=4,
            dropout=0.0,
            activation_dropout=0.0,
            max_target_positions=2048,
        )
        model = RetNet(config)
        state = torch.load(str(weight_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model.to(device)
        self.hidden_size = 384
        self.k = k
        self.max_raw = 2000
        self.device = device
        import itertools

        self._kmer_to_index = {
            "".join(bases): idx + 6 for idx, bases in enumerate(itertools.product("ATCG", repeat=k))
        }

    def _tokenize(self, seq: str) -> np.ndarray:
        """Official fasta_tokenizer layout. For k=1 this is one id per nucleotide."""
        length = len(seq)
        if length <= 0:
            return np.zeros(0, dtype=np.int64)
        kmers = [seq[i : i + self.k] for i in range(length - self.k + 1)]
        indices = np.array([self._kmer_to_index.get(kmer, 2) for kmer in kmers], dtype=np.int64)
        if self.k == 1:
            return indices
        tokens = np.zeros(self.k // 2 + len(indices) + max(self.k - 1 - self.k // 2, 0), dtype=np.int64)
        tokens[: self.k // 2] = 1
        tokens[self.k // 2 : self.k // 2 + len(indices)] = indices
        tokens[self.k // 2 + len(indices) : self.k + len(indices) - 1] = 1
        return tokens

    def embed(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = [clean_seq(seq).replace("U", "T")[: self.max_raw] for seq in sequences]
        lengths = [len(seq) for seq in sequences]
        token_rows = [self._tokenize(seq) for seq in sequences]
        width = max((len(row) for row in token_rows), default=1)
        batch = torch.zeros(len(sequences), width, dtype=torch.long, device=self.device)
        for row, tokens in enumerate(token_rows):
            if len(tokens):
                batch[row, : len(tokens)] = torch.as_tensor(tokens, device=self.device)
        with torch.no_grad():
            _logits, aux = self.model(batch)
        hidden = aux["inner_states"][-1]
        embeddings, padding = nucleotide_states(hidden.float(), lengths)
        return embeddings.detach(), padding


def _load_checkpoint(path: str):
    """Lightning checkpoints pickled classes that lived in ``__main__``."""
    import __main__

    for _attempt in range(40):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except AttributeError as exc:
            text = str(exc)
            if "Can't get attribute" not in text or "__main__" not in text:
                raise
            name = text.split("attribute '", 1)[1].split("'", 1)[0]
            if not name.isidentifier() or hasattr(__main__, name):
                raise
            setattr(__main__, name, type(name, (), {}))
    raise RuntimeError(f"Could not unpickle {path}")


class UniRNAEncoder(nn.Module):
    """Uni-RNA frozen encoder via unirna_tf AutoModel."""

    name = "unirna"

    def __init__(self, weights: str, device: str):
        super().__init__()
        package = REFERENCE_ROOT / "Uni-RNA-main"
        if not package.is_dir():
            raise FileNotFoundError(f"Missing Uni-RNA source at {package}")
        sys.path.insert(0, str(package))
        import unirna_tf  # noqa: F401  registers Auto* classes
        from transformers import AutoModel, AutoTokenizer

        weight_path = Path(weights)
        if weight_path.is_file():
            weight_path = weight_path.parent
        if not (weight_path / "config.json").is_file():
            # Accept Uni-RNA root containing extracted/weights/unirna_L16
            candidates = [
                weight_path / "unirna_L16",
                weight_path / "extracted" / "weights" / "unirna_L16",
                weight_path / "weights" / "unirna_L16",
            ]
            for candidate in candidates:
                if (candidate / "config.json").is_file():
                    weight_path = candidate
                    break
        if not (weight_path / "config.json").is_file():
            raise FileNotFoundError(f"Missing Uni-RNA HF directory under {weights}")
        tokenizer = AutoTokenizer.from_pretrained(str(weight_path), local_files_only=True)
        model = AutoModel.from_pretrained(str(weight_path), local_files_only=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.hidden_size = int(getattr(model.config, "hidden_size", 1024))
        max_pos = int(getattr(model.config, "max_position_embeddings", 1026))
        self.max_raw = max(1, max_pos - 2)
        self.tokenizer.model_max_length = max_pos
        self.device = device

    def embed(self, sequences: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Official README: uppercase only; examples use U (vocab has both T and U).
        sequences = [clean_seq(seq).replace("T", "U")[: self.max_raw] for seq in sequences]
        lengths = [len(seq) for seq in sequences]
        texts = [seq if seq else "N" for seq in sequences]
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_raw + 2,
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.no_grad():
            output = self.model(**tokens)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        embeddings, padding = strip_special(hidden.float(), lengths)
        return embeddings.detach(), padding


def build_special_encoder(name: str, weights: str, device: str) -> nn.Module:
    if name == "rnaernie":
        return RNAErnieEncoder(weights, device)
    if name == "rnabert":
        return RNABertEncoder(weights, device)
    if name == "rnamsm":
        return RNAMSMEncoder(weights, device)
    if name == "hydrarna":
        return HydraRNAEncoder(weights, device)
    if name == "birnabert":
        return BiRNABERTEncoder(weights, device)
    if name == "rnaret":
        return RNAretEncoder(weights, device)
    if name == "unirna":
        return UniRNAEncoder(weights, device)
    raise ValueError(f"Unknown special encoder {name}")


def _install_attrdict() -> None:
    if "attrdict" in sys.modules:
        return
    module = types.ModuleType("attrdict")

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    module.AttrDict = AttrDict
    sys.modules["attrdict"] = module


class _LazyModule(types.ModuleType):
    def __getattr__(self, item: str):
        if item.startswith("__"):
            raise AttributeError(item)
        full = f"{self.__name__}.{item}"
        module = sys.modules.get(full)
        if not isinstance(module, _LazyModule):
            module = _LazyModule(full)
            sys.modules[full] = module
        setattr(self, item, module)
        return module


def _stub_rnamsm_imports() -> None:
    if "pytorch_lightning" not in sys.modules:
        lightning = types.ModuleType("pytorch_lightning")

        class LightningModule(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        lightning.LightningModule = LightningModule
        sys.modules["pytorch_lightning"] = lightning
    if "numba" not in sys.modules:
        numba = types.ModuleType("numba")

        def njit(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def wrap(function):
                return function

            return wrap

        numba.njit = njit
        sys.modules["numba"] = numba
    for name in ("tape", "transformers"):
        if name in sys.modules and not isinstance(sys.modules[name], _LazyModule):
            continue
        module = _LazyModule(name)
        sys.modules[name] = module
