"""
RNA tokenizer: A, U, C, G, N (non-AUCG -> N) + [PAD], [MASK], [CLS], [SEP].
"""
import os
from typing import List, Optional, Union

# Vocab: [PAD]=0, [CLS]=1, [SEP]=2, [MASK]=3, A=4, U=5, C=6, G=7, N=8
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
NUCLEOTIDES = ["A", "U", "C", "G", "N"]
VOCAB = SPECIAL_TOKENS + NUCLEOTIDES
VOCAB_SIZE = len(VOCAB)

TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
ID_TO_TOKEN = {i: t for t, i in TOKEN_TO_ID.items()}

PAD_ID = TOKEN_TO_ID["[PAD]"]
CLS_ID = TOKEN_TO_ID["[CLS]"]
SEP_ID = TOKEN_TO_ID["[SEP]"]
MASK_ID = TOKEN_TO_ID["[MASK]"]


class RNATokenizer:
    """
    Tokenizer for RNA sequences: A, U, C, G, N (any other base -> N).
    Special tokens: [PAD], [MASK], [CLS], [SEP].
    """

    def __init__(
        self,
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
    ):
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.vocab_size = VOCAB_SIZE

    def _base_to_id(self, c: str) -> int:
        c = c.upper()
        if c in ("A", "U", "C", "G"):
            return TOKEN_TO_ID[c]
        return TOKEN_TO_ID["N"]

    def encode(
        self,
        sequence: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], "torch.Tensor"]:
        """
        Encode RNA sequence to token ids.
        sequence: raw string of A/U/C/G (and others -> N).
        """
        ids = [self._base_to_id(c) for c in sequence]
        if add_special_tokens:
            ids = [CLS_ID] + ids + [SEP_ID]
        if return_tensors == "pt":
            import torch
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def decode(
        self,
        ids: Union[List[int], "torch.Tensor"],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token ids to RNA sequence string."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        chars = []
        for i in ids:
            if i in (PAD_ID, CLS_ID, SEP_ID, MASK_ID):
                if skip_special_tokens:
                    continue
                chars.append(ID_TO_TOKEN.get(i, "?"))
            else:
                chars.append(ID_TO_TOKEN.get(i, "N"))
        return "".join(chars)

    def tokenize(self, sequence: str) -> List[str]:
        """Tokenize to list of token strings (one char per token)."""
        return [c.upper() if c.upper() in ("A", "U", "C", "G") else "N" for c in sequence]

    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    @property
    def cls_token_id(self) -> int:
        return CLS_ID

    @property
    def sep_token_id(self) -> int:
        return SEP_ID

    @property
    def mask_token_id(self) -> int:
        return MASK_ID

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "vocab.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(VOCAB))

    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs) -> "RNATokenizer":
        return cls(**kwargs)
