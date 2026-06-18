from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class CharTokenizer:
    """A tiny character tokenizer for readable language-model examples."""

    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        chars = sorted(set(text))
        stoi = {char: index for index, char in enumerate(chars)}
        itos = {index: char for char, index in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as error:
            char = error.args[0]
            raise ValueError(f"Character {char!r} is not in the tokenizer.") from error

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)

    def to_dict(self) -> dict[str, object]:
        return {"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CharTokenizer:
        stoi = cast(dict[str, Any], data["stoi"])
        raw_itos = cast(dict[str, Any], data["itos"])
        itos = {int(key): str(value) for key, value in raw_itos.items()}
        return cls(stoi={str(k): int(v) for k, v in stoi.items()}, itos=itos)
