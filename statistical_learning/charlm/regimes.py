"""Fixed word vocabularies ("regimes") for the character-level task.

Each regime is a small set of equal-length words. Because the words in a regime
are all the same length they are automatically prefix-free, so a separator-free
character stream segments uniquely by a fixed word length.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Regime:
    name: str
    words: tuple[str, ...]

    @property
    def word_len(self) -> int:
        lengths = {len(w) for w in self.words}
        if len(lengths) != 1:
            raise ValueError(
                f"regime {self.name!r} has mixed word lengths {sorted(lengths)}; "
                "fixed-length segmentation requires a single length"
            )
        return lengths.pop()


_REGIME_LIST = [
    Regime(
        "ten_word_overlap",
        ("cat", "hat", "mat", "rat", "met", "pet", "net", "ate", "eat", "tea"),
    ),
    Regime(
        "ten_four_letter_overlap",
        ("bake", "cake", "lake", "make", "bank", "tank", "cane", "cant", "late", "mate"),
    ),
    Regime(
        "six_word_overlap",
        ("cat", "hat", "mat", "con", "cob", "cot"),
    ),
    Regime(
        "six_word_overlap_sin",
        ("sin", "six", "sir", "cat", "hat", "mat"),
    ),
    Regime(
        "twelve_word_overlap",
        ("ban", "rot", "cat", "hat", "mat", "con", "cob", "cot", "son", "din", "fun", "bun"),
    ),
    Regime(
        "sixteen_word_overlap",
        (
            "cat", "hat", "mat", "rat", "met", "pet", "net", "can", "ban", "pan",
            "car", "bar", "tar", "ant", "and", "bed", "bet", "tea", "oil",
        ),
    ),
]

REGIMES: dict[str, Regime] = {r.name: r for r in _REGIME_LIST}


def get_regime(name: str) -> Regime:
    if name not in REGIMES:
        raise KeyError(
            f"unknown regime {name!r}; choices: {', '.join(REGIMES)}"
        )
    return REGIMES[name]
