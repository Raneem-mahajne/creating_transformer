"""Vocabulary validation for the statistical-learning task.

A vocabulary is just a set of words. Because sequences are raw concatenations
with no separator (e.g. "cathatmap"), the vocabulary must be PREFIX-FREE: no
word may be a prefix of another, so every generated stream has exactly one
segmentation into words.
"""


def letter_set(word: str) -> set[str]:
    return set(word)


def letters_disjoint(w_a: str, w_b: str) -> bool:
    return letter_set(w_a).isdisjoint(letter_set(w_b))


def all_pairs_letter_disjoint(words: list[str]) -> bool:
    for i, w_i in enumerate(words):
        for j, w_j in enumerate(words):
            if i < j and not letters_disjoint(w_i, w_j):
                return False
    return True


def is_prefix_free(words: list[str]) -> bool:
    for i, w_i in enumerate(words):
        for j, w_j in enumerate(words):
            if i != j and w_j.startswith(w_i):
                return False
    return True


def describe_vocabulary(words: list[str]) -> str:
    """Non-enforcing descriptive label for display/metadata only."""
    if len(words) == 1:
        return "one_word"
    if all_pairs_letter_disjoint(words):
        return "disjoint_letters"
    return "shared_letters"


def validate_vocabulary(words: list[str]) -> None:
    if not words:
        raise ValueError("vocabulary must be non-empty")
    if any(not w for w in words):
        raise ValueError("vocabulary must not contain empty strings")
    if len(set(words)) != len(words):
        raise ValueError("vocabulary words must be unique")
    if not is_prefix_free(words):
        raise ValueError(
            "vocabulary must be prefix-free: no word may be a prefix of another "
            "(otherwise a separator-free stream would be ambiguous). "
            "Tip: use equal-length words, e.g. ['cat', 'hat', 'map']."
        )
