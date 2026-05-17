"""Word-list validation for statistical-learning complexity regimes."""


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


def has_letter_overlap(words: list[str]) -> bool:
    for i, w_i in enumerate(words):
        for j, w_j in enumerate(words):
            if i < j and not letters_disjoint(w_i, w_j):
                return True
    return False


def validate_words(words: list[str], complexity: str) -> None:
    if not words:
        raise ValueError("words must be non-empty")
    if any(not w for w in words):
        raise ValueError("words must not contain empty strings")
    if len(set(words)) != len(words):
        raise ValueError("words must be unique")

    if complexity == "one_word":
        if len(words) != 1:
            raise ValueError(f"one_word complexity requires exactly one word, got {len(words)}")
    elif complexity == "disjoint_letters":
        if len(words) < 2:
            raise ValueError("disjoint_letters complexity requires at least two words")
        if not all_pairs_letter_disjoint(words):
            raise ValueError("disjoint_letters words must not share any letters between words")
    elif complexity == "shared_letters":
        if len(words) < 2:
            raise ValueError("shared_letters complexity requires at least two words")
        if not has_letter_overlap(words):
            raise ValueError("shared_letters words must share at least one letter across some pair")
    else:
        raise ValueError(f"unknown complexity: {complexity}")
