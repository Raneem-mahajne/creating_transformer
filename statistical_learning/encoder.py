"""Character-level encoder for statistical-learning corpora."""


def build_alphabet(words: list[str], delimiter: str) -> list[str]:
    chars = set()
    for w in words:
        chars.update(w)
    chars.add(delimiter)
    return sorted(chars)


def build_char_encoder(alphabet: list[str]):
    token_to_index = {ch: i for i, ch in enumerate(alphabet)}
    index_to_token = {i: ch for i, ch in enumerate(alphabet)}
    vocab_size = len(alphabet)
    index_to_string = {i: ch for i, ch in enumerate(alphabet)}
    string_to_index = {ch: i for i, ch in enumerate(alphabet)}

    def encode(chars: list[str]) -> list[int]:
        return [token_to_index[c] for c in chars]

    def decode(token_indices: list[int]) -> list[str]:
        return [index_to_token[idx] for idx in token_indices]

    return encode, decode, vocab_size, index_to_string, string_to_index
