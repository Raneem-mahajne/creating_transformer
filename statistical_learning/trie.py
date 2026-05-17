"""Prefix trie for a finite word list."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrieNode:
    children: dict[str, TrieNode] = field(default_factory=dict)
    is_terminal: bool = False
    word: str | None = None
    word_id: int | None = None


def build_trie(words: list[str]) -> TrieNode:
    root = TrieNode()
    for word_id, word in enumerate(words):
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_terminal = True
        node.word = word
        node.word_id = word_id
    return root


def trie_to_dict(root: TrieNode, words: list[str]) -> dict:
    def node_to_dict(node: TrieNode) -> dict:
        out: dict = {"terminal": node.is_terminal, "children": {}}
        if node.is_terminal:
            out["word"] = node.word
            out["word_id"] = node.word_id
        for ch, child in sorted(node.children.items()):
            out["children"][ch] = node_to_dict(child)
        return out

    return {"words": words, "root": node_to_dict(root)}


def dict_to_trie(data: dict) -> tuple[TrieNode, list[str]]:
    words = list(data["words"])

    def dict_to_node(d: dict) -> TrieNode:
        node = TrieNode(
            is_terminal=d.get("terminal", False),
            word=d.get("word"),
            word_id=d.get("word_id"),
        )
        for ch, child_d in d.get("children", {}).items():
            node.children[ch] = dict_to_node(child_d)
        return node

    return dict_to_node(data["root"]), words


def get_node(root: TrieNode, prefix: str) -> TrieNode | None:
    node = root
    for ch in prefix:
        if ch not in node.children:
            return None
        node = node.children[ch]
    return node


def valid_next_chars(root: TrieNode, prefix: str) -> list[str]:
    """Characters that can legally extend prefix (in-trie continuations)."""
    node = get_node(root, prefix)
    if node is None:
        return []
    return sorted(node.children.keys())
