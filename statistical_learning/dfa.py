"""Cyclic concatenation DFA built from a prefix-free word trie.

Sequences are raw concatenations of vocabulary words with no separator
(e.g. "cathatmap"). The automaton walks the trie while reading a word and,
on reaching a terminal (end of a word), restarts a new word by following the
root's first-character transitions. For a prefix-free vocabulary terminal
nodes have no children, so the result is a deterministic cyclic DFA.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from statistical_learning.trie import TrieNode, build_trie


@dataclass
class DFA:
    states: list[int]
    start: int
    accepting: list[int]
    alphabet: list[str]
    transitions: dict[tuple[int, str], int]
    state_metadata: dict[int, dict] = field(default_factory=dict)


def dfa_from_trie(root: TrieNode, words: list[str]) -> DFA:
    """Build a cyclic DFA over the concatenation language (w1|w2|...)+."""
    id_map: dict[int, int] = {}  # id(node) -> state_id
    nodes: list[TrieNode] = []
    queue: deque[TrieNode] = deque([root])
    seen: set[int] = set()

    while queue:
        node = queue.popleft()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        id_map[nid] = len(nodes)
        nodes.append(node)
        for child in node.children.values():
            if id(child) not in seen:
                queue.append(child)

    transitions: dict[tuple[int, str], int] = {}
    accepting: list[int] = []
    state_metadata: dict[int, dict] = {}
    alphabet_chars: set[str] = set()

    for state_id, node in enumerate(nodes):
        depth = _depth_from_root(root, node, nodes)
        words_ending = [node.word] if node.is_terminal and node.word else []
        state_metadata[state_id] = {
            "depth": depth,
            "words_ending": words_ending,
            "is_terminal": node.is_terminal,
        }
        if node.is_terminal:
            accepting.append(state_id)
        for ch, child in node.children.items():
            alphabet_chars.add(ch)
            transitions[(state_id, ch)] = id_map[id(child)]

    # Concatenation: after finishing a word (terminal state), the next character
    # starts a new word, so restart by following the root's first-char edges.
    root_state = id_map[id(root)]
    for acc in accepting:
        for ch, child in root.children.items():
            transitions[(acc, ch)] = id_map[id(child)]

    return DFA(
        states=list(range(len(nodes))),
        start=root_state,
        accepting=sorted(accepting),
        alphabet=sorted(alphabet_chars),
        transitions=transitions,
        state_metadata=state_metadata,
    )


def _depth_from_root(root: TrieNode, target: TrieNode, nodes: list[TrieNode]) -> int:
    if target is root:
        return 0

    def dfs(node: TrieNode, depth: int) -> int | None:
        if node is target:
            return depth
        for child in node.children.values():
            found = dfs(child, depth + 1)
            if found is not None:
                return found
        return None

    return dfs(root, 0) or 0


def dfa_to_dict(dfa: DFA, words: list[str]) -> dict:
    transitions = {f"{s},{ch}": t for (s, ch), t in dfa.transitions.items()}
    return {
        "words": words,
        "states": dfa.states,
        "start": dfa.start,
        "accepting": dfa.accepting,
        "alphabet": dfa.alphabet,
        "transitions": transitions,
        "state_metadata": {str(k): v for k, v in dfa.state_metadata.items()},
    }


def dict_to_dfa(data: dict) -> tuple[DFA, list[str]]:
    words = list(data["words"])
    transitions: dict[tuple[int, str], int] = {}
    for key, target in data["transitions"].items():
        s_str, ch = key.split(",", 1)
        transitions[(int(s_str), ch)] = int(target)
    metadata = {int(k): v for k, v in data.get("state_metadata", {}).items()}
    dfa = DFA(
        states=[int(s) for s in data["states"]],
        start=int(data["start"]),
        accepting=[int(s) for s in data["accepting"]],
        alphabet=list(data["alphabet"]),
        transitions=transitions,
        state_metadata=metadata,
    )
    return dfa, words


def build_dfa(words: list[str]) -> DFA:
    root = build_trie(words)
    return dfa_from_trie(root, words)
