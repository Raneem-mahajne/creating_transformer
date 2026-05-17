"""Export trie and DFA to Graphviz DOT files."""
from __future__ import annotations

from pathlib import Path

from statistical_learning.dfa import DFA
from statistical_learning.trie import TrieNode


def export_trie_dot(root: TrieNode, path: Path) -> None:
    lines = [
        "digraph trie {",
        '  node [shape=circle];',
        "  root [label=root];",
    ]
    counter = [0]

    def node_id(node: TrieNode) -> str:
        if node is root:
            return "root"
        counter[0] += 1
        return f"n{counter[0]}"

    def walk(node: TrieNode, parent: str) -> None:
        nid = node_id(node)
        label = nid
        if node.is_terminal and node.word:
            label = f"{nid}\\n{node.word}"
        if parent != nid:
            lines.append(f'  {parent} [label="{parent}"];')
        for ch, child in sorted(node.children.items()):
            cid = node_id(child)
            lines.append(f'  {nid} -> {cid} [label="{ch}"];')
            walk(child, nid)

    walk(root, "root")
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_dfa_dot(dfa: DFA, path: Path) -> None:
    lines = [
        "digraph dfa {",
        '  node [shape=circle];',
    ]
    for s in dfa.states:
        label = str(s)
        if s in dfa.accepting:
            label = f"{s} (accept)"
        lines.append(f'  {s} [label="{label}"];')
    for (s, ch), t in sorted(dfa.transitions.items()):
        lines.append(f'  {s} -> {t} [label="{ch}"];')
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")
