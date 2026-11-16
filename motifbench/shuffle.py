"""Dinucleotide-preserving shuffle (Altschul and Erickson, 1985).

Used to make negative sequences for the CNN: each shuffled copy keeps the exact
count of every base pair (AA, AC, ...) of its original but scrambles the order,
so a planted motif is destroyed while the base composition is kept.
"""
import numpy as np


def dinucleotide_shuffle(seq: str, rng: np.random.Generator) -> str:
    if len(seq) < 3:
        return seq
    # Outgoing edges of each letter, in order of appearance.
    edges = {}
    for a, b in zip(seq[:-1], seq[1:]):
        edges.setdefault(a, []).append(b)
    last = seq[-1]

    # Random "last edge" for every vertex, forming a tree rooted at the last letter
    # (random walks, Wilson's algorithm). This guarantees an Eulerian path exists.
    in_tree = {last}
    next_of = {}
    for v in edges:
        u = v
        while u not in in_tree:
            next_of[u] = edges[u][rng.integers(len(edges[u]))]
            u = next_of[u]
        u = v
        while u not in in_tree:
            in_tree.add(u)
            u = next_of[u]

    # Shuffle each edge list, keeping the chosen last edge at the end.
    ordered = {}
    for v, outs in edges.items():
        outs = list(outs)
        if v != last:
            outs.remove(next_of[v])
        perm = rng.permutation(len(outs))
        outs = [outs[i] for i in perm]
        if v != last:
            outs.append(next_of[v])
        ordered[v] = outs

    # Walk the Eulerian path from the original first letter.
    pointer = {v: 0 for v in ordered}
    out = [seq[0]]
    cur = seq[0]
    for _ in range(len(seq) - 1):
        nxt = ordered[cur][pointer[cur]]
        pointer[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)
