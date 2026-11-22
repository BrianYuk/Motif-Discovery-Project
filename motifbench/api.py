import numpy as np

from .alphabet import encode_all

METHODS = ("em", "gibbs", "cnn")


def find_motif(sequences, width: int, method: str, seed: int = 0, **kwargs) -> dict:
    # Find one motif of the given width.
    X = sequences if isinstance(sequences, np.ndarray) else encode_all(sequences)
    if method == "em":
        from .em import find_motif_em
        return find_motif_em(X, width, seed=seed, **kwargs)
    if method == "gibbs":
        from .gibbs import find_motif_gibbs
        return find_motif_gibbs(X, width, seed=seed, **kwargs)
    if method == "cnn":
        from .cnn import find_motif_cnn
        return find_motif_cnn(X, width, seed=seed, **kwargs)
    raise ValueError(f"unknown method {method!r}; choose one of {METHODS}")
