"""DNA alphabet helpers: encoding, one-hot, reverse complement."""
import numpy as np

ALPHABET = "ACGT"
_INDEX = {b: i for i, b in enumerate(ALPHABET)}


def encode(seq: str) -> np.ndarray:
    """Encode an ACGT string as an int8 array (A=0, C=1, G=2, T=3)."""
    try:
        return np.fromiter((_INDEX[b] for b in seq.upper()), dtype=np.int8, count=len(seq))
    except KeyError as err:
        raise ValueError(f"unsupported base {err.args[0]!r}; only A, C, G, T are allowed") from None


def decode(arr) -> str:
    return "".join(ALPHABET[int(i)] for i in arr)


def encode_all(seqs) -> np.ndarray:
    """Encode equal-length sequences as an (N, L) int8 array."""
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError("all sequences must have the same length")
    return np.stack([encode(s) for s in seqs])


def one_hot(X: np.ndarray) -> np.ndarray:
    """(N, L) int array -> (N, 4, L) float32 one-hot array."""
    out = np.zeros((X.shape[0], 4, X.shape[1]), dtype=np.float32)
    n_idx, l_idx = np.indices(X.shape)
    out[n_idx, X, l_idx] = 1.0
    return out


def revcomp_pwm(pwm: np.ndarray) -> np.ndarray:
    """Reverse complement of a 4 x W matrix (rows A, C, G, T)."""
    return pwm[::-1, ::-1].copy()
