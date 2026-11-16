"""One-layer CNN motif model (DeepBind style) with filter-to-PWM conversion.

The network separates the input sequences from dinucleotide-shuffled copies.
Each first-layer filter scans the one-hot sequence like a log-odds PWM; after
training, a filter becomes a PWM by counting the bases of the windows that
activate it strongly.
"""
import numpy as np

from .alphabet import one_hot
from .shuffle import dinucleotide_shuffle
from .alphabet import decode, encode_all


def _torch():
    import torch  # imported lazily so EM and Gibbs work without PyTorch
    return torch


def build_model(width: int, n_filters: int):
    torch = _torch()
    nn = torch.nn

    class MotifCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(4, n_filters, kernel_size=width)
            self.out = nn.Linear(n_filters, 1)

        def activations(self, x):                 # (B, 4, L) -> (B, F, m)
            return torch.relu(self.conv(x))

        def forward(self, x):                     # global max-pool over positions
            return self.out(self.activations(x).amax(dim=2)).squeeze(1)

    return MotifCNN()


def make_training_set(X: np.ndarray, rng: np.random.Generator):
    """Positives = the sequences; negatives = one dinucleotide shuffle of each."""
    neg = encode_all([dinucleotide_shuffle(decode(row), rng) for row in X])
    data = np.concatenate([one_hot(X), one_hot(neg)])
    labels = np.concatenate([np.ones(len(X)), np.zeros(len(neg))]).astype(np.float32)
    return data, labels


def train(model, data, labels, rng, epochs=100, batch_size=64, lr=1e-2,
          weight_decay=1e-4, val_frac=0.2, patience=15):
    torch = _torch()
    idx = rng.permutation(len(data))
    n_val = max(1, int(len(data) * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    xt, yt = torch.from_numpy(data[tr_idx]), torch.from_numpy(labels[tr_idx])
    xv, yv = torch.from_numpy(data[val_idx]), torch.from_numpy(labels[val_idx])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_loss, best_state, bad = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(xt))
        for start in range(0, len(order), batch_size):
            b = torch.from_numpy(order[start:start + batch_size])
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(xv), yv))
        if v < best_loss - 1e-4:
            best_loss, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"val_loss": best_loss, "epochs": epoch + 1}


def filter_to_pwm(acts: np.ndarray, win: np.ndarray, threshold_frac=0.5, pseudocount=0.25):
    """acts: (N, m) activations of one filter on the positives; win: (N, m, W) windows.
    Count the bases of every window whose activation exceeds threshold_frac x max."""
    width = win.shape[-1]
    peak = acts.max()
    counts = np.zeros((4, width))
    if peak > 0:
        n_idx, a_idx = np.nonzero(acts > threshold_frac * peak)
        cols = np.arange(width)
        for n, a in zip(n_idx, a_idx):
            np.add.at(counts, (win[n, a], cols), 1.0)
    return (counts + pseudocount) / (counts.sum(axis=0) + 4 * pseudocount)


def find_motif_cnn(X, width, seed=0, n_filters=16, background=None, **train_kwargs) -> dict:
    torch = _torch()
    from .em import windows

    torch.manual_seed(seed)
    torch.set_num_threads(1)              # same floating-point results on any machine size
    rng = np.random.default_rng(seed)
    data, labels = make_training_set(X, rng)
    model = build_model(width, n_filters)
    info = train(model, data, labels, rng, **train_kwargs)

    model.eval()
    with torch.no_grad():
        pos = torch.from_numpy(data[:len(X)])
        neg = torch.from_numpy(data[len(X):])
        act_pos = model.activations(pos).numpy()          # (N, F, m)
        act_neg = model.activations(neg).numpy()
    weights = model.out.weight.detach().numpy()[0]        # (F,)
    # The filter that best separates positives from negatives, weighted by how the
    # output layer uses it.
    gain = weights * (act_pos.max(axis=2).mean(axis=0) - act_neg.max(axis=2).mean(axis=0))
    f = int(np.argmax(gain))
    acts = act_pos[:, f, :]
    win = windows(X, width)
    pwm = filter_to_pwm(acts, win)
    peak = acts.max()
    best = acts.argmax(axis=1)
    sites = [int(a) if peak > 0 and acts[n, a] > 0.5 * peak else None for n, a in enumerate(best)]
    return {"pwm": pwm, "sites": sites, "score": float(gain[f]),
            "extra": {"filter": f, **info}}
