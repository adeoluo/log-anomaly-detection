"""Next-event sequence detectors: DeepLog (Du et al., CCS 2017) and a transformer variant.

Both models read the last ``window`` events of a sequence and predict a distribution over the next
event. Training uses normal sequences only. At test time every event of a sequence is checked: if
the event that actually occurred is not among the model's ``top_g`` most likely next events, the
sequence is anomalous. The per-sequence score is the worst rank (0 = the most likely event) of any
actual event, so ``score > top_g - 0.5`` reproduces DeepLog's decision and ROC AUC uses the ranks.

Sequences are left-padded so the first events are checked too, and events never seen in training
map to an "unknown" token the model can never rank highly. Because logs repeat themselves, both
training and scoring work on distinct (history, next event) windows with counts: HDFS's millions
of windows reduce to tens of thousands. Requires PyTorch (``uv sync --extra deep``); runs on CUDA,
Apple MPS or CPU.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from log_anomaly_detection.features import Sequences


def unique_sequences(
    seqs: Sequences,
) -> tuple[Sequences, NDArray[np.int64], NDArray[np.int64]]:
    """Distinct sequences, how often each occurs, and each input sequence's distinct index."""

    index: dict[bytes, int] = {}
    inverse = np.empty(len(seqs), dtype=np.int64)
    first: list[int] = []
    events = seqs.events
    offsets = seqs.offsets
    for i in range(len(seqs)):
        key = events[offsets[i] : offsets[i + 1]].tobytes()
        found = index.get(key)
        if found is None:
            found = len(first)
            index[key] = found
            first.append(i)
        inverse[i] = found
    counts = np.bincount(inverse, minlength=len(first)).astype(np.int64)
    return seqs.subset(np.asarray(first, dtype=np.int64)), counts, inverse


def make_windows(
    seqs: Sequences, window: int, pad: int
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int64]]:
    """(history, next event, sequence index) for every event of every sequence, left-padded."""

    n = int(seqs.events.size)
    if n == 0:
        return np.empty((0, window), np.int32), np.empty(0, np.int32), np.empty(0, np.int64)
    rows = seqs.rows()
    pos = np.arange(n, dtype=np.int64)
    idx = pos[:, None] - window + np.arange(window, dtype=np.int64)[None, :]
    valid = idx >= seqs.offsets[rows][:, None]
    hist = np.where(valid, seqs.events[np.clip(idx, 0, None)], pad).astype(np.int32)
    return hist, seqs.events.astype(np.int32), rows


def _dedupe_windows(
    hist: NDArray[np.int32], nxt: NDArray[np.int32], weights: NDArray[np.int64]
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int64], NDArray[np.int64]]:
    both = np.concatenate([hist, nxt[:, None]], axis=1)
    uniq, inverse = np.unique(both, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    agg = np.bincount(inverse, weights=weights, minlength=uniq.shape[0]).astype(np.int64)
    return uniq[:, :-1].copy(), uniq[:, -1].copy(), agg, inverse


def _device(name: str) -> Any:
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NextEventDetector:
    """Shared training and scoring for next-event models; subclasses build the network."""

    name = "next-event"

    def __init__(
        self,
        window: int = 10,
        top_g: int = 9,
        embedding: int = 32,
        hidden: int = 64,
        layers: int = 2,
        epochs: int = 10,
        batch_size: int = 1024,
        samples_per_epoch: int = 200_000,
        learning_rate: float = 1e-3,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        if window < 1 or top_g < 1 or epochs < 1 or batch_size < 1 or samples_per_epoch < 1:
            raise ValueError("window, top_g, epochs, batch_size and samples must be positive")
        self.window = window
        self.top_g = top_g
        self.embedding = embedding
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.learning_rate = learning_rate
        self.device_name = device
        self.seed = seed
        self.threshold_ = top_g - 0.5
        self.vocab_: int = 0
        self.model_: Any = None
        self.history_: list[float] = []
        self.train_windows_: int = 0
        self.distinct_windows_: int = 0
        self.train_seconds_: float = 0.0

    # tokens: 0..n_events-1 events, n_events = unknown, n_events + 1 = padding
    def _build(self, vocab: int) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def params(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "top_g": self.top_g,
            "embedding": self.embedding,
            "hidden": self.hidden,
            "layers": self.layers,
            "epochs": self.epochs,
            "samples_per_epoch": self.samples_per_epoch,
            "seed": self.seed,
            "train_windows": self.train_windows_,
            "distinct_windows": self.distinct_windows_,
            "final_loss": round(self.history_[-1], 4) if self.history_ else None,
        }

    def _encode(self, seqs: Sequences) -> Sequences:
        unknown = self.vocab_
        return Sequences(np.minimum(seqs.events, unknown), seqs.offsets.copy())

    def fit(self, seqs: Sequences, n_events: int) -> NextEventDetector:
        import torch

        if len(seqs) == 0 or seqs.events.size == 0:
            raise ValueError("cannot fit on empty sequences")
        start = time.perf_counter()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.vocab_ = n_events
        uniq, counts, _ = unique_sequences(self._encode(seqs))
        hist, nxt, rows = make_windows(uniq, self.window, pad=n_events + 1)
        hist, nxt, weights, _ = _dedupe_windows(hist, nxt, counts[rows])
        self.train_windows_ = int(weights.sum())
        self.distinct_windows_ = int(weights.size)
        device = _device(self.device_name)
        model = self._build(n_events + 2).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        x_all = torch.from_numpy(hist.astype(np.int64))
        y_all = torch.from_numpy(nxt.astype(np.int64))
        prob = weights / weights.sum()
        per_epoch = min(self.samples_per_epoch, max(self.train_windows_, self.batch_size))
        self.history_ = []
        model.train()
        for _ in range(self.epochs):
            picks = rng.choice(weights.size, size=per_epoch, p=prob)
            total = 0.0
            batches = 0
            for b in range(0, per_epoch, self.batch_size):
                sel = torch.from_numpy(picks[b : b + self.batch_size])
                xb = x_all[sel].to(device)
                yb = y_all[sel].to(device)
                optim.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optim.step()
                total += float(loss.detach().cpu())
                batches += 1
            self.history_.append(total / max(batches, 1))
        model.eval()
        self.model_ = model
        self.train_seconds_ = time.perf_counter() - start
        return self

    def _ranks(self, hist: NDArray[np.int32], nxt: NDArray[np.int32]) -> NDArray[np.int64]:
        """Rank of each actual next event under the model (0 = most likely)."""

        import torch

        if self.model_ is None:
            raise RuntimeError(f"{self.name} is not fitted")
        device = next(self.model_.parameters()).device
        out = np.empty(nxt.size, dtype=np.int64)
        with torch.no_grad():
            for b in range(0, nxt.size, 8192):
                xb = torch.from_numpy(hist[b : b + 8192].astype(np.int64)).to(device)
                yb = torch.from_numpy(nxt[b : b + 8192].astype(np.int64)).to(device)
                logits = self.model_(xb).float()
                # Unknown and padding tokens are never valid predictions.
                logits[:, self.vocab_ :] = -math.inf
                actual = logits.gather(1, yb[:, None])
                rank = (logits > actual).sum(dim=1)
                unknown = yb >= self.vocab_
                rank[unknown] = self.vocab_
                out[b : b + 8192] = rank.cpu().numpy()
        return out

    def score(self, seqs: Sequences) -> NDArray[np.float64]:
        """Worst rank of any actual event in each sequence (empty sequences score 0)."""

        if self.model_ is None:
            raise RuntimeError(f"{self.name} is not fitted")
        uniq, _, inverse = unique_sequences(self._encode(seqs))
        hist, nxt, rows = make_windows(uniq, self.window, pad=self.vocab_ + 1)
        out = np.zeros(len(uniq), dtype=np.float64)
        if nxt.size:
            uh, un, _, winv = _dedupe_windows(hist, nxt, np.ones(nxt.size, dtype=np.int64))
            ranks = self._ranks(uh, un)[winv]
            np.maximum.at(out, rows, ranks.astype(np.float64))
        return out[inverse]

    def predict(self, seqs: Sequences) -> NDArray[np.bool_]:
        return self.score(seqs) > self.threshold_


class DeepLogDetector(NextEventDetector):
    """DeepLog: a stacked LSTM over the event history."""

    name = "DeepLog"

    def _build(self, vocab: int) -> Any:
        import torch
        from torch import nn

        embedding, hidden, layers = self.embedding, self.hidden, self.layers

        class LSTMNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab, embedding)
                self.lstm = nn.LSTM(embedding, hidden, num_layers=layers, batch_first=True)
                self.out = nn.Linear(hidden, vocab)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h, _ = self.lstm(self.embed(x))
                return self.out(h[:, -1, :])  # type: ignore[no-any-return]

        return LSTMNet()


class TransformerDetector(NextEventDetector):
    """A small transformer encoder over the event history (learned positions, last-token head)."""

    name = "Transformer"

    def __init__(self, heads: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.heads = heads

    def params(self) -> dict[str, Any]:
        return {**super().params(), "heads": self.heads}

    def _build(self, vocab: int) -> Any:
        import torch
        from torch import nn

        hidden, layers, heads, window = self.hidden, self.layers, self.heads, self.window

        class TransformerNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab, hidden)
                self.pos = nn.Parameter(torch.zeros(1, window, hidden))
                layer = nn.TransformerEncoderLayer(
                    hidden, heads, dim_feedforward=2 * hidden, dropout=0.0, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(hidden)
                self.out = nn.Linear(hidden, vocab)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.encoder(self.embed(x) + self.pos)
                return self.out(self.norm(h[:, -1, :]))  # type: ignore[no-any-return]

        return TransformerNet()


SEQUENCE_DETECTORS: dict[str, type[NextEventDetector]] = {
    "deeplog": DeepLogDetector,
    "transformer": TransformerDetector,
}
