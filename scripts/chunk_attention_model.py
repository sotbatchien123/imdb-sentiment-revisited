"""Plan C head: shared BiLSTM chunk vectors pooled by mean (B2) or additive attention (C1).

Chunks are kept flat with an owner index per chunk, so there are no padding chunks:
softmax and pooling run per review over that review's real chunks only.
"""

import hashlib
import torch
from torch import nn
from phase2_core import padded, encoder_features

AGGREGATIONS = ("mean", "attention")


class ChunkCompareHead(nn.Module):
    """B2 and C1 differ only in pooling. lstm/classifier names match ChunkHead so shared init loads into both."""

    def __init__(self, cfg, aggregation, input_size=768):
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        self.aggregation = aggregation
        hidden = cfg["lstm_hidden_size"]
        self.dropout = nn.Dropout(cfg["dropout"])
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden, 2)
        if aggregation == "attention":
            self.attention = nn.Linear(2 * hidden, cfg["attention_dim"])
            self.attention_vector = nn.Linear(cfg["attention_dim"], 1, bias=False)

    def represent(self, x, mask):
        x = self.dropout(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return torch.cat((h[-2], h[-1]), dim=1)

    def pool(self, reps, owners, nreviews):
        """Return pooled review vectors and one weight per chunk (weights of a review sum to 1). FP32 throughout."""
        reps = reps.float()
        counts = torch.bincount(owners, minlength=nreviews)
        if self.aggregation == "mean":
            weights = (1.0 / counts.to(reps.dtype))[owners]
        else:
            scores = self.attention_vector(torch.tanh(self.attention(reps))).squeeze(-1)
            # Per-review max shift for a stable segment softmax; the shift cancels, so it carries no gradient.
            peak = scores.new_full((nreviews,), -torch.inf).scatter_reduce(
                0, owners, scores.detach(), "amax"
            )
            exp = (scores - peak[owners]).exp()
            weights = exp / exp.new_zeros(nreviews).index_add(0, owners, exp)[owners]
        pooled = reps.new_zeros((nreviews, reps.shape[-1])).index_add(
            0, owners, weights.unsqueeze(1) * reps
        )
        return pooled, weights

    def forward(self, reps, owners, nreviews, return_weights=False):
        pooled, weights = self.pool(reps, owners, nreviews)
        logits = self.classifier(self.dropout(pooled))
        return (logits, weights) if return_weights else logits


def forward_reviews(
    encoder,
    head,
    review_sequences,
    device,
    microbatch=4,
    checkpoint_head=False,
    return_weights=False,
):
    """Frozen encoder over chunk microbatches, shared head per chunk, then one prediction per review."""
    flat = []
    owners = []
    for owner, chunks in enumerate(review_sequences):
        flat.extend(chunks)
        owners.extend([owner] * len(chunks))
    reps = []
    for start in range(0, len(flat), microbatch):
        ids, mask = padded(flat[start : start + microbatch], device)
        features = encoder_features(encoder, ids, mask, device)
        if checkpoint_head and head.training:
            from torch.utils.checkpoint import checkpoint

            reps.append(checkpoint(head.represent, features, mask, use_reentrant=False))
        else:
            reps.append(head.represent(features, mask))
    return head(
        torch.cat(reps), torch.tensor(owners, device=device), len(review_sequences), return_weights
    )


def shared_init_state(cfg, seed):
    """BiLSTM + classifier init for one seed, identical for B2 and C1; global RNG is left untouched."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = ChunkCompareHead(cfg, "mean")
    return base.state_dict()


def build_head(cfg, aggregation, seed, shared_state):
    """Attention params come from their own seed so adding them never shifts shared init or batch order."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + cfg["attention_init_seed_offset"])
        head = ChunkCompareHead(cfg, aggregation)
    missing, unexpected = head.load_state_dict(shared_state, strict=False)
    if unexpected or any(not k.startswith("attention") for k in missing):
        raise RuntimeError(f"Shared init mismatch: missing={missing} unexpected={unexpected}")
    return head


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def accumulation_windows(n, batch_size, accumulation):
    """Yield (window_size, [(start,end),...]); loss of each window is normalized by its real review count."""
    window = batch_size * accumulation
    for ws in range(0, n, window):
        end = min(ws + window, n)
        yield end - ws, [(s, min(s + batch_size, end)) for s in range(ws, end, batch_size)]


def attention_stats(weights, owners, nreviews):
    """Mean normalized entropy and mean max weight over reviews with more than one chunk (log(1) never divides)."""
    entropies = []
    peaks = []
    for r in range(nreviews):
        w = weights[owners == r]
        if len(w) < 2:
            continue
        entropies.append(
            float(-(w * w.clamp_min(1e-12).log()).sum() / torch.log(torch.tensor(float(len(w)))))
        )
        peaks.append(float(w.max()))
    return entropies, peaks
