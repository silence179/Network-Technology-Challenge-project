"""
Popularity tracker using Exponentially Weighted Moving Average (EWMA).
"""

from collections import defaultdict

from ..config import EWMA_DECAY


class PopularityTracker:
    """Global content popularity tracker with EWMA scoring."""

    def __init__(self, decay=EWMA_DECAY):
        self.decay = decay
        self.scores = defaultdict(float)
        self.request_counts = defaultdict(int)

    def record(self, content_id):
        self.scores[content_id] = self.decay * self.scores[content_id] + 1.0
        self.request_counts[content_id] += 1

    def decay_all(self):
        for cid in list(self.scores.keys()):
            self.scores[cid] *= self.decay
            if self.scores[cid] < 0.01:
                del self.scores[cid]

    def top_k(self, k):
        return [c for c, _ in sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:k]]

    def score(self, content_id):
        return self.scores.get(content_id, 0.0)

    def reset_counts(self):
        self.request_counts = defaultdict(int)
