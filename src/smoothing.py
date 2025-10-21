# Trigram smoothing and backoff models.
# This file defines:
#   1. Add-1 (Laplace) smoothing
#   2. Linear interpolation (unigram + bigram + trigram)
#   3. Stupid Backoff
# Plus a small helper to tune interpolation weights.

from collections import Counter
from typing import List, Tuple, Optional
import math
from src.ngram_model import NgramMLE, pad_sentence, iter_ngrams

def vocab_size_from_train(train: List[List[str]]) -> int:
    """Return the number of unique words in the training set."""
    vocab = set()
    for s in train:
        vocab.update(s)
    return len(vocab)


# 1. Add-1 (Laplace) Smoothing 

class LaplaceTrigram:
    """
    Trigram model with Add-1 (Laplace) smoothing: P(w | h) = (count(h, w) + 1) / (count(h) + V)
    where: h = bigram history; V = vocabulary size
    """

    def __init__(self):
        self.V = 0
        self.trigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trained = False

    def fit(self, train: List[List[str]]) -> None:
        """Build bigram and trigram counts and store vocabulary size."""
        self.V = vocab_size_from_train(train)
        self.trigram_counts.clear()
        self.bigram_counts.clear()

        for sent in train:
            padded = pad_sentence(sent, 3)
            for ng in iter_ngrams(padded, 3):
                self.trigram_counts[ng] += 1
                self.bigram_counts[ng[:-1]] += 1

        self.trained = True

    def prob(self, history: Tuple[str, str], word: str) -> float:
        """Compute Add-1 smoothed probability."""

        numerator = self.trigram_counts.get(history + (word,), 0) + 1
        denominator = self.bigram_counts.get(history, 0) + self.V
        return numerator / denominator

    def sentence_logprob(self, sent: List[str]) -> float:
        """Return log P(sentence) using the smoothed model."""
        logp = 0.0
        padded = pad_sentence(sent, 3)
        for ng in iter_ngrams(padded, 3):
            hist, w = ng[:-1], ng[-1]
            logp += math.log(self.prob(hist, w))
        return logp


# 2. Linear Interpolation (unigram + bigram + trigram)

class InterpolatedTrigram:
    """
    Linear interpolation of unigram, bigram, and trigram MLE models: P(w | h) = l1*P1(w) + l2*P2(w|w-1) + l3*P3(w|w-2, w-1)
    where l1 + l2 + l3 = 1.
    """

    def __init__(self, lambdas: Tuple[float, float, float]):
        self.l1, self.l2, self.l3 = lambdas
        self.unigram = NgramMLE(1)
        self.bigram = NgramMLE(2)
        self.trigram = NgramMLE(3)
        self.trained = False

    def fit(self, train: List[List[str]]) -> None:
        """Train unigram, bigram, and trigram MLE models."""
        self.unigram.fit(train)
        self.bigram.fit(train)
        self.trigram.fit(train)
        self.trained = True

    def prob(self, history: Tuple[str, str], word: str) -> float:
        """Return the interpolated probability."""

        p1 = self.unigram.prob_mle((), word)
        p2 = self.bigram.prob_mle((history[-1],), word)
        p3 = self.trigram.prob_mle(history, word)
        return self.l1 * p1 + self.l2 * p2 + self.l3 * p3

    def sentence_logprob(self, sent: List[str]) -> float:
        """Compute the total log probability of a sentence."""
        logp = 0.0
        padded = pad_sentence(sent, 3)
        for ng in iter_ngrams(padded, 3):
            hist, w = ng[:-1], ng[-1]
            p = self.prob(hist, w)
            if p <= 0.0:
                return float("-inf")
            logp += math.log(p)
        return logp


def tune_interpolation_lambdas(train: List[List[str]], valid: List[List[str]], grid: Optional[List[Tuple[float, float, float]]] = None) -> Tuple[Tuple[float, float, float], float]:
    """
    Try a few sets of (l1, l2, l3) values that sum to 1. Return the best combination based on validation perplexity.
    """
    from src.evaluate import perplexity

    if grid is None:
        grid = [(0.1, 0.3, 0.6),(0.2, 0.3, 0.5),(0.3, 0.3, 0.4),(0.33, 0.33, 0.34),(0.4, 0.3, 0.3),(0.1, 0.2, 0.7)]

    base = InterpolatedTrigram((1.0, 0.0, 0.0))
    base.fit(train)

    best_lambdas = None
    best_perplexity = float("inf")

    for lmb in grid:
        model = InterpolatedTrigram(lmb)
        model.unigram, model.bigram, model.trigram = base.unigram, base.bigram, base.trigram
        model.trained = True

        pp = perplexity(model, valid)
        if pp < best_perplexity:
            best_perplexity = pp
            best_lambdas = lmb

    return best_lambdas, best_perplexity


# 3. Stupid Backoff (trigram to bigram to unigram) 

class StupidBackoffTrigram:
    """
    Stupid Backoff uses trigram probability if available. Otherwise, use backoff_factor * bigram probability. 
    If bigram is unseen, use backoff_factor * unigram probability.
    """

    def __init__(self, backoff_factor: float = 0.4):
        self.backoff_factor = backoff_factor
        self.unigram = NgramMLE(1)
        self.bigram = NgramMLE(2)
        self.trigram = NgramMLE(3)
        self.trained = False

    def fit(self, train: List[List[str]]) -> None:
        """Train all component MLE models."""
        self.unigram.fit(train)
        self.bigram.fit(train)
        self.trigram.fit(train)
        self.trained = True

    def prob(self, history: Tuple[str, str], word: str) -> float:
        """Return probability with backoff if needed."""

        p3 = self.trigram.prob_mle(history, word)
        if p3 > 0.0:
            return p3

        p2 = self.bigram.prob_mle((history[-1],), word)
        if p2 > 0.0:
            return self.backoff_factor * p2

        p1 = self.unigram.prob_mle((), word)
        if p1 > 0.0:
            return self.backoff_factor * p1

        return 0.0

    def sentence_logprob(self, sent: List[str]) -> float:
        """Compute sentence log probability with backoff."""
        logp = 0.0
        padded = pad_sentence(sent, 3)
        for ng in iter_ngrams(padded, 3):
            hist, w = ng[:-1], ng[-1]
            p = self.prob(hist, w)
            if p <= 0.0:
                return float("-inf")
            logp += math.log(p)
        return logp
