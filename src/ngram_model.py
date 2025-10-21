from collections import Counter
from typing import Iterable, List, Tuple

BOS = "<s>"
EOS = "</s>"

def pad_sentence(tokens: List[str], n: int) -> List[str]:
    """
    Adds (n-1) <s> tokens to the beginning of a sentence.
    """
    if n <= 1:
        return tokens
    return [BOS] * (n - 1) + tokens

def iter_ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    """
    Yields all n-grams in a padded sentence.
    """
    L = len(tokens)
    if L < n:
        return
    for i in range(L - n + 1):
        yield tuple(tokens[i : i + n])

class NgramMLE:
    """
    Basic (unsmoothed) n-gram model using Maximum Likelihood Estimation. Estimates probabilities directly from observed counts in the training data.
    """
    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.ngram_counts: Counter = Counter()      # counts of n-grams
        self.ctx_counts: Counter = Counter()        # counts of (n-1)-gram contexts
        self.trained = False

    def fit(self, corpus: List[List[str]]) -> None:
        """
        Builds frequency counts for all n-grams and their contexts.
        """
        n = self.n
        self.ngram_counts.clear()
        self.ctx_counts.clear()

        if n == 1:
            # For unigrams, there is no history. Store total number of tokens under key () so the same formula works.
            for sent in corpus:
                for w in sent:
                    self.ngram_counts[(w,)] += 1
            self.ctx_counts[()] = sum(self.ngram_counts.values())
        else:
            for sent in corpus:
                padded = pad_sentence(sent, n)
                for ng in iter_ngrams(padded, n):
                    self.ngram_counts[ng] += 1
                    history = ng[:-1]
                    self.ctx_counts[history] += 1

        self.trained = True

    def prob_mle(self, history: Tuple[str, ...], word: str) -> float:
        """
        Returns P(word | history) = count(history + word) / count(history); Returns 0.0 if unseen.
        """

        if self.n == 1:
            # P(w) = count(w)/total_tokens
            num = self.ngram_counts.get((word,), 0)
            denom = self.ctx_counts.get((), 0)
            return (num / denom) if denom > 0 else 0.0

        if len(history) != self.n - 1:
            raise ValueError(f"history must be length {self.n-1} for an {self.n}-gram model")

        num = self.ngram_counts.get(history + (word,), 0)
        denom = self.ctx_counts.get(history, 0)
        return (num / denom) if denom > 0 else 0.0

    def sentence_logprob(self, sent: List[str]) -> float:
        """
        Calculates log P(sentence) = sum of log probabilities of each word. Returns -inf if any probability is zero.
        """
        import math
        
        if self.n == 1:
            logp = 0.0
            for w in sent:
                p = self.prob_mle((), w)
                if p == 0.0:
                    return float("-inf")
                logp += math.log(p)
            return logp

        logp = 0.0
        padded = pad_sentence(sent, self.n)
        for ng in iter_ngrams(padded, self.n):
            hist, w = ng[:-1], ng[-1]
            p = self.prob_mle(hist, w)
            if p == 0.0:
                return float("-inf")
            import math
            logp += math.log(p)
        return logp
