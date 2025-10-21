# src/evaluate.py

import math
from typing import List

def sentence_logprob(model, sent: List[str]) -> float:
    """
    Calculate the log probability of one sentence using the given model.

    It calls the model's method (sentence_logprob_mle) that goes through each word in the sentence and sums the log of its probability
    given its history (previous words).

    Returns:
        The total log probability (a negative number). If any word has zero probability, returns -inf.
    """
    return model.sentence_logprob(sent)


def corpus_logprob(model, corpus: List[List[str]]) -> float:
    """
    Add up log probabilities for all sentences in the corpus. If any sentence has zero probability (log = -inf), return -inf.
    """
    total = 0.0
    for sent in corpus:
        lp = sentence_logprob(model, sent)
        if lp == float("-inf"):
            return float("-inf")
        total += lp
    return total


def perplexity(model, corpus: List[List[str]]) -> float:
    """
    Compute perplexity for a whole corpus.

    Perplexity = exp( - (total log probability) / (total number of tokens) )

    A lower perplexity means the model predicts the data better. If the model assigns zero probability to any word, perplexity is infinite.
    """
    total_tokens = sum(len(s) for s in corpus)
    if total_tokens == 0:
        return float("inf")

    total_lp = corpus_logprob(model, corpus)
    if total_lp == float("-inf"):
        return float("inf")

    return math.exp(- total_lp / total_tokens)
