from collections import Counter
from typing import List, Tuple, Set, Dict
import os

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

def read_sentences(path: str, lowercase: bool = True) -> List[List[str]]:
    """
    Reads a text file where each line is a tokenized sentence (words separated by spaces). Returns a list of sentences, where each sentence is a list      of tokens (strings).
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if lowercase:
                line = line.lower()
            lines.append(line.split())
        return lines

def build_vocab(train_sentences: List[List[str]], min_freq: int = 1) -> Tuple[Set[str], Counter]:
    """
    Builds a vocabulary from the training sentences only. Any word that appears fewer times than min_freq will be replaced by <unk>.
    """
    freq = Counter(tok for sent in train_sentences for tok in sent)
    vocab: Set[str] = {tok for tok, c in freq.items() if c >= min_freq}
    vocab.update({BOS, EOS, UNK})
    return vocab, freq

def map_to_vocab(sentences: List[List[str]], vocab: Set[str], add_eos: bool = True) -> List[List[str]]:
    """
    Goes through each sentence and replaces words not in the vocab with <unk> and adds </s> at the end of the sentence (if add_eos=True)
    """
    mapped: List[List[str]] = []
    for sent in sentences:
        s = [tok if tok in vocab else UNK for tok in sent]
        if add_eos:
            s = s + [EOS]
        mapped.append(s)
    return mapped

def load_and_preprocess(data_dir: str, min_freq: int = 1, lowercase: bool = True, add_eos: bool = True) -> Tuple[List[List[str]], List[List[str]], List[List[str]], Set[str], Counter]:
    """
    Loads the PTB dataset and preprocesses it:
      1. Reads train, validation, and test sets.
      2. Builds a vocabulary from the training data.
      3. Maps rare or unknown words in all splits to <unk>.
      4. Adds </s> to the end of each sentence.
    """
    train_path = os.path.join(data_dir, "ptb.train.txt")
    valid_path = os.path.join(data_dir, "ptb.valid.txt")
    test_path  = os.path.join(data_dir, "ptb.test.txt")

    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        raise FileNotFoundError("One or more PTB files are missing in the data directory")

    train_raw = read_sentences(train_path, lowercase=lowercase)
    valid_raw = read_sentences(valid_path, lowercase=lowercase)
    test_raw  = read_sentences(test_path,  lowercase=lowercase)

    vocab, train_freq = build_vocab(train_raw, min_freq=min_freq)

    train = map_to_vocab(train_raw, vocab, add_eos=add_eos)
    valid = map_to_vocab(valid_raw, vocab, add_eos=add_eos)
    test  = map_to_vocab(test_raw,  vocab, add_eos=add_eos)

    return train, valid, test, vocab, train_freq

def corpus_stats(sentences: List[List[str]]) -> Tuple[int, int, float]:
    """
    Returns basic statistics: number of sentences, number of tokens, and average sentence length.
    """
    n_sent = len(sentences)
    n_tok = sum(len(s) for s in sentences)
    avg_len = (n_tok / n_sent) if n_sent else 0.0
    return n_sent, n_tok, avg_len
