import random
from typing import List, Tuple

def sample_word(words: List[str], probs: List[float]) -> str:
    """
    Randomly sample one word based on its probability. Each word has a chance proportional to its probability in 'probs'.
    """
    r = random.random()
    cumulative = 0.0
    for word, p in zip(words, probs):
        cumulative += p
        if r <= cumulative:
            return word
    # fallback in case of rounding issues
    return words[-1]


def next_word(model, history: Tuple[str, str], vocab: List[str]) -> str:
    """
    Predict the next word given a 2-word history (for trigrams). Uses model.prob() to get probabilities for all possible next words.
    """
    probs = []
    for w in vocab:
        p = model.prob(history, w)
        probs.append(p)

    total = sum(probs)
    if total == 0.0:
        # If all probabilities are zero, pick a random word
        return random.choice(vocab)

    # Normalize to make probabilities sum to 1
    normalized = [p / total for p in probs]

    # Sample one word according to the probability distribution
    return sample_word(vocab, normalized)


def generate_sentence(model, vocab: List[str], max_len: int = 25) -> List[str]:
    """
    Generate a sentence using a trained trigram model. Starts from <s> <s> and keeps sampling next words until </s> or max_len.
    """
    sentence = []
    history = ("<s>", "<s>")  # Start with two BOS tokens

    for _ in range(max_len):
        word = next_word(model, history, vocab)
        if word == "</s>":
            break
        sentence.append(word)
        history = (history[1], word)  # move window

    if not sentence:
        return ["<empty>"]
    return sentence


def run_generation(model, vocab_set, num_sentences: int = 5):
    """
    Generate and print multiple sentences using the model.
    """
    vocab = sorted(vocab_set)
    print("\n Generated Sentences")
    for i in range(num_sentences):
        s = generate_sentence(model, vocab)
        print(f"{i+1}: {' '.join(s)}")
