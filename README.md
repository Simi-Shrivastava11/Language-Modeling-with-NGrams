# N-Gram Language Modeling

This project implements and evaluates n-gram language models (1-gram to 4-gram) on the Penn Treebank (PTB) dataset.  
It includes unsmoothed MLE and smoothed/backoff models (Laplace add-1, Linear Interpolation, Stupid Backoff).  
Perplexity is reported on validation and test sets, and a comparison plot is saved.

## Project Structure
**NGramModel/**
- **data/**
  - `ptb.train.txt`
  - `ptb.valid.txt`
  - `ptb.test.txt`
- **src/**
  - `preprocess.py`
  - `ngram_model.py`
  - `smoothing.py`
  - `evaluate.py`
  - `generate.py`
- **results/** (created automatically)
  - `perplexities.csv`
  - `perplexities.png`
- `main.py`
- `README.md`
- `NGram_Model_Report.pdf`

## Requirements

- Python 3.8 or newer  
- `matplotlib`  
- `numpy`

Install dependencies:
```bash
pip install matplotlib numpy
```
## How to run

Run with the following command:
```bash
python main.py
```

This will:

1. Load and preprocess the PTB dataset
2. Train and evaluate MLE models for n = 1 – 4
3. Train and evaluate smoothed/backoff models (Laplace, Interpolation, Backoff)
4. Save:
    - results/perplexities.csv – all scores
    - results/perplexities.png – bar chart comparison
5. Print sample generated sentences from the best model

## Output

**CSV:** results/perplexities.csv — model, split, perplexity

**Plot:** results/perplexities.png — validation/test comparison

**Console:** generated sentences printed from the best model

## Notes

- Infinite perplexity for higher-order unsmoothed MLE models is expected because of unseen n-grams.
- Smoothed and backoff models, especially Stupid Backoff, generally achieve the lowest perplexities and most fluent generations.
- The results folder is automatically created if it does not exist.

