# N-Gram Language Modeling

Implementation and evaluation of **n-gram language models (1-gram to 4-gram)** on the **Penn Treebank (PTB)** dataset, including both unsmoothed and smoothed variants for perplexity-based comparison and sentence generation.

This project was developed as an **individual NLP project** and focuses on classical language modeling, smoothing techniques, and the tradeoffs between simple probabilistic models and generalization on unseen text.

## Overview

Language modeling is a foundational problem in natural language processing. In this project, I implemented and evaluated **n-gram language models** ranging from unigram to 4-gram models using the Penn Treebank dataset.

The project compares:
- **unsmoothed maximum likelihood estimation (MLE)** models
- **Laplace (add-1) smoothing**
- **Linear Interpolation**
- **Stupid Backoff**

Performance is evaluated using **perplexity** on validation and test sets, and the best-performing models are also used for **sentence generation**.

## Why This Project Matters

Although modern NLP often relies on deep learning and transformers, n-gram models remain important for understanding:
- how language probability is estimated
- why sparsity becomes a challenge in text data
- how smoothing methods improve robustness
- the foundations that inspired later neural language models

This project demonstrates strong NLP fundamentals and algorithmic implementation from first principles.

## Features

- Implementation of **1-gram through 4-gram** language models
- Support for **unsmoothed MLE** models
- Support for **Laplace smoothing**
- Support for **Linear Interpolation**
- Support for **Stupid Backoff**
- **Perplexity-based evaluation** on validation and test sets
- **Sentence generation** from the best-performing model
- Automatic saving of comparison results as CSV and plot outputs

## Methodology

### 1. Data Preprocessing

The project uses the **Penn Treebank (PTB)** dataset and preprocesses the train, validation, and test splits for n-gram modeling.

### 2. N-Gram Model Construction

The system builds language models from:
- **1-gram**
- **2-gram**
- **3-gram**
- **4-gram**

Each model estimates token probabilities based on observed counts in the training corpus.

### 3. Smoothing and Backoff

To address sparse data and unseen n-grams, the project evaluates multiple strategies:
- **Laplace smoothing**
- **Linear Interpolation**
- **Stupid Backoff**

These methods help improve generalization compared to raw MLE, especially for higher-order models.

### 4. Evaluation

Models are evaluated using **perplexity** on validation and test datasets.

The implementation also generates:
- a CSV file with all perplexity scores
- a comparison plot of model performance
- sample generated sentences from the best model

## Tech Stack

- **Python**
- **NumPy**
- **Matplotlib**

## How to Run

### Requirements

- Python 3.8+
- `numpy`
- `matplotlib`

### Install Dependencies

```bash
pip install numpy matplotlib
```

### Run the Project

```bash
python main.py
```

Running the script will:
1. load and preprocess the PTB dataset
2. train and evaluate MLE models for n = 1 to 4
3. train and evaluate smoothed/backoff variants
4. save results to CSV and plot files
5. print sample generated sentences from the best model

## Expected Output

The project generates:

- `results/perplexities.csv` — perplexity scores for each model and split
- `results/perplexities.png` — bar chart comparing validation and test perplexities
- console output with generated sample sentences

## Key Learning Outcomes

Through this project, I strengthened my understanding of:
- probabilistic language modeling
- data sparsity in NLP
- smoothing and backoff methods
- perplexity as an evaluation metric
- the relationship between model complexity and generalization

## Notes

- Infinite perplexity for higher-order unsmoothed MLE models is expected when unseen n-grams appear in evaluation data.
- Smoothed and backoff methods generally perform better because they assign non-zero probability to unseen events.
- Sentence generation provides a useful qualitative view of model behavior in addition to perplexity scores.

## Future Improvements

Potential next steps include:
- adding **Kneser-Ney smoothing**
- comparing results against neural language models
- adding more detailed preprocessing experiments
- evaluating generated text quality more systematically
- packaging the project as a reusable NLP module
