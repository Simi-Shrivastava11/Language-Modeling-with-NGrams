from src.preprocess import load_and_preprocess, corpus_stats
from src.ngram_model import NgramMLE
from src.evaluate import perplexity
from src.smoothing import LaplaceTrigram, InterpolatedTrigram, StupidBackoffTrigram, tune_interpolation_lambdas
from src.generate import run_generation
import csv
from pathlib import Path
import math
import matplotlib.pyplot as plt

def save_results_table(results: dict, out_csv: str = "results/perplexities.csv") -> None:
    """
    Write a table with columns: model, split, perplexity. Uses 'inf' text for infinite values to keep the CSV readable.
    """
    Path("results").mkdir(exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "perplexity"])
        for name, vals in results.items():
            for split in ["valid", "test"]:
                pp = vals[split]
                w.writerow([name, split, "inf" if pp == float("inf") else f"{pp:.6f}"])

def plot_results(results, out_png="results/perplexities.png"):

    labels = list(results.keys())
    valid_vals = [results[k]["valid"] for k in labels]
    test_vals  = [results[k]["test"] for k in labels]
    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))

    valid_plot = [0 if v == float("inf") else v for v in valid_vals]
    test_plot  = [0 if v == float("inf") else v for v in test_vals]

    bars1 = plt.bar([i - width/2 for i in x], valid_plot, width, label="valid")
    bars2 = plt.bar([i + width/2 for i in x], test_plot, width, label="test")

    # Add INF labels for missing bars
    for i, v in enumerate(valid_vals):
        if v == float("inf"):
            plt.text(i - width/2, 50, "INF", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(test_vals):
        if v == float("inf"):
            plt.text(i + width/2, 50, "INF", ha="center", va="bottom", fontsize=9)

    # Add numeric labels above visible bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 30,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.xticks(list(x), labels, rotation=25)
    plt.ylabel("Perplexity (lower is better)")
    plt.title("Language model perplexities")
    plt.legend()
    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def evaluate_mle(train, valid, test):
    """
    Train unsmoothed MLE n-gram models (n=1,2,3,4) and print perplexity. If any zero-probability occurs, perplexity is reported as INF.
    """
    print("\n Perplexity (unsmoothed MLE)")
    results = {}

    for n in [1, 2, 3, 4]:
        model = NgramMLE(n=n)
        model.fit(train)

        pp_valid = perplexity(model, valid)
        pp_test = perplexity(model, test)

        pp_valid_str = f"{pp_valid:.3f}" if pp_valid != float("inf") else "INF"
        pp_test_str = f"{pp_test:.3f}" if pp_test != float("inf") else "INF"
        print(f"{n}-gram:  valid = {pp_valid_str}   test = {pp_test_str}")

        results[f"MLE-{n}"] = {"valid": pp_valid, "test": pp_test}

    return results

def smoothing_and_backoff(train, valid, test):
    """
    Train and evaluate smoothed/backoff trigram models. Returns the results and the best model for generation.
    """
    results = {}

    # Add-1 (Laplace) 
    print("\n Trigram + Add-1 (Laplace)")
    laplace = LaplaceTrigram()
    laplace.fit(train)
    pp_valid = perplexity(laplace, valid)
    pp_test = perplexity(laplace, test)
    print(f"valid = {pp_valid:.3f}   test = {pp_test:.3f}")
    results["Laplace-3"] = {"valid": pp_valid, "test": pp_test}

    # Linear Interpolation 
    print("\n Trigram + Linear Interpolation")
    best_lambdas, best_valid_pp = tune_interpolation_lambdas(train, valid)
    print(f"Best lambdas (from validation): {best_lambdas}")
    print(f"Validation perplexity = {best_valid_pp:.3f}")
    interp = InterpolatedTrigram(best_lambdas)
    interp.fit(train)
    pp_valid = perplexity(interp, valid)
    pp_test = perplexity(interp, test)
    print(f"valid = {pp_valid:.3f}   test = {pp_test:.3f}")
    results["Interpolated-3"] = {"valid": pp_valid, "test": pp_test}

    # Stupid Backoff 
    print("\n Trigram + Stupid Backoff")
    backoff = StupidBackoffTrigram(backoff_factor=0.4)
    backoff.fit(train)
    pp_valid = perplexity(backoff, valid)
    pp_test = perplexity(backoff, test)
    print(f"valid = {pp_valid:.3f}   test = {pp_test:.3f}")
    results["Backoff-3"] = {"valid": pp_valid, "test": pp_test}

    # Return results and best model (lowest perplexity)
    return results, backoff


def main():
    """
    1) Load and preprocess PTB data.
    2) Print basic corpus statistics.
    3) Evaluate unsmoothed MLE models (n=1..4).
    4) Evaluate smoothed and backoff trigram models.
    5) Generate sentences from the best model.
    """
    train, valid, test, vocab, _ = load_and_preprocess("data", min_freq=1)

    print("Vocabulary size:", len(vocab))
    print("Train  (sentences, tokens, avg_len):", corpus_stats(train))
    print("Valid  (sentences, tokens, avg_len):", corpus_stats(valid))
    print("Test   (sentences, tokens, avg_len):", corpus_stats(test))
    print("Sample train sentence:", " ".join(train[0][:30]))

    mle_results = evaluate_mle(train, valid, test)
    smooth_results, best_model = smoothing_and_backoff(train, valid, test)
    

    # Combine and print summary
    print("\n Summary of Perplexities")
    all_results = {**mle_results, **smooth_results}
    for name, vals in all_results.items():
        print(f"{name:<15}   valid = {vals['valid']:.3f}   test = {vals['test']:.3f}")

    # Combine all results
    all_results = {**mle_results, **smooth_results}

    # Save CSV and plot PNG
    save_results_table(all_results)
    plot_results(all_results)

    # Generate sample sentences from the best model
    run_generation(best_model, vocab)

if __name__ == "__main__":
    main()
