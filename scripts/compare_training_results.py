import pandas as pd
import argparse
from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
)


def score_predictions(training_path, results_path):

    train = pd.read_parquet(training_path)
    results = pd.read_csv(results_path)

    y_train = train[["label"]]
    y_results = results[["score"]]

    print(f"train roc auc: {round(roc_auc_score(y_train, y_results),4)}")
    precision, recall, thresholds = precision_recall_curve(y_train, y_results)
    print(f"train pr auc: {round(auc(recall, precision),4)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "training_path", type=str, help="Path to the training file with labels"
    )
    parser.add_argument("results_path", type=str, help="Path to the results file")

    args = parser.parse_args()

    training_path = args.training_path
    results_path = args.results_path

    print("Scoring Predictions")
    score_predictions(training_path, results_path)


if __name__ == "__main__":
    main()
