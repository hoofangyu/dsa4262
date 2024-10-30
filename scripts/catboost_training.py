import pandas as pd
import argparse

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
)
from catboost import CatBoostClassifier

def generate_predictions(training_path):

    train = pd.read_parquet(training_path)

    train["seq_1"] = train["seq"].apply(lambda x: x[0:5])
    train["seq_2"] = train["seq"].apply(lambda x: x[1:6])
    train["seq_3"] = train["seq"].apply(lambda x: x[2:7])

    x_train = train.drop(
        columns=["transcript_id", "transcript_position", "seq", "gene_id", "label"]
    )
    y_train = train[["label"]]

    # Fitting Catboost
    print("Train Dataset Loaded, Begin Model Training...")
    categorical_features = [135, 136, 137]
    cb = CatBoostClassifier()
    cb.fit(x_train, y_train, cat_features=categorical_features)
    print("Model Trained")

    print(
        f"train roc auc: {round(roc_auc_score(y_train, cb.predict_proba(x_train)[:,1]),4)}"
    )
    precision, recall, thresholds = precision_recall_curve(
        y_train, cb.predict_proba(x_train)[:, 1]
    )
    print(f"train pr auc: {round(auc(recall, precision),4)}")

    return cb


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("training_path", type=str, help="Path to the training file")
    parser.add_argument("output_name", type=str, help="Name of the output file")

    args = parser.parse_args()

    training_path = args.training_path
    output_name = args.output_name

    print("Generate Predictions")
    cb = generate_predictions(training_path)

    output_path = f"models/{output_name}.cbm"
    cb.save_model(output_path)
    print(f"Training complete, model saved to {output_path}")


if __name__ == "__main__":
    main()
