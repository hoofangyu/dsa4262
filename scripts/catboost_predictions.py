import pandas as pd
import argparse
import os
from catboost import CatBoostClassifier


def generate_predictions(testing_path, model_path):

    test = pd.read_parquet(testing_path)

    test["seq_1"] = test["seq"].apply(lambda x: x[0:5])
    test["seq_2"] = test["seq"].apply(lambda x: x[1:6])
    test["seq_3"] = test["seq"].apply(lambda x: x[2:7])

    x_test = test.drop(columns=["transcript_id", "transcript_position", "seq"])

    print("Loading Model...")
    cb = CatBoostClassifier()
    cb.load_model(model_path)
    print("Model loaded")

    print("Generating Predictions on Test Set...")
    df_final = test[["transcript_id", "transcript_position"]].copy()
    predictions = cb.predict_proba(x_test)[:, 1]
    df_final["score"] = predictions

    return df_final


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("testing_path", type=str, help="Path to the testing file")
    parser.add_argument("model_path", type=str, help="Path to the model file")
    parser.add_argument("output_name", type=str, help="Name of the output file")

    args = parser.parse_args()

    testing_path = args.testing_path
    model_path = args.model_path
    output_name = args.output_name

    print("Generate Predictions")
    df = generate_predictions(testing_path, model_path)

    os.makedirs("output", exist_ok=True)
    output_path = f"output/{output_name}_results.csv"
    output_path_parquet = f"output/{output_name}_results.parquet"
    df.to_csv(output_path)
    df.to_parquet(output_path_parquet)
    print(f"Processing complete, dataset saved to {output_path}")


if __name__ == "__main__":
    main()
