import pandas as pd
import argparse
import os
from catboost import CatBoostClassifier


def generate_predictions(testing_path, model_path):

    test = pd.read_parquet(testing_path)
    na_row_count = test.isna().any(axis=1).sum()
    test = test.dropna()

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

    print(f"Number of Invalid Rows: {na_row_count}")

    return df_final


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("testing_path", type=str, help="Path to the testing file")
    parser.add_argument("model_path", type=str, help="Path to the model file")
    parser.add_argument("output_name", type=str, help="Name of the output file")
    parser.add_argument("--parquet", action="store_true", help="Save output as Parquet instead of CSV")

    args = parser.parse_args()

    testing_path = args.testing_path
    model_path = args.model_path
    output_name = args.output_name
    save_as_parquet = args.parquet

    print("Generate Predictions")
    df = generate_predictions(testing_path, model_path)

    os.makedirs("output", exist_ok=True)
    if save_as_parquet:
        output_path = f"output/{output_name}_results.parquet"
        df.to_parquet(output_path)
    else:
        output_path = f"output/{output_name}_results.csv"
        df.to_csv(output_path)
        
    print(f"Processing complete, dataset saved to {output_path}")


if __name__ == "__main__":
    main()
