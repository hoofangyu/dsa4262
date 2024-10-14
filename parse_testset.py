import json
import sys
import os
import gzip
import pandas as pd
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans

def generate_features(array):
    mean_array = np.mean(array, axis=0)
    median_array = np.median(array, axis=0)
    max_array = np.max(array, axis=0)
    min_array = np.min(array, axis=0)
    sd_array = np.std(array, axis=0)
    concatenated_array = np.concatenate((mean_array, median_array, max_array, min_array, sd_array))
    return concatenated_array

def cluster_samples(array):
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(array)
    labels = kmeans.labels_
    cluster_1 = array[labels == 0]
    cluster_2 = array[labels == 1]
    return cluster_1, cluster_2

def parse_row(row: dict) -> list:
    ls = []
    for key,value in row.items():
        ls.append(key)
        for key1, value1 in value.items():
            ls.append(key1)
            for key2, value2 in value1.items():
                ls.append(key2)
                array = np.array(value2)
                cluster_1, cluster_2 = cluster_samples(array)
                
                whole_set = generate_features(array).tolist()
                cluster_1_set = generate_features(cluster_1).tolist()
                cluster_2_set = generate_features(cluster_2).tolist()
                ls += whole_set + cluster_1_set + cluster_2_set
    return ls

def process_line(index: int, line: str) -> list:
    row = json.loads(line)
    parsed_row = parse_row(row)
    print(f"Processed line {index + 1}")
    return parsed_row

def parse_json(json_path: str, features_path: str) -> pd.DataFrame:
    with gzip.open(json_path, 'rt') as f:
        lines = f.readlines()

    with open(features_path, "r") as file:
        final_columns = json.load(file)
    print(f"{len(final_columns)} features loaded!")

    # Create new dimensions
    columns = ["transcript_id", "transcript_position", "seq"]
    values = ["dt_1", "sd_1", "curr_1", "dt_2", "sd_2", "curr_2", "dt_3", "sd_3", "curr_3"]

    for data_range in ["whole", "cluster_1", "cluster_2"]:
        for aggregate in ["mean","median","max","min","sd"]:
            for val in values:
                columns.append(f"{data_range}_{aggregate}_{val}")

    parsed_rows = []  

    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(process_line, i, line): i for i, line in enumerate(lines)}
        for future in as_completed(future_to_index):
            parsed_row = future.result()
            parsed_rows.append(parsed_row)

    df = pd.DataFrame(parsed_rows, columns = columns)
    
    df["transcript_position"] = df["transcript_position"].astype(int)
    df = df.sort_values(by = ["transcript_id","transcript_position"])

    df_reduced = df[["transcript_id", "transcript_position"] + final_columns]
    print(f"{len(df_reduced)} entries created for testing")

    return df_reduced

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataset_path', type=str, help='Path to the dataset file')
    parser.add_argument('features_path', type=str, help='Path to features')
    parser.add_argument('output_name', type=str, help='Name of the output file')

    args = parser.parse_args()

    dataset_path = args.dataset_path
    features_path = args.features_path
    output_name = args.output_name

    print("Processing Test Set")
    df = parse_json(dataset_path, features_path)

    output_path = f"data/{output_name}.parquet"
    df.to_parquet(output_path)
    df.to_csv(f"data/{output_name}.csv")
    print(f"Processing complete, dataset saved to {output_path}")
    
if __name__ == "__main__":
    main()