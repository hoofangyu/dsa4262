import json
import gzip
import os
import pandas as pd
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

def generate_features(array):
    """
    Generates statistical features from a given 2D array, including mean, median,
    maximum, minimum, and standard deviation values.

    Parameters
    ----------
    array : np.ndarray
        A 2D NumPy array of numerical data.

    Returns
    -------
    concatenated_array : np.ndArray
        A 1D array that contains the concatenated mean, median, max, min, and
        standard deviation of the input array along the columns.
    """
    mean_array = np.mean(array, axis=0)
    median_array = np.median(array, axis=0)
    max_array = np.max(array, axis=0)
    min_array = np.min(array, axis=0)
    sd_array = np.std(array, axis=0)
    concatenated_array = np.concatenate(
        (mean_array, median_array, max_array, min_array, sd_array)
    )
    return concatenated_array


def cluster_samples(array):
    """
    Performs KMeans clustering on the input data, grouping it into
    two clusters.

    Parameters
    ----------
    array : np.ndarray
        A 2D NumPy array of numerical data.

    Returns
    -------
    cluster_1 : np.ndArray
        Subset of the input array corresponding to the first cluster with labels == 0.

    cluster_2 : np.ndArray
        Subset of the input array corresponding to the second cluster with labels == 1.
    """
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(array)
    labels = kmeans.labels_
    cluster_1 = array[labels == 0]
    cluster_2 = array[labels == 1]
    if len(cluster_2) == 0:
        return cluster_1, cluster_1
    return cluster_1, cluster_2


def parse_row(row: dict) -> list:
    """
    Parses a single JSON row of hierarchical data, extracts arrays, and
    generates feature sets for the whole data and for each cluster.

    Parameters
    ----------
    row : dict
        A dictionary containing hierarchical data, where the values contain numerical arrays.

    Returns
    -------
    ls : list
        A list containing the parsed keys, followed by the generated features
        for the entire array and two KMeans clusters.
    """
    ls = []
    for key, value in row.items():
        ls.append(key)
        for key1, value1 in value.items():
            ls.append(key1)
            for key2, value2 in value1.items():
                ls.append(key2)
                
                if len(value2) == 0:
                    print("Invalid Row")
                    ls += [None] * 135
                    return ls

                array = np.array(value2)
                whole_set = generate_features(array).tolist()
                if len(array) >= 2:
                    cluster_1, cluster_2 = cluster_samples(array)
                    cluster_1_set = generate_features(cluster_1).tolist()
                    cluster_2_set = generate_features(cluster_2).tolist()
                else:
                    cluster_1_set = whole_set
                    cluster_2_set = whole_set
                ls += whole_set + cluster_1_set + cluster_2_set
                
    return ls


def process_line(index: int, line: str, size: int) -> list:
    """
    Processes a single line from a JSON file and returns a generated feature set.

    Parameters
    ----------
    index : int
        The index of the current line being processed.

    line : str
        A string containing a single line of JSON data.

    size : int
        Total number of lines of a JSON file

    Returns
    -------
    parsed_row : list
        The list of extracted features from the parsed JSON row.
    """
    row = json.loads(line)
    parsed_row = parse_row(row)
    print(f"Processed line {index + 1}/{size}")
    return parsed_row


def parse_json(json_path: str) -> pd.DataFrame:
    """
    Processes a single line from a JSON or gzipped JSON (.json.gz) file and returns a generated feature set.

    Parameters
    ----------
     json_path : str
        Path to the JSON or gzipped JSON (.json.gz) file that contains the input data.

    csv_path : str
        Path to the CSV file containing labels to be merged with the parsed data.

    Returns
    -------
    df_with_labels : pd.DataFrame
        A pandas DataFrame that includes the processed feature data merged with the label information from the CSV.
    """
    if json_path.endswith(".gz"):
        with gzip.open(json_path, "rt") as f:
            lines = f.readlines()
    elif json_path.endswith(".json"):
        with open(json_path, "r") as f:
            lines = f.readlines()
    else:
        raise ValueError("File format not supported. Please provide a .json or .json.gz file.")

    size = len(lines)
    columns = ["transcript_id", "transcript_position", "seq"]
    values = [
        "dt_1",
        "sd_1",
        "curr_1",
        "dt_2",
        "sd_2",
        "curr_2",
        "dt_3",
        "sd_3",
        "curr_3",
    ]

    for data_range in ["whole", "cluster_1", "cluster_2"]:
        for aggregate in ["mean", "median", "max", "min", "sd"]:
            for val in values:
                columns.append(f"{data_range}_{aggregate}_{val}")

    parsed_rows = []

    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(process_line, i, line, size): i for i, line in enumerate(lines)
        }
        for future in as_completed(future_to_index):
            parsed_row = future.result()
            parsed_rows.append(parsed_row)

    df = pd.DataFrame(parsed_rows, columns=columns)

    df["transcript_position"] = df["transcript_position"].astype(int)
    df = df.sort_values(by=["transcript_id", "transcript_position"])

    print(f"{len(df)} entries created for testing")
    # print(df.columns)

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", type=str, help="Path to the dataset file")
    parser.add_argument("output_name", type=str, help="Name of the output file")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_name = args.output_name

    print("Processing Test Set")
    df = parse_json(dataset_path)

    os.makedirs("data", exist_ok=True)
    output_path = f"data/{output_name}.parquet"
    df.to_parquet(output_path)
    df.to_csv(f"data/{output_name}.csv")
    print(f"Processing complete, dataset saved to {output_path}")


if __name__ == "__main__":
    main()
