import json
import sys
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parse Json
def parse_row(row):
    ls = []

    for key,value in row.items():
        ls.append(key)
        for key1, value1 in value.items():
            ls.append(key1)
            for key2, value2 in value1.items():
                ls.append(key2)
                array = np.array(value2)
                mean_array = np.mean(array, axis=0)
                mean_list = mean_array.tolist()
                ls += mean_list

    return ls

def process_line(index, line):
    row = json.loads(line)
    parsed_row = parse_row(row)
    print(f"Processed line {index + 1}")
    return parsed_row

def parse_json(json_path,csv_path):
    with open(json_path) as f:
        lines = f.readlines()

    parsed_rows = []  

    with ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(process_line, i, line): i for i, line in enumerate(lines)}
        for future in as_completed(future_to_index):
            parsed_row = future.result()
            parsed_rows.append(parsed_row)

    df = pd.DataFrame(parsed_rows, columns=["transcript_id", "transcript_position", "seq", 
                                            "dt_1", "sd_1", "curr_1", 
                                            "dt_2", "sd_2", "curr_2", 
                                            "dt_3", "sd_3", "curr_3"])
    
    df["transcript_position"] = df["transcript_position"].astype(int)
    df = df.sort_values(by = ["transcript_id","transcript_position"])

    labels = pd.read_csv(csv_path)
    df_with_labels = df.merge(labels, on = ["transcript_id","transcript_position"])

    return df_with_labels

json_file_path = sys.argv[1]
csv_file_path = sys.argv[2]

# Extract file names (without directory paths) and remove extensions
json_file_name = os.path.splitext(os.path.basename(json_file_path))[0]
csv_file_name = os.path.splitext(os.path.basename(csv_file_path))[0]

df = parse_json(json_file_path, csv_file_path)
df.to_parquet(f"/data/{json_file_name}_{csv_file_name}.parquet")

print(f"Dataframe successfully saved to /data/{json_file_name}_{csv_file_name}.parquet")
