import json
import sys
import os
import gzip
import pandas as pd
import numpy as np
import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

def generate_predictions(training_path, testing_path, features_path):
    with open(features_path, "r") as file:
        final_columns = json.load(file)
    print(f"{len(final_columns)} features loaded!")

    train = pd.read_parquet(training_path)
    x_train = train[final_columns]
    y_train = train[['label']]

    # Minmax Scaling
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    
    # Fitting LGBM
    print("Train Dataset Loaded, Begin Model Training...")
    lgbm = LGBMClassifier()
    lgbm.fit(x_train_scaled, y_train)
    print("Model Trained")

    print(f'roc auc: {round(roc_auc_score(y_train, lgbm.predict_proba(x_train_scaled)[:,1]),4)}')
    precision, recall, thresholds = precision_recall_curve(y_train, lgbm.predict_proba(x_train_scaled)[:,1])
    print(f'pr auc: {round(auc(recall, precision),4)}')

    print("Generating Predictions on Test Set...")
    test = pd.read_parquet(testing_path)
    df_final = test[["transcript_id", "transcript_position"]].copy()
    x_test = test[final_columns]
    x_test_scaled = scaler.transform(x_test)
    predictions = lgbm.predict_proba(x_test_scaled)[:,1]
    df_final["score"] = predictions
    
    return df_final

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('training_path', type=str, help='Path to the training file')
    parser.add_argument('testing_path', type=str, help='Path to the testing file')
    parser.add_argument('features_path', type=str, help='Path to features')
    parser.add_argument('output_name', type=str, help='Name of the output file')

    args = parser.parse_args()

    training_path = args.training_path
    testing_path = args.testing_path
    features_path = args.features_path
    output_name = args.output_name

    print("Generate Predictions")
    df = generate_predictions(training_path, testing_path, features_path)

    output_path = f"output/{output_name}_results.csv"
    df.to_csv(output_path)
    print(f"Processing complete, dataset saved to {output_path}")
    
if __name__ == "__main__":
    main()