# Getting Started
This repository is organised as follows:

```
root
├── .github        # GitHub configuration files (e.g., workflows for CI/CD)
├── scripts        # Main scripts for data processing and model training
├── model          # Stored trained models
├── data           # Input raw data sets (in JSON)
├── output         # Results in CSV format from model predictions
└── tests          # Unit tests for scripts
```

## Main Flow
The main workflow consists of data processing, model training, and generating predictions. The component diagram below provides a high-level view:

![flow diagram](.github/assets/main_flow.png)


## Installation
1. Clone the repo
```bash
git clone https://github.com/hoofangyu/dsa4262.git 
```
2. Move into dsa4262 directory
```bash
cd dsa4262
```
3. Create virtual environment (optional)
```bash
sudo apt install python3.8-venv
python3 -m venv <name_of_env>
```
4. Install required packages
```bash
sudo apt install python3-pip
python3 -m pip install -r requirements.txt
```

# Usage
## Using our Pre-Trained Model (Follow steps here for DSA4262)
By using our pre-trained model, the workflow will consist only the data processing and prediction generation steps. Here is the high-level view:

![flow diagram](.github/assets/usage_flow.png)

1. Move or download the testset directly to /data folder
2. Parse testset
```bash
python3 scripts/parse_testset.py <dataset_path> <output_file_name>
```

3. Run predition
```bash
python3 scripts/catboost_predictions.py <testing_path> <model_path> <output_name>
```

### Example Usage
1. Download public testset to /data folder
```bash
aws s3 cp --no-sign-request s3://sg-nex-data/data/processed_data/m6Anet/SGNex_A549_directRNA_replicate5_run1/data.json data/
```
2. gzip file with increased buffer size
```bash
buffer -s 100000 -m 10000000 -p 100 < data/data.json | gzip > data/data.json.gz          
``` 
3. Parse testset
```bash
python3 scripts/parse_testset.py data/data.json.gz eval
```
4. Run prediction
```bash
python3 scripts/catboost_predictions.py data/eval.parquet models/final_catboost_model.cbm dataset1_final_catboost_model_results
```
