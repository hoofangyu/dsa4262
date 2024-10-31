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

2. Install required packages
```bash
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
1. Download online testset to /data folder
```bash
```
3. Parse testset
```bash
python3 scripts/parse_testset.py data/dataset1.json.gz eval
```
3. Run prediction
```bash
python3 scripts/catboost_predictions.py data/eval.parquet models/final_catboost_model.cbm dataset1_final_catboost_model_results
```
