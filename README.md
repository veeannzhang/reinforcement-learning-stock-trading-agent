# Project Pipeline Overview

This repository contains a series of notebooks that run an end-to-end pipeline, including steps for data ingestion, features engineering, model training, performance evaluation, and model diagnostics.

---

## Basic setup and configuration
- Download the repository.
- Install the libraries using the requirements.txt file.
- Set the absolute path to the data folder in the .env file (DATA_DIR)
- Set the FRED API key in the .env file (FRED_API_KEY). This is required to run the data ingestion step.
- Run the notebooks in the notebooks folder in the order that they are listed, following the instructions and comments within the cells of each notebook.

## pipeline_01_data_ingestion.ipynb
- This notebook contains code to download data from external databases using APIs.

## pipeline_02_data_preprocessing.ipynb
This notebook serves the following purposes:
- Visually examine data timeseries
- Trim starting / ending dates of data, and discard series with too little data
- Map data to the full range of dates and imputes missing data
- Resample series of different frequencies and merge

## pipeline_03_features_engineering.ipynb
- This notebook engineers predictive features on the preprocessed dataset.

## pipeline_04_validation_and_analysis.ipynb
- This notebook performs essential validations on the final dataset. 
- Visual and correlation anaylses are also performed to select the final set of features for model training.

## pipeline_05_train_base_model.ipynb
- This notebook trains the base agent (Reccurent PPO + LSTM with MLP feature encoder). 
- NOTE: This notebook must be run in a Google Colab environment with a GPU compute.

## pipeline_06_train_mamba_model.ipynb
- This notebook trains the Mamba agent (PPO with Mamba feature encoder). 
- NOTE: This notebook must be run in a Google Colab environment with a GPU compute.

## pipeline_07_evaluate_naive_policy.ipynb
- This notebook evaluates the naive equal-allocation, buy-and-hold policy on the test set using a set of financial metrics.

## pipeline_08_evaluate_trading_agents.ipynb
- This notebook evaluates the performance of the trained RL agents on the test set using a set of financial metrics.

## pipeline_09_training_diagnostics.ipynb
- This notebook performs qualitative diagnostics of the learning process by visualizing statistics accumulated during training.

## pipeline_10_base_mode_VIX_Study.ipynb
- Runs VIX analysis for the base model.

## pipeline_11_Mamba_PPO_VIX_Study.ipynb
- Runs VIX analysis for the Mamba model.