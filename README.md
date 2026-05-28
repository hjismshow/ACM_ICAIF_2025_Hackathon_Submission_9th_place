# ACM_ICAIF_2025_Hackathon_Submission_9th_place


> Forecasting the next 10 minutes of closing prices across 100 cryptocurrencies using engineered features and XGBoost.
---
Competition Overview
The goal was to develop robust short-horizon forecasting models for cryptocurrency markets that explicitly incorporate heterogeneous information sources. Participants were challenged to forecast the next 10 minutes of closing prices for 100 tokens, given the preceding 60 minutes of close and volume data.
Task: Given 60-minute sequences of closing prices and trading volumes, predict the next 10-minute price trajectory.
Formally:
Input: `(X_i, Z_i)` — 60-minute close and volume sequences per asset
Output: `Y_i = [p_{t+1}, ..., p_{t+10}]` — next 10 closing prices
Evaluation: Statistical accuracy + simulated trading performance
---
Approach
Pipeline Summary
```
Raw Data → Windowing → Train/Val/Holdout Split → Row-wise MinMax Scaling
→ Feature Engineering → XGBoost (one model per target step) → Inverse Scale → Submission
```
1. Windowing Strategy
Both overlapping (stride=1) and non-overlapping windows of length 70 were implemented. The final model used non-overlapping windows, splitting each 70-step window into:
Observation window: timesteps 1–60 (close + volume)
Prediction targets: timesteps 61–70 (close only, renamed `target_1` through `target_10`)
2. Train / Validation / Holdout Split
A temporal split was used to respect the time-series nature of the data. The most recent windows per token were held out for evaluation, with tokens that had insufficient data excluded from the test/holdout sets.
3. Custom Row-wise MinMax Scaling
A key design choice was a row-wise (per-window) MinMax scaler that normalizes:
Close and target columns using the min/max of that window's 60-step close sequence
Volume columns using the min/max of that window's 60-step volume sequence
Scaling criteria are saved and later used for inverse transformation at inference time.
4. Feature Engineering
Three families of features were computed on top of the raw 60-step sequences:
Feature Type	Description
Rate of Change	Price and volume ROC over lookback windows of 2, 5, 10, 15, 30, 60 minutes
Rolling Statistics	Rolling mean and standard deviation over windows of 5 to 60 minutes (step 5) for both close and volume
Price-Volume Slope	Linear regression slope of close on volume over lookback windows of 15 to 60 minutes (step 5)
5. Model: XGBoost (Multi-output via 10 Independent Models)
One `XGBRegressor` was trained per target step (`target_1` through `target_10`). Hyperparameters were tuned via Optuna:
```python
n_estimators       = 500
learning_rate      = 0.054
max_depth          = 5
min_child_weight   = 0.012
subsample          = 0.526
colsample_bytree   = 0.897
reg_alpha          = 0.003
reg_lambda         = 0.138
gamma              = 2.757
early_stopping_rounds = 150
eval_metric        = "rmse"
```
Trained models are serialized to `model_weights.pkl`.
---
Repository Structure
```
├── 9th_place_modtest.ipynb   # Full training and inference notebook
├── model_weights.pkl          # Saved XGBoost models (generated after training)
├── train_df.csv               # Scaling criteria for training set
├── validation_set.csv         # Scaling criteria for validation set
├── holdout_set.csv            # Scaling criteria for holdout set
└── README.md
```
---
Requirements
```
numpy==2.0.1
pandas==2.2.2
scikit-learn==1.5.1
xgboost==2.1.1
scipy==1.13.1
torch
tqdm
```
Install with:
```bash
pip install numpy==2.0.1 pandas==2.2.2 scikit-learn==1.5.1 xgboost==2.1.1 scipy==1.13.1 torch tqdm
```
---
How to Run
Place the competition data files in `/content/`:
`dataset_info.json`
`train.pkl`
`x_test.pkl`
`y_test_local.pkl`
Open and run `9th_place_modtest.ipynb` end-to-end. The notebook is organized into clearly numbered sections:
Libraries
File loading & data extraction
Windowing functions (overlapping / non-overlapping)
Utility functions (prediction extraction, preprocessing, submission formatting)
Preprocessing & target creation
Train / validation / holdout split
Row-wise MinMax scaling
Feature engineering
Final dataset preparation
XGBoost model training
Inference & metric evaluation
---
Results
Final leaderboard position: 9th place 🎉
Evaluation followed the competition's combined metric of statistical accuracy and simulated trading performance across all 100 cryptocurrency tokens.
---
Key Takeaways
Row-wise normalization was critical — it allows the model to learn price movement patterns independent of absolute price levels, making it transferable across tokens with very different price ranges.
A separate model per prediction horizon outperformed a single multi-output model in this setting.
Price-volume slope features provided meaningful signal beyond simple rolling statistics.
XGBoost with early stopping on a held-out validation set was effective and fast to iterate on.
