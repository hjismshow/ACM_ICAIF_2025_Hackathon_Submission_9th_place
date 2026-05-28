import os, sys, json, time, warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
import pickle
from tqdm.auto import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost.callback import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from typing import Dict, Any
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Has local numpy shadowing?", os.path.exists("./numpy") or os.path.exists("./numpy.py"))
print("xgboost:", xgb.__version__)
############### Model Class ############### 
class MyModel:
    def __init__(self, models: Dict[str, Any]):
      # Example: {"target_1": model1, "target_2": model2, ...}
      self.models = models
    def Preprocess_before_predictions(self,df: pd.DataFrame):
      pivoted = (
          df
          .pivot(index="window_id", columns="time_step", values=["close", "volume"])
          .sort_index(axis=1, level=1)
      )
      pivoted = pivoted.reindex(columns=["close", "volume"], level=0)
      # Flatten columns — rename 'volume' → 'vol'
      pivoted.columns = [
          f"timestamp_{t+1}_{'vol' if var == 'volume' else var}"
          for var, t in pivoted.columns
      ]
      #Call feature engineering function
      return (pivoted.reset_index())
    def feature_engineering(self,df: pd.DataFrame):
      #### Rate of change ####
      roc_Feature_list=[]
      var_list=['vol','close']
      time_frame=['2','5','10','15','30','60']
      for var in var_list:
          for time in time_frame:
              roc_Feature_list.append('rate_of_change_'+var+'_L'+time+'M')
              df[f'rate_of_change_{var}_L{time}M']= np.where(df[f'timestamp_{time}_{var}'] == 0, 0,(df[f'timestamp_1_{var}'] - df[f'timestamp_{time}_{var}']) / df[f'timestamp_{time}_{var}'])
      print("ROC Features:",roc_Feature_list)

      #### Statistical Features ####
      stat_Feature_list = []
      var_list = ['vol', 'close']
      for var in var_list:
          for t in range(5,65,5):
                  # Find the last `t` columns that match the variable name pattern
              cols = ["timestamp_"+str(60-c)+"_"+var for c in range(t)]
              df[f'rolling_mean_{var}_{t}'] = df[cols].mean(axis=1)
              stat_Feature_list.append(f'rolling_mean_{var}_{t}')
              df[f'rolling_var_{var}_{t}'] = df[cols].std(axis=1)
              stat_Feature_list.append(f'rolling_var_{var}_{t}')
      print("Statistical Features:",stat_Feature_list)
      #### Slope Features ####
      slope_Feature_list=[]
      for slope in [15,30,45,60]:
          cols_vol=["timestamp_"+str(60-c)+"_"+"vol" for c in range(slope)]
          cols_close=["timestamp_"+str(60-c)+"_"+"close" for c in range(slope)]
          #(df[cols] - df[cols].mean()).prod(axis=1)
          X = df[cols_vol].to_numpy()
          Y = df[cols_close].to_numpy()
          # Subtract mean (centered regression)
          X_centered = X - X.mean(axis=1, keepdims=True)
          Y_centered = Y - Y.mean(axis=1, keepdims=True)
          # Compute slope per row:  Σ(x*y) / Σ(x²)
          slopes = (X_centered * Y_centered).sum(axis=1) / (X_centered**2).sum(axis=1)
          df[f'slope_{slope}'] = slopes
          slope_Feature_list.append('slope_'+str(slope))
      print("Slope Features:",slope_Feature_list)
      return df

    def Extract_Predictions(self,df: pd.DataFrame):
      #df=self.Preprocess_before_predictions(df)
      y_pred = {}
      if "window_id" in df.columns:
          df_dropped = df.drop(columns=["window_id"])
          for tgt, mdl in self.models.items():
              y_pred[tgt] = mdl.predict(df_dropped)
          y_pred_df = pd.DataFrame(y_pred, index=getattr(df, "index", None))
          y_pred_df['window_id']=df['window_id']
          return  (y_pred_df)
      else:
          for tgt, mdl in self.models.items():
              y_pred[tgt] = mdl.predict(df)
          y_pred_df = pd.DataFrame(y_pred, index=getattr(df, "index", None))
          y_pred_df["window_id"] = range(1, len(y_pred_df) + 1)
          return (y_pred_df)
############### Utilities ############### 
def init_model(weights_path: str | None = "/model_weights.pkl") -> MyModel:
    # Load all trained models from pickle
    with open(weights_path, "rb") as f:
        trained_models = pickle.load(f)   # Dict of 10 models

    return MyModel(trained_models)

def Convert_in_submission_df(df: pd.DataFrame):
    y_long = df.melt(
        id_vars='window_id',
        var_name='time_step',
        value_name='pred_close'
    )
    # # # Extract numeric part from 'target_x' columns
    y_long['time_step'] = y_long['time_step'].str.extract('(\\d+)').astype(int) - 1
    # y_long = y_long.rename(columns={'index': 'window_id'})
    # y_long['window_id'] += 1  # to start IDs from 1 instead of 0
    y_long = y_long.sort_values(['window_id', 'time_step']).reset_index(drop=True)
    return y_long
def save_submission(df: pd.DataFrame,out_path: str = "/submission.pkl"):
    out_path = Path(out_path)
    df.to_pickle(out_path)
    print(f"[OK] Saved forecast to {out_path} with {len(df)} rows")
    return df
def row_minmax_scaler(df: pd.DataFrame):
    """
    Row-wise MinMax scaling with two criteria sets:
    1. Based on timestamp_*_close for close + target columns
    2. Based on timestamp_*_vol for vol columns
    """

    # Define column groups
    close_cols = [f'timestamp_{i}_close' for i in range(1, 61)]
    vol_cols   = [f'timestamp_{i}_vol' for i in range(1, 61)]
    target_cols = [f'target_{i}' for i in range(1, 11)]

    # --- Compute row-wise min/max for close and vol ---
    row_min_close = df.loc[:, close_cols].min(axis=1)
    row_max_close = df.loc[:, close_cols].max(axis=1)

    row_min_vol = df.loc[:, vol_cols].min(axis=1)
    row_max_vol = df.loc[:, vol_cols].max(axis=1)

    # --- Store scaling criteria ---
    criteria = pd.DataFrame({
        'row_min_close': row_min_close,
        'row_max_close': row_max_close,
        'row_min_vol': row_min_vol,
        'row_max_vol': row_max_vol
    }, index=df.index)

    # --- Scale ---
    df_scaled = df.copy()

    # Scale close + target columns using close min/max
    denom_close = (criteria['row_max_close'] - criteria['row_min_close']).replace(0, np.nan)
    df_scaled[close_cols] = (
        df[close_cols].sub(criteria['row_min_close'], axis=0)
        .div(denom_close, axis=0)
    )

    # Scale vol columns using vol min/max
    denom_vol = (criteria['row_max_vol'] - criteria['row_min_vol']).replace(0, np.nan)
    df_scaled[vol_cols] = (
        df[vol_cols].sub(criteria['row_min_vol'], axis=0)
        .div(denom_vol, axis=0)
    )

    # --- Optional save ---


    return df_scaled, criteria
def inverse_row_minmax(df_scaled, criteria):
    """
    Restores original close/vol/target values using stored row-wise criteria.
    """
    df_restored = df_scaled.copy()

    close_cols = [f'timestamp_{i}_close' for i in range(1, 61)]
    vol_cols   = [f'timestamp_{i}_vol' for i in range(1, 61)]
    target_cols = [f'target_{i}' for i in range(1, 11)]

    # Inverse for close + target
    df_restored[target_cols] = (
        df_scaled[target_cols]
        .mul(criteria['row_max_close'] - criteria['row_min_close'], axis=0)
        .add(criteria['row_min_close'], axis=0)
    )
    return df_restored
