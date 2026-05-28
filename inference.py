import pandas as pd
from pathlib import Path
from model import init_model
import numpy as np
from model import row_minmax_scaler
from model import inverse_row_minmax
from model import Convert_in_submission_df

def generate_forecast(x_test_path: str, out_path: str = "/submission.pkl"):
  model = init_model()
  x_test = pd.read_pickle(x_test_path)
  X=model.Preprocess_before_predictions(x_test)
  X_scale,X_crit=row_minmax_scaler(X)
  X_scale=model.feature_engineering(X_scale)
  y_pred=model.Extract_Predictions(X_scale)
  y_pred=inverse_row_minmax(y_pred,X_crit)
  y_pred=Convert_in_submission_df(y_pred)
  out_path = Path(out_path)
  y_pred.to_pickle(out_path)
  print(f"[OK] Saved forecast to {out_path} with {len(y_pred)} rows")
  return y_pred
