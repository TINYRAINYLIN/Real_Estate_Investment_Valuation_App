import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
)

from src.exception import CustomException
from src.logger import logging 
import dill

# def save_object(file_path, obj):
#     try:
#         dir_path = os.path.dirname(file_path)
#         os.makedirs(dir_path, exist_ok=True)

#         with open(file_path, "wb") as file_obj:
#             dill.dump(obj, file_obj)

#     except Exception as e:
#         logging.error(f"Error occurred while saving object: {e}")
#         raise CustomException(e, sys)

# -------------------------------------------------------------------
# Object persistence
# -------------------------------------------------------------------
def save_object(file_path, obj):
    """
    Save a Python object to disk using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from disk using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info(f"Object loaded successfully from: {file_path}")
        return obj

    except Exception as e:
        logging.error(f"Error occurred while loading object: {e}")
        raise CustomException(e, sys)


# -------------------------------------------------------------------
# Regression evaluation
# -------------------------------------------------------------------
def evaluate_regression(model, X_train, y_train, X_test, y_test):
    """
    Evaluate regression model with multiple metrics.
    Logs results and returns dictionary of metrics.
    """
    try:
        logging.info(f"Evaluating model: {model.__class__.__name__}")

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        # Core metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test  = r2_score(y_test, y_pred_test)
        n, p = X_test.shape
        adj_r2_test = 1 - (1-r2_test) * (n-1)/(n-p-1)

        mae   = mean_absolute_error(y_test, y_pred_test)
        mse   = mean_squared_error(y_test, y_pred_test)
        rmse  = np.sqrt(mse)
        medae = median_absolute_error(y_test, y_pred_test)
        mape  = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

        # Threshold-based accuracy
        within_10k   = np.mean(np.abs(y_test - y_pred_test) <= 10000) * 100
        within_20k   = np.mean(np.abs(y_test - y_pred_test) <= 20000) * 100
        within_50k   = np.mean(np.abs(y_test - y_pred_test) <= 50000) * 100
        within_5pct  = np.mean(np.abs(y_test - y_pred_test) / y_test <= 0.05) * 100
        within_10pct = np.mean(np.abs(y_test - y_pred_test) / y_test <= 0.10) * 100

        metrics = {
            "R2_train": r2_train,
            "R2_test": r2_test,
            "Adj_R2_test": adj_r2_test,
            "MAE": mae,
            "MedAE": medae,
            "RMSE": rmse,
            "MAPE": mape,
            "Within_10k": within_10k,
            "Within_20k": within_20k,
            "Within_50k": within_50k,
            "Within_5pct": within_5pct,
            "Within_10pct": within_10pct,
        }

        logging.info(f"Metrics for {model.__class__.__name__}: {metrics}")
        return metrics

    except Exception as e:
        logging.error("Error during regression evaluation.")
        raise CustomException(e, sys)