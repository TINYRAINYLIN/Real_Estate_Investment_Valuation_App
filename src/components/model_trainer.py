# src/components/model_trainer.py

import os
import sys
import joblib
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_regression


@dataclass
class ModelTrainerConfig:
    ridge_model_path: str = os.path.join("artifacts", "best_ridge.pkl")
    rf_model_path: str    = os.path.join("artifacts", "best_randomforest.pkl")
    lgbm_model_path: str  = os.path.join("artifacts", "best_lightgbm.pkl")
    train_path: str       = os.path.join("artifacts", "train_transformed.csv")
    test_path: str        = os.path.join("artifacts", "test_transformed.csv")
    metrics_path: str     = os.path.join("artifacts", "model_metrics.csv")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self):
        try:
            logging.info("Loading transformed train/test datasets...")
            train_df = pd.read_csv(self.config.train_path)
            test_df  = pd.read_csv(self.config.test_path)

            X_train, y_train = train_df.drop(columns=["taxvaluedollarcnt"]), train_df["taxvaluedollarcnt"]
            X_test, y_test   = test_df.drop(columns=["taxvaluedollarcnt"]), test_df["taxvaluedollarcnt"]

            results = {}

            # ---------------- Ridge Regression ----------------
            logging.info("Training Ridge Regression with GridSearchCV...")
            ridge = Ridge()
            ridge_param_grid = {"alpha": [100, 150, 200, 250]}
            ridge_search = GridSearchCV(ridge, ridge_param_grid, cv=5, scoring="r2", n_jobs=-1)
            ridge_search.fit(X_train, y_train)
            best_ridge = ridge_search.best_estimator_
            joblib.dump(best_ridge, self.config.ridge_model_path)
            logging.info(f"✅ Ridge model saved to {self.config.ridge_model_path}")
            results["Ridge"] = evaluate_regression(best_ridge, X_train, y_train, X_test, y_test)

            # ---------------- Random Forest ----------------
            logging.info("Training Random Forest with RandomizedSearchCV...")
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            rf_param_dist = {
                "n_estimators": [300, 500, 800],
                "max_depth": [8, 12, 16],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.5, 0.7],
            }
            rf_search = RandomizedSearchCV(rf, rf_param_dist, n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42)
            rf_search.fit(X_train, y_train)
            best_rf = rf_search.best_estimator_
            joblib.dump(best_rf, self.config.rf_model_path)
            logging.info(f"✅ Random Forest model saved to {self.config.rf_model_path}")
            results["RandomForest"] = evaluate_regression(best_rf, X_train, y_train, X_test, y_test)

            # ---------------- LightGBM ----------------
            logging.info("Training LightGBM with RandomizedSearchCV...")
            lgbm = LGBMRegressor(objective="regression", random_state=42, n_jobs=-1)
            lgbm_param_dist = {
                "n_estimators": [300, 500, 800],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 50, 70],
                "max_depth": [-1, 5, 10],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
            }
            lgbm_search = RandomizedSearchCV(lgbm, lgbm_param_dist, n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42)
            lgbm_search.fit(X_train, y_train)
            best_lgbm = lgbm_search.best_estimator_
            joblib.dump(best_lgbm, self.config.lgbm_model_path)
            logging.info(f"✅ LightGBM model saved to {self.config.lgbm_model_path}")
            results["LightGBM"] = evaluate_regression(best_lgbm, X_train, y_train, X_test, y_test)

            # ---------------- Save metrics ----------------
            metrics_df = pd.DataFrame(results).T
            metrics_df.to_csv(self.config.metrics_path, index=True)
            logging.info(f"📊 Model evaluation metrics saved to {self.config.metrics_path}")

            return metrics_df

        except Exception as e:
            logging.error("❌ Error in model training pipeline.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    metrics = trainer.initiate_model_trainer()
    print(metrics)
