"""
data_transformation.py
-----------------------
This module applies data cleaning and feature engineering to the Zillow dataset.

Main steps:
1. Clean data (EDA-style):
   - Drop cols with >95% missing
   - Drop rows with missing target
   - Impute missing values
   - Drop identifiers, redundant, and sparse low-value columns
   - Drop invalid rows (0 bedrooms/bathrooms)
   - Cap extreme lot sizes (99th percentile)
2. Feature engineering:
   - Fix room count, clean garage sqft
   - Domain-driven engineered features (price_per_sqft, age_of_home, etc.)
   - Add binary flags (multi_unit, has_garage)
   - Encode categorical variables:
       * One-hot encode low-cardinality
       * Top-K encode high-cardinality
3. Save transformed train/test datasets

Outputs:
- artifacts/train_transformed.csv
- artifacts/test_transformed.csv
"""

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    """
    Configuration for saving transformed datasets and preprocessor.
    """
    transformed_train_path: str = os.path.join("artifacts", "train_transformed.csv")
    transformed_test_path: str = os.path.join("artifacts", "test_transformed.csv")


class DataTransformation:
    """
    Class for performing data cleaning and feature engineering
    on the Zillow dataset.
    """

    def __init__(self):
        self.config = DataTransformationConfig()

    # -------------------------------
    # Cleaning (EDA-style preprocessing)
    # -------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Running EDA-style cleaning...")

            # Drop columns with >95% missing
            missing_pct = df.isnull().mean() * 100
            cols_to_drop = missing_pct[missing_pct > 95].index.tolist()
            df.drop(columns=cols_to_drop, errors="ignore", inplace=True)
            logging.info(f"Dropped {len(cols_to_drop)} cols with >95% missing.")

            # Drop rows with missing target
            before = df.shape[0]
            df = df[~df["taxvaluedollarcnt"].isna()].copy()
            after = df.shape[0]
            logging.info(f"Dropped {before - after} rows with missing target.")

            # Impute missing values
            num_cols = df.select_dtypes(include=["float64", "int64"]).columns
            for col in num_cols:
                if df[col].isna().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)

            cat_cols = df.select_dtypes(include=["object"]).columns
            for col in cat_cols:
                if df[col].isna().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)

            # Drop identifiers/redundant columns
            drop_features = [
                "parcelid",
                "rawcensustractandblock", "censustractandblock",
                "calculatedbathnbr", "fullbathcnt", "threequarterbathnbr",
                "finishedfloor1squarefeet", "finishedsquarefeet12", "finishedsquarefeet50",
                "pooltypeid7", "assessmentyear"
            ]
            df.drop(columns=[c for c in drop_features if c in df.columns], errors="ignore", inplace=True)
            logging.info(f"Dropped identifier/redundant features.")

            # Drop invalid rows (0 bathrooms/bedrooms)
            before = df.shape[0]
            df = df[(df["bathroomcnt"] > 0) & (df["bedroomcnt"] > 0)]
            after = df.shape[0]
            logging.info(f"Dropped {before - after} invalid rows with 0 bed/bath.")

            # Cap extreme lot sizes
            if "lotsizesquarefeet" in df.columns:
                lot_cap = df["lotsizesquarefeet"].quantile(0.99)
                df["lotsizesquarefeet"] = df["lotsizesquarefeet"].clip(upper=lot_cap)
                logging.info(f"Capped lotsizesquarefeet at {lot_cap:,.0f} sqft.")

            return df

        except Exception as e:
            logging.error("Error during clean_data step.")
            raise CustomException(e, sys)

    # -------------------------------
    # Feature Engineering
    # -------------------------------
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting feature engineering...")

            # Fix room count
            df["roomcnt_fixed"] = np.where(
                df["roomcnt"] > 0,
                df["roomcnt"],
                df["bedroomcnt"] + df["bathroomcnt"] + 1
            )
            df.drop(columns=["roomcnt"], inplace=True, errors="ignore")

            # Clean garage sqft
            df.loc[(df["garagetotalsqft"] == 0) & (df["garagecarcnt"] > 0), "garagetotalsqft"] = np.nan
            df["garagetotalsqft"] = df.groupby("garagecarcnt")["garagetotalsqft"]\
                                      .transform(lambda x: x.fillna(x.median()))

            # Domain-driven engineered features
            df["price_per_sqft"] = df["taxvaluedollarcnt"] / (df["calculatedfinishedsquarefeet"] + 1e-5)
            df["age_of_home"] = 2025 - df["yearbuilt"]
            df["bath_per_bed"] = df["bathroomcnt"] / (df["bedroomcnt"] + 1e-5)
            df["rooms_per_sqft"] = df["roomcnt_fixed"] / (df["calculatedfinishedsquarefeet"] + 1e-5)
            df["garage_sqft_ratio"] = df["garagetotalsqft"] / (df["calculatedfinishedsquarefeet"] + 1e-5)

            # Binary flags
            df["multi_unit"] = (df["unitcnt"] > 1).astype(int)
            df["has_garage"] = ((df["garagecarcnt"].fillna(0) > 0) |
                                (df["garagetotalsqft"].fillna(0) > 0)).astype(int)

            # Encode high-cardinality categorical features
            for col, k in [("regionidcity", 50), ("regionidzip", 50), ("regionidneighborhood", 50)]:
                if col in df.columns:
                    top_vals = df[col].value_counts().nlargest(k).index
                    df[col + "_top"] = np.where(df[col].isin(top_vals), df[col], -1)
                    df = pd.get_dummies(df, columns=[col + "_top"], drop_first=True)

            # Encode propertycountylandusecode (top-15)
            if "propertycountylandusecode" in df.columns:
                top_landuse = df["propertycountylandusecode"].value_counts().nlargest(15).index
                df["propertycountylanduse_top"] = np.where(
                    df["propertycountylandusecode"].isin(top_landuse),
                    df["propertycountylandusecode"], "other"
                )
                df = pd.get_dummies(df, columns=["propertycountylanduse_top"], drop_first=True)

            # Encode propertylandusetypeid (top-5)
            if "propertylandusetypeid" in df.columns:
                top_landusetype = df["propertylandusetypeid"].value_counts().nlargest(5).index
                df["propertylandusetype_top"] = np.where(
                    df["propertylandusetypeid"].isin(top_landusetype),
                    df["propertylandusetypeid"], "other"
                )
                df = pd.get_dummies(df, columns=["propertylandusetype_top"], drop_first=True)

            # One-hot encode low-cardinality categorical features
            low_card = ["airconditioningtypeid", "heatingorsystemtypeid", "fips", "regionidcounty"]
            for col in low_card:
                if col in df.columns:
                    df = pd.get_dummies(df, columns=[col], drop_first=True)

            # Drop messy columns
            if "propertyzoningdesc" in df.columns:
                df.drop(columns=["propertyzoningdesc"], inplace=True)

            # Final safety: encode leftover object columns
            object_cols = df.select_dtypes(include=["object"]).columns
            if len(object_cols) > 0:
                logging.warning(f"Encoding leftover object columns: {list(object_cols)}")
                for col in object_cols:
                    df[col] = df[col].astype("category").cat.codes

            return df

        except Exception as e:
            logging.error("Error during feature_engineering step.")
            raise CustomException(e, sys)

    # -------------------------------
    # Transformation Orchestration
    # -------------------------------
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info(f"Reading train data from {train_path}")
            train_df = pd.read_csv(train_path)

            logging.info(f"Reading test data from {test_path}")
            test_df = pd.read_csv(test_path)

            # Clean + Feature Engineering
            train_df = self.clean_data(train_df)
            train_df = self.feature_engineering(train_df)

            test_df = self.clean_data(test_df)
            test_df = self.feature_engineering(test_df)

            # Split into X and y
            target_col = "taxvaluedollarcnt"
            X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
            X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

            # Save transformed datasets
            os.makedirs(os.path.dirname(self.config.transformed_train_path), exist_ok=True)
            pd.concat([X_train, y_train], axis=1).to_csv(self.config.transformed_train_path, index=False)
            pd.concat([X_test, y_test], axis=1).to_csv(self.config.transformed_test_path, index=False)

            logging.info("Data transformation pipeline completed successfully.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("Error during initiate_data_transformation.")
            raise CustomException(e, sys)
