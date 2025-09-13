"""
data_transformation.py
-----------------------
This module applies feature engineering and preprocessing to the Zillow dataset.

Main steps:
1. Fix inconsistent values (room count, garage sqft).
2. Create domain-driven engineered features (price_per_sqft, age_of_home, etc.).
3. Add binary flags (multi_unit, has_garage).
4. Encode categorical variables:
   - One-hot encode low-cardinality features.
   - Top-K encode high-cardinality features.
5. Save transformed train/test datasets for model training.

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

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration for saving transformed datasets and preprocessor.
    """
    transformed_train_path: str = os.path.join("artifacts", "train_transformed.csv")
    transformed_test_path: str = os.path.join("artifacts", "test_transformed.csv")
    preprocessor_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Class for performing data cleaning and feature engineering
    on the Zillow dataset.
    """

    def __init__(self):
        self.config = DataTransformationConfig()

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.

        Args:
            df (pd.DataFrame): Raw input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe with new features and encodings.
        """
        try:
            logging.info("Starting feature engineering...")

            # -------------------------------
            # Fix room count
            # -------------------------------
            # Replace missing/zero room counts with an approximation:
            # bedrooms + bathrooms + 1 (for kitchen/living space).
            logging.info("Fixing room count values...")
            df["roomcnt_fixed"] = np.where(
                df["roomcnt"] > 0,
                df["roomcnt"],
                df["bedroomcnt"] + df["bathroomcnt"] + 1
            )
            df.drop(columns=["roomcnt"], inplace=True)

            # -------------------------------
            # Clean garage sqft
            # -------------------------------
            # If garagecarcnt > 0 but garagetotalsqft == 0, mark as missing.
            # Then impute with median sqft for that garagecarcnt group.
            logging.info("Cleaning garage sqft values...")
            df.loc[(df["garagetotalsqft"] == 0) & (df["garagecarcnt"] > 0), "garagetotalsqft"] = np.nan
            df["garagetotalsqft"] = df.groupby("garagecarcnt")["garagetotalsqft"]\
                                      .transform(lambda x: x.fillna(x.median()))

            # -------------------------------
            # Domain-driven engineered features
            # -------------------------------
            logging.info("Creating engineered features...")
            df["price_per_sqft"] = df["taxvaluedollarcnt"] / (df["calculatedfinishedsquarefeet"] + 1e-5)
            df["age_of_home"] = 2025 - df["yearbuilt"]
            df["bath_per_bed"] = df["bathroomcnt"] / (df["bedroomcnt"] + 1e-5)
            df["rooms_per_sqft"] = df["roomcnt_fixed"] / (df["calculatedfinishedsquarefeet"] + 1e-5)
            df["garage_sqft_ratio"] = df["garagetotalsqft"] / (df["calculatedfinishedsquarefeet"] + 1e-5)

            # -------------------------------
            # Binary flags
            # -------------------------------
            logging.info("Adding binary flag features...")
            df["multi_unit"] = (df["unitcnt"] > 1).astype(int)
            df["has_garage"] = ((df["garagecarcnt"].fillna(0) > 0) |
                                (df["garagetotalsqft"].fillna(0) > 0)).astype(int)

            # -------------------------------
            # Encode high-cardinality categorical features
            # -------------------------------
            logging.info("Encoding high-cardinality categorical features...")
            for col, k in [("regionidcity", 50), ("regionidzip", 50), ("regionidneighborhood", 50)]:
                logging.info(f"Encoding {col} with top-{k} categories...")
                top_vals = df[col].value_counts().nlargest(k).index
                df[col + "_top"] = np.where(df[col].isin(top_vals), df[col], -1)
                df = pd.get_dummies(df, columns=[col + "_top"], drop_first=True)

            # -------------------------------
            # Encode propertycountylandusecode (top-15)
            # -------------------------------
            logging.info("Encoding propertycountylandusecode with top-15 categories...")
            top_landuse = df["propertycountylandusecode"].value_counts().nlargest(15).index
            df["propertycountylanduse_top"] = np.where(df["propertycountylandusecode"].isin(top_landuse),
                                                       df["propertycountylandusecode"], "other")
            df = pd.get_dummies(df, columns=["propertycountylanduse_top"], drop_first=True)

            # -------------------------------
            # Encode propertylandusetypeid (top-5)
            # -------------------------------
            logging.info("Encoding propertylandusetypeid with top-5 categories...")
            top_landusetype = df["propertylandusetypeid"].value_counts().nlargest(5).index
            df["propertylandusetype_top"] = np.where(df["propertylandusetypeid"].isin(top_landusetype),
                                                     df["propertylandusetypeid"], "other")
            df = pd.get_dummies(df, columns=["propertylandusetype_top"], drop_first=True)

            # -------------------------------
            # One-hot encode low-cardinality categorical features
            # -------------------------------
            logging.info("One-hot encoding low-cardinality categorical features...")
            low_card = ["airconditioningtypeid", "heatingorsystemtypeid", "fips", "regionidcounty"]
            df = pd.get_dummies(df, columns=low_card, drop_first=True)

            # -------------------------------
            # Drop messy columns
            # -------------------------------
            # propertyzoningdesc has 1800+ unique values and is not useful.
            if "propertyzoningdesc" in df.columns:
                logging.info("Dropping propertyzoningdesc column (too high-cardinality)...")
                df.drop(columns=["propertyzoningdesc"], inplace=True)

            logging.info("Feature engineering complete.")
            return df

        except Exception as e:
            logging.error("Error during feature engineering.")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Run the full data transformation pipeline.

        Args:
            train_path (str): Path to raw train.csv (from ingestion).
            test_path (str): Path to raw test.csv (from ingestion).

        Returns:
            X_train, X_test, y_train, y_test: Transformed features and targets.
        """
        try:
            logging.info(f"Reading train data from {train_path}")
            train_df = pd.read_csv(train_path)

            logging.info(f"Reading test data from {test_path}")
            test_df = pd.read_csv(test_path)

            logging.info("Applying feature engineering to train data...")
            train_df = self.feature_engineering(train_df)

            logging.info("Applying feature engineering to test data...")
            test_df = self.feature_engineering(test_df)

            # Split into features (X) and target (y)
            target_col = "taxvaluedollarcnt"
            X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
            X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

            # Save transformed datasets
            logging.info("Saving transformed datasets to artifacts folder...")
            os.makedirs(os.path.dirname(self.config.transformed_train_path), exist_ok=True)
            pd.concat([X_train, y_train], axis=1).to_csv(self.config.transformed_train_path, index=False)
            pd.concat([X_test, y_test], axis=1).to_csv(self.config.transformed_test_path, index=False)

            logging.info("Data transformation pipeline completed successfully.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("Error during data transformation.")
            raise CustomException(e, sys)
