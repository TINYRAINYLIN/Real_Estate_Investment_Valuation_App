import os
import sys
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


@dataclass
class PredictionPipelineConfig:
    ridge_model_path: str = os.path.join("artifacts", "best_ridge.pkl")
    rf_model_path: str = os.path.join("artifacts", "best_randomforest.pkl")
    lgbm_model_path: str = os.path.join("artifacts", "best_lightgbm.pkl")


class PredictPipeline:
    def __init__(self, model_type="randomforest"):
        self.config = PredictionPipelineConfig()
        self.model_type = model_type.lower()
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the specified model"""
        try:
            if self.model_type == "ridge":
                model_path = self.config.ridge_model_path
            elif self.model_type == "randomforest":
                model_path = self.config.rf_model_path
            elif self.model_type == "lightgbm":
                model_path = self.config.lgbm_model_path
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            logging.info(f"Loading {self.model_type} model from {model_path}")
            model = joblib.load(model_path)
            logging.info(f"✅ {self.model_type} model loaded successfully")
            return model
        
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise CustomException(e, sys)
    
    def predict(self, features_df):
        """
        Make predictions on input features
        
        Args:
            features_df: DataFrame with same features as training data (213 columns)
        
        Returns:
            predictions: numpy array of predicted property values
        """
        try:
            logging.info(f"Making predictions with {self.model_type} model")
            predictions = self.model.predict(features_df)
            logging.info(f"✅ Predictions completed: {len(predictions)} properties")
            return predictions
        
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise CustomException(e, sys)


class CustomData:
    """
    Class for handling custom input data and feature engineering
    """
    def __init__(
        self,
        calculatedfinishedsquarefeet: float,
        bedroomcnt: int,
        bathroomcnt: float,
        yearbuilt: int,
        fips: int,
        regionidzip: int = None,
        garagetotalsqft: float = 0,
        poolsizesum: float = 0,
        lotsizesquarefeet: float = None,
        **kwargs
    ):
        self.calculatedfinishedsquarefeet = calculatedfinishedsquarefeet
        self.bedroomcnt = bedroomcnt
        self.bathroomcnt = bathroomcnt
        self.yearbuilt = yearbuilt
        self.fips = fips
        self.regionidzip = regionidzip
        self.garagetotalsqft = garagetotalsqft
        self.poolsizesum = poolsizesum
        self.lotsizesquarefeet = lotsizesquarefeet or calculatedfinishedsquarefeet * 2
        self.additional_features = kwargs
    
    def engineer_features(self):
        """
        Apply feature engineering transformations
        """
        try:
            # Basic engineered features
            age_of_home = 2025 - self.yearbuilt
            bath_per_bed = self.bathroomcnt / self.bedroomcnt if self.bedroomcnt > 0 else 0
            garage_sqft_ratio = self.garagetotalsqft / self.calculatedfinishedsquarefeet if self.calculatedfinishedsquarefeet > 0 else 0
            has_garage = 1 if self.garagetotalsqft > 0 else 0
            has_pool = 1 if self.poolsizesum > 0 else 0
            
            # Create base feature dictionary
            features = {
                'calculatedfinishedsquarefeet': self.calculatedfinishedsquarefeet,
                'bedroomcnt': self.bedroomcnt,
                'bathroomcnt': self.bathroomcnt,
                'yearbuilt': self.yearbuilt,
                'age_of_home': age_of_home,
                'bath_per_bed': bath_per_bed,
                'garagetotalsqft': self.garagetotalsqft,
                'garage_sqft_ratio': garage_sqft_ratio,
                'has_garage': has_garage,
                'poolsizesum': self.poolsizesum,
                'has_pool': has_pool,
                'lotsizesquarefeet': self.lotsizesquarefeet,
                'fips': self.fips,
            }
            
            if self.regionidzip:
                features['regionidzip'] = self.regionidzip
            
            # Add any additional features
            features.update(self.additional_features)
            
            return features
        
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise CustomException(e, sys)
    
    def get_data_as_dataframe(self):
        """
        Convert custom data to DataFrame format for prediction
        
        Note: This needs to match the exact 213 features from training
        You'll need to add proper one-hot encoding and all features
        """
        try:
            features = self.engineer_features()
            
            # TODO: Add proper feature transformation to match training data
            # This should include:
            # - One-hot encoding for categorical variables
            # - All 213 features in correct order
            # - Proper handling of missing values
            
            df = pd.DataFrame([features])
            logging.info(f"Created feature DataFrame with shape: {df.shape}")
            
            return df
        
        except Exception as e:
            logging.error(f"Error creating DataFrame: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example usage
    sample_data = CustomData(
        calculatedfinishedsquarefeet=2000,
        bedroomcnt=3,
        bathroomcnt=2.0,
        yearbuilt=2000,
        fips=6037,
        regionidzip=90001,
        garagetotalsqft=400,
        poolsizesum=0
    )
    
    df = sample_data.get_data_as_dataframe()
    print(f"Feature DataFrame shape: {df.shape}")
    print(df.head())
