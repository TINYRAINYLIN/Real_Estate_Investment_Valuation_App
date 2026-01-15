"""
Complete prediction pipeline with all 213 features
Uses smart defaults for unknown features
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging


@dataclass
class PredictionPipelineConfig:
    rf_model_path: str = os.path.join("notebook", "Best_Models", "best_randomforest.pkl")
    lgbm_model_path: str = os.path.join("notebook", "Best_Models", "best_lightgbm.pkl")
    feature_names_path: str = os.path.join("artifacts", "feature_names.json")


# Default values from training data (medians)
FEATURE_DEFAULTS = {
    'bathroomcnt': 2.0,
    'bedroomcnt': 3.0,
    'buildingqualitytypeid': 6.0,
    'calculatedfinishedsquarefeet': 1542.0,
    'fireplacecnt': 1.0,
    'garagecarcnt': 2.0,
    'garagetotalsqft': 436.0,
    'latitude': 34.0227,  # Converted from training data
    'longitude': -118.1791,
    'lotsizesquarefeet': 7208.0,
    'poolcnt': 1.0,
    'propertylandusetypeid': 261.0,
    'regionidcity': 0,
    'regionidneighborhood': 0,
    'regionidzip': 0,
    'unitcnt': 1.0,
    'yearbuilt': 1970.0,
    'numberofstories': 1.0,
}


class PredictPipeline:
    def __init__(self, model_type="randomforest"):
        self.config = PredictionPipelineConfig()
        self.model_type = model_type.lower()
        self.model = self._load_model()
        self.feature_names = self._load_feature_names()
    
    def _load_model(self):
        """Load the specified model"""
        try:
            if self.model_type == "randomforest":
                model_path = self.config.rf_model_path
            elif self.model_type == "lightgbm":
                model_path = self.config.lgbm_model_path
            else:
                raise ValueError(f"Unknown model type: {self.model_type}. Use 'randomforest' or 'lightgbm'")
            
            logging.info(f"Loading {self.model_type} model from {model_path}")
            model = joblib.load(model_path)
            logging.info(f"✅ {self.model_type} model loaded successfully")
            return model
        
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise CustomException(e, sys)
    
    def _load_feature_names(self):
        """Load feature names from JSON"""
        try:
            with open(self.config.feature_names_path, 'r') as f:
                data = json.load(f)
            return data['features']
        except Exception as e:
            logging.error(f"Error loading feature names: {e}")
            raise CustomException(e, sys)
    
    def predict(self, features_df):
        """Make predictions on input features"""
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
    Handles user input and creates all 213 features for prediction
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
        latitude: float = None,
        longitude: float = None,
        buildingqualitytypeid: int = None,
        fireplacecnt: int = 0,
        numberofstories: int = 1,
        unitcnt: int = 1,
        airconditioningtypeid: int = None,
        heatingorsystemtypeid: int = None,
    ):
        # User-provided features
        self.calculatedfinishedsquarefeet = calculatedfinishedsquarefeet
        self.bedroomcnt = bedroomcnt
        self.bathroomcnt = bathroomcnt
        self.yearbuilt = yearbuilt
        self.fips = fips
        self.regionidzip = regionidzip or 0
        self.garagetotalsqft = garagetotalsqft
        self.poolsizesum = poolsizesum
        self.lotsizesquarefeet = lotsizesquarefeet or (calculatedfinishedsquarefeet * 5)
        self.latitude = latitude or FEATURE_DEFAULTS['latitude']
        self.longitude = longitude or FEATURE_DEFAULTS['longitude']
        self.buildingqualitytypeid = buildingqualitytypeid or FEATURE_DEFAULTS['buildingqualitytypeid']
        self.fireplacecnt = fireplacecnt
        self.numberofstories = numberofstories
        self.unitcnt = unitcnt
        self.airconditioningtypeid = airconditioningtypeid
        self.heatingorsystemtypeid = heatingorsystemtypeid
    
    def _create_base_features(self):
        """Create base numerical features"""
        features = {
            'bathroomcnt': self.bathroomcnt,
            'bedroomcnt': self.bedroomcnt,
            'buildingqualitytypeid': self.buildingqualitytypeid,
            'calculatedfinishedsquarefeet': self.calculatedfinishedsquarefeet,
            'fireplacecnt': self.fireplacecnt,
            'garagecarcnt': 1 if self.garagetotalsqft > 0 else 0,
            'garagetotalsqft': self.garagetotalsqft,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'lotsizesquarefeet': self.lotsizesquarefeet,
            'poolcnt': 1 if self.poolsizesum > 0 else 0,
            'propertylandusetypeid': 261,  # Single family residential
            'regionidcity': 0,  # Unknown - will use other location features
            'regionidneighborhood': 0,  # Unknown
            'regionidzip': self.regionidzip,
            'unitcnt': self.unitcnt,
            'yearbuilt': self.yearbuilt,
            'numberofstories': self.numberofstories,
        }
        return features
    
    def _create_engineered_features(self, base_features):
        """Create engineered features"""
        sqft = base_features['calculatedfinishedsquarefeet']
        beds = base_features['bedroomcnt']
        baths = base_features['bathroomcnt']
        garage_sqft = base_features['garagetotalsqft']
        
        # Engineered features
        engineered = {
            'price_per_sqft': 0,  # Will be calculated by model
            'age_of_home': 2025 - self.yearbuilt,
            'bath_per_bed': baths / beds if beds > 0 else 0,
            'rooms_per_sqft': (beds + baths) / sqft if sqft > 0 else 0,
            'roomcnt_fixed': beds + baths,
            'garage_sqft_ratio': garage_sqft / sqft if sqft > 0 else 0,
            'multi_unit': 1 if self.unitcnt > 1 else 0,
            'has_garage': 1 if garage_sqft > 0 else 0,
        }
        
        return engineered
    
    def _create_categorical_features(self):
        """Create one-hot encoded categorical features"""
        categorical = {}
        
        # Air conditioning types (baseline is missing/other)
        ac_types = [5.0, 9.0, 11.0, 13.0]
        for ac_type in ac_types:
            key = f'airconditioningtypeid_{ac_type}'
            categorical[key] = 1 if self.airconditioningtypeid == ac_type else 0
        
        # Heating system types
        heating_types = [2.0, 6.0, 7.0, 10.0, 11.0, 13.0, 18.0, 20.0, 24.0]
        for heat_type in heating_types:
            key = f'heatingorsystemtypeid_{heat_type}'
            categorical[key] = 1 if self.heatingorsystemtypeid == heat_type else 0
        
        # FIPS (county) - baseline is 6037 (LA County)
        categorical['fips_6059.0'] = 1 if self.fips == 6059 else 0  # Orange County
        categorical['fips_6111.0'] = 1 if self.fips == 6111 else 0  # Ventura County
        
        # County IDs
        categorical['regionidcounty_2061.0'] = 1 if self.fips == 6111 else 0
        categorical['regionidcounty_3101.0'] = 1 if self.fips == 6059 else 0
        
        return categorical
    
    def _create_top_k_features(self):
        """Create top-K encoded features (property types, cities, ZIPs, neighborhoods)"""
        features = {}
        
        # Property county land use (15 categories)
        land_use_codes = ['0101', '010C', '010D', '010E', '012C', '0200', '0300', '0400', 
                          '1', '1110', '1111', '1129', '122', '34', 'other']
        for code in land_use_codes:
            features[f'propertycountylanduse_top_{code}'] = 0  # Default to baseline
        
        # Property land use type (5 categories)
        land_use_types = [248.0, 261.0, 266.0, 269.0, 'other']
        for lut in land_use_types:
            features[f'propertylandusetype_top_{lut}'] = 0
        features['propertylandusetype_top_261.0'] = 1  # Single family residential
        
        # Top 50 cities - all set to 0 (baseline/unknown)
        city_ids = [4406.0, 5534.0, 10608.0, 10723.0, 11626.0, 12292.0, 12447.0, 12773.0,
                    13150.0, 13693.0, 14542.0, 14634.0, 15554.0, 16764.0, 18874.0, 20008.0,
                    21412.0, 24174.0, 24245.0, 24384.0, 24812.0, 24832.0, 25218.0, 25459.0,
                    26964.0, 27110.0, 27491.0, 32380.0, 33252.0, 33612.0, 33840.0, 34278.0,
                    34543.0, 37086.0, 38032.0, 40227.0, 45457.0, 46298.0, 47019.0, 47568.0,
                    50749.0, 51239.0, 52650.0, 52835.0, 53571.0, 53636.0, 54053.0, 54311.0,
                    54722.0, 396054.0]
        for city_id in city_ids:
            features[f'regionidcity_top_{city_id}'] = 0
        
        # Top 50 ZIP codes
        zip_ids = [96023.0, 96027.0, 96030.0, 96050.0, 96117.0, 96122.0, 96124.0, 96186.0,
                   96193.0, 96236.0, 96351.0, 96364.0, 96368.0, 96370.0, 96373.0, 96377.0,
                   96378.0, 96383.0, 96385.0, 96389.0, 96401.0, 96505.0, 96954.0, 96962.0,
                   96964.0, 96966.0, 96974.0, 96978.0, 96985.0, 96987.0, 96989.0, 96993.0,
                   96995.0, 96996.0, 96998.0, 97041.0, 97068.0, 97078.0, 97083.0, 97089.0,
                   97091.0, 97097.0, 97116.0, 97118.0, 97317.0, 97318.0, 97319.0, 97328.0,
                   97329.0, 97330.0]
        for zip_id in zip_ids:
            key = f'regionidzip_top_{zip_id}'
            features[key] = 1 if self.regionidzip == zip_id else 0
        
        # Top 50 neighborhoods - all set to 0 (baseline/unknown)
        neighborhood_ids = [6952.0, 7877.0, 13017.0, 19810.0, 21056.0, 26134.0, 27080.0, 27987.0,
                            30731.0, 31817.0, 32059.0, 32368.0, 33183.0, 34213.0, 37739.0, 37835.0,
                            40548.0, 41131.0, 41466.0, 46736.0, 46795.0, 47880.0, 47950.0, 48200.0,
                            48570.0, 51906.0, 54300.0, 113455.0, 113910.0, 114914.0, 115609.0, 116415.0,
                            118208.0, 118849.0, 118872.0, 118920.0, 268496.0, 268548.0, 268588.0, 274049.0,
                            274514.0, 274517.0, 275078.0, 275405.0, 275496.0, 276119.0, 276450.0, 276476.0,
                            276514.0, 403184.0]
        for neighborhood_id in neighborhood_ids:
            features[f'regionidneighborhood_top_{neighborhood_id}'] = 0
        
        return features
    
    def get_data_as_dataframe(self):
        """
        Create DataFrame with all 213 features in correct order
        """
        try:
            # Create all feature groups
            base_features = self._create_base_features()
            engineered_features = self._create_engineered_features(base_features)
            categorical_features = self._create_categorical_features()
            topk_features = self._create_top_k_features()
            
            # Combine all features
            all_features = {
                **base_features,
                **engineered_features,
                **categorical_features,
                **topk_features
            }
            
            # Load feature names in correct order
            with open('artifacts/feature_names.json', 'r') as f:
                feature_order = json.load(f)['features']
            
            # Create DataFrame with features in correct order
            ordered_features = {feature: all_features.get(feature, 0) for feature in feature_order}
            df = pd.DataFrame([ordered_features])
            
            logging.info(f"Created feature DataFrame with shape: {df.shape}")
            logging.info(f"Expected 213 features, got {df.shape[1]} features")
            
            return df
        
        except Exception as e:
            logging.error(f"Error creating DataFrame: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Testing Prediction Pipeline with All 213 Features")
    print("=" * 60)
    
    # Create sample property
    sample_property = CustomData(
        calculatedfinishedsquarefeet=2000,
        bedroomcnt=3,
        bathroomcnt=2.0,
        yearbuilt=2000,
        fips=6037,  # LA County
        regionidzip=96023,  # Sample ZIP
        garagetotalsqft=400,
        poolsizesum=0
    )
    
    # Create feature DataFrame
    df = sample_property.get_data_as_dataframe()
    print(f"\n✅ Feature DataFrame created: {df.shape}")
    print(f"   Expected: (1, 213)")
    print(f"   Match: {'✅ YES' if df.shape[1] == 213 else '❌ NO'}")
    
    # Test prediction with each model
    for model_type in ['randomforest', 'lightgbm']:  # Skip Ridge (feature mismatch)
        try:
            print(f"\n🔮 Testing {model_type.upper()} model...")
            pipeline = PredictPipeline(model_type=model_type)
            prediction = pipeline.predict(df)[0]
            print(f"   Predicted value: ${prediction:,.2f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline test complete!")
    print("=" * 60)
