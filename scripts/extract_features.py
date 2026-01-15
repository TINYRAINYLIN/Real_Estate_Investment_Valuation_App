"""
Extract feature names from trained model for deployment
This helps ensure prediction pipeline matches training features
"""

import os
import joblib
import pandas as pd
import json

def extract_features_from_model():
    """Extract feature names from trained Random Forest model"""
    try:
        # Try multiple locations for the model
        model_paths = [
            "artifacts/best_randomforest.pkl",
            "notebook/Best_Models/best_randomforest.pkl"
        ]
        
        model = None
        for path in model_paths:
            if os.path.exists(path):
                print(f"Loading model from: {path}")
                model = joblib.load(path)
                break
        
        if model is None:
            print("❌ Model file not found in any location")
            return None
        
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_.tolist()
            print(f"✅ Found {len(features)} features from model")
            
            # Save to JSON
            with open("artifacts/feature_names.json", "w") as f:
                json.dump({"features": features, "n_features": len(features)}, f, indent=2)
            
            print(f"✅ Feature names saved to artifacts/feature_names.json")
            
            # Print first 20 features
            print("\nFirst 20 features:")
            for i, feat in enumerate(features[:20], 1):
                print(f"  {i}. {feat}")
            
            return features
        else:
            print("❌ Model doesn't have feature_names_in_ attribute")
            return None
            
    except FileNotFoundError:
        print("❌ Model file not found. Please train model first.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def extract_features_from_data():
    """Extract feature names from transformed training data"""
    try:
        train_df = pd.read_csv("artifacts/train_transformed.csv")
        features = [col for col in train_df.columns if col != 'taxvaluedollarcnt']
        
        print(f"✅ Found {len(features)} features from training data")
        
        # Save to JSON
        with open("artifacts/feature_names_from_data.json", "w") as f:
            json.dump({"features": features, "n_features": len(features)}, f, indent=2)
        
        print(f"✅ Feature names saved to artifacts/feature_names_from_data.json")
        
        # Categorize features
        categorical_features = [f for f in features if any(x in f for x in ['_', 'fips', 'region', 'city', 'zip'])]
        numerical_features = [f for f in features if f not in categorical_features]
        
        print(f"\n📊 Feature breakdown:")
        print(f"  - Numerical features: {len(numerical_features)}")
        print(f"  - Categorical features: {len(categorical_features)}")
        
        print("\nSample numerical features:")
        for feat in numerical_features[:10]:
            print(f"  - {feat}")
        
        print("\nSample categorical features:")
        for feat in categorical_features[:10]:
            print(f"  - {feat}")
        
        return features
        
    except FileNotFoundError:
        print("❌ Training data not found")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Extracting features from trained model...")
    print("=" * 60)
    
    features_from_model = extract_features_from_model()
    
    print("\n" + "=" * 60)
    print("Extracting features from training data...")
    print("=" * 60)
    
    features_from_data = extract_features_from_data()
    
    if features_from_model and features_from_data:
        if features_from_model == features_from_data:
            print("\n✅ Features match between model and data!")
        else:
            print("\n⚠️ Features differ between model and data")
            print(f"Model features: {len(features_from_model)}")
            print(f"Data features: {len(features_from_data)}")
