"""
Test prediction pipeline locally before deployment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
from src.pipeline.predicting_pipeline import PredictPipeline

def test_with_sample_data():
    """Test prediction with a sample from test set"""
    try:
        # Load test data
        test_df = pd.read_csv("artifacts/test_transformed.csv")
        
        # Get a sample (without target)
        sample = test_df.drop(columns=['taxvaluedollarcnt']).iloc[0:1]
        actual_value = test_df['taxvaluedollarcnt'].iloc[0]
        
        print("=" * 60)
        print("Testing Prediction Pipeline")
        print("=" * 60)
        
        # Test each model
        for model_type in ['ridge', 'randomforest', 'lightgbm']:
            print(f"\n🔮 Testing {model_type.upper()} model...")
            
            pipeline = PredictPipeline(model_type=model_type)
            prediction = pipeline.predict(sample)[0]
            
            error = abs(prediction - actual_value)
            error_pct = (error / actual_value) * 100
            
            print(f"  Predicted: ${prediction:,.2f}")
            print(f"  Actual:    ${actual_value:,.2f}")
            print(f"  Error:     ${error:,.2f} ({error_pct:.2f}%)")
        
        print("\n✅ Prediction pipeline test complete!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


def test_direct_model_load():
    """Test loading and using model directly"""
    try:
        print("\n" + "=" * 60)
        print("Testing Direct Model Loading")
        print("=" * 60)
        
        # Load test data
        test_df = pd.read_csv("artifacts/test_transformed.csv")
        X_test = test_df.drop(columns=['taxvaluedollarcnt'])
        y_test = test_df['taxvaluedollarcnt']
        
        # Test Random Forest
        print("\n📦 Loading Random Forest model...")
        rf_model = joblib.load("artifacts/best_randomforest.pkl")
        
        # Make predictions on first 5 samples
        predictions = rf_model.predict(X_test.iloc[:5])
        actuals = y_test.iloc[:5].values
        
        print("\nFirst 5 predictions:")
        for i, (pred, actual) in enumerate(zip(predictions, actuals), 1):
            error_pct = abs(pred - actual) / actual * 100
            print(f"  {i}. Predicted: ${pred:,.2f} | Actual: ${actual:,.2f} | Error: {error_pct:.2f}%")
        
        print("\n✅ Direct model loading test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_with_sample_data()
    test_direct_model_load()
