import joblib
import pandas as pd
import numpy as np

# Load Ridge model
ridge = joblib.load('notebook/Best_Models/best_ridge.pkl')

print("=" * 60)
print("Ridge Model Investigation")
print("=" * 60)

print(f"\nModel type: {type(ridge)}")
print(f"Is Pipeline? {hasattr(ridge, 'steps')}")

if hasattr(ridge, 'steps'):
    print(f"\nPipeline steps:")
    for name, step in ridge.steps:
        print(f"  - {name}: {type(step)}")

# Load test data
test_df = pd.read_csv('artifacts/test_transformed.csv')
X_test = test_df.drop(columns=['taxvaluedollarcnt'])
y_test = test_df['taxvaluedollarcnt']

print(f"\nTest data shape: {X_test.shape}")
print(f"Test target range: ${y_test.min():,.0f} - ${y_test.max():,.0f}")
print(f"Test target mean: ${y_test.mean():,.0f}")

# Test prediction on actual test data
pred_test = ridge.predict(X_test.iloc[:5])
actual_test = y_test.iloc[:5].values

print(f"\nPredictions on actual test data:")
for i, (pred, actual) in enumerate(zip(pred_test, actual_test), 1):
    print(f"  {i}. Predicted: ${pred:,.2f} | Actual: ${actual:,.2f}")

# Now test with our custom data
print("\n" + "=" * 60)
print("Testing with Custom Data")
print("=" * 60)

# Load feature names
import json
with open('artifacts/feature_names.json', 'r') as f:
    feature_names = json.load(f)['features']

# Create simple test case with all zeros except key features
custom_features = {feature: 0 for feature in feature_names}

# Set some basic values
custom_features['calculatedfinishedsquarefeet'] = 2000
custom_features['bedroomcnt'] = 3
custom_features['bathroomcnt'] = 2
custom_features['yearbuilt'] = 2000
custom_features['age_of_home'] = 25
custom_features['bath_per_bed'] = 2/3
custom_features['lotsizesquarefeet'] = 7000
custom_features['latitude'] = 34.0227
custom_features['longitude'] = -118.1791
custom_features['buildingqualitytypeid'] = 6
custom_features['propertylandusetypeid'] = 261
custom_features['unitcnt'] = 1
custom_features['numberofstories'] = 1

df_custom = pd.DataFrame([custom_features])

print(f"\nCustom data shape: {df_custom.shape}")
print(f"Non-zero features: {(df_custom != 0).sum().sum()}")

pred_custom = ridge.predict(df_custom)[0]
print(f"\nPrediction on custom data: ${pred_custom:,.2f}")

# Check if it's a scaling issue
if hasattr(ridge, 'named_steps') and 'standardscaler' in ridge.named_steps:
    scaler = ridge.named_steps['standardscaler']
    print(f"\nScaler mean shape: {scaler.mean_.shape}")
    print(f"Scaler scale shape: {scaler.scale_.shape}")
    print(f"Expected features: {len(feature_names)}")
    
    # Check for feature mismatch
    if len(scaler.mean_) != len(feature_names):
        print(f"\n⚠️ FEATURE MISMATCH!")
        print(f"   Scaler expects: {len(scaler.mean_)} features")
        print(f"   We're providing: {len(feature_names)} features")
