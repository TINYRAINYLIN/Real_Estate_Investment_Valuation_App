"""
Retrain Ridge model to match Random Forest features
"""

import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

print("=" * 60)
print("Retraining Ridge Model")
print("=" * 60)

# Load data
train_df = pd.read_csv('artifacts/train_transformed.csv')
test_df = pd.read_csv('artifacts/test_transformed.csv')

X_train = train_df.drop(columns=['taxvaluedollarcnt'])
y_train = train_df['taxvaluedollarcnt']
X_test = test_df.drop(columns=['taxvaluedollarcnt'])
y_test = test_df['taxvaluedollarcnt']

print(f"\nTraining data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Grid search
param_grid = {'ridge__alpha': [100, 150, 200, 250]}

print(f"\nTraining Ridge with GridSearchCV...")
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n✅ Best alpha: {grid_search.best_params_['ridge__alpha']}")
print(f"✅ Best CV R²: {grid_search.best_score_:.4f}")

# Test performance
test_score = grid_search.score(X_test, y_test)
print(f"✅ Test R²: {test_score:.4f}")

# Save model
joblib.dump(grid_search.best_estimator_, 'notebook/Best_Models/best_ridge_fixed.pkl')
print(f"\n✅ Model saved to: notebook/Best_Models/best_ridge_fixed.pkl")

# Test prediction
sample_pred = grid_search.predict(X_test.iloc[:1])[0]
sample_actual = y_test.iloc[0]
print(f"\nSample prediction: ${sample_pred:,.2f}")
print(f"Actual value: ${sample_actual:,.2f}")
print(f"Error: ${abs(sample_pred - sample_actual):,.2f}")
