# 🏡 Zillow Property Value Prediction  

## 📌 Current Status  
- ✅ Data Preparation, Feature Engineering, Modeling, Evaluation  
- ✅ Explainability (SHAP global + local interpretability)  
- ✅ Geospatial Analysis (Folium heatmaps + property-level maps)  
- ⏳ Deployment phase in progress:  
  - Building interactive **Streamlit app** for predictions  
  - Containerization with **Docker**  
  - Hosting planned on **AWS/Streamlit Cloud**  


---

## 📌 Overview  
An end-to-end machine learning pipeline for predicting residential property values using Zillow housing data.  
- Improved **R² from 0.69 → 0.87** with ensemble methods  
- Reduced **RMSE by ~150k** compared to baseline Ridge Regression  
- Delivered **explainable AI** insights with SHAP (global + local explanations)  
- Built **geospatial visualizations** (heatmaps & property-level maps) to highlight regional pricing trends  
- Deployment planned on **AWS with Streamlit + Docker + S3**  

---

## 📊 Dataset  
- **Source:** Zillow housing dataset (Kaggle / Zillow Research)  
- **Size:** ~77,000 rows × 213 features (after encoding & feature engineering)  
- **Target:** `taxvaluedollarcnt` (property tax value)  
- **Key Features:** living area, bedrooms, bathrooms, year built, ZIP code, engineered ratios  

---

## 🔧 Feature Engineering  
- **Domain-driven features:**  
  - `price_per_sqft = tax_value / living_area`  
  - `age_of_home = 2025 - year_built`  
  - `bath_per_bed = bathrooms / bedrooms`  
  - `rooms_per_sqft = total_rooms / sqft` (fixed inconsistent room counts)  
  - `garage_sqft_ratio = garage_sqft / living_area`  
  - `multi_unit` flag (single vs multi-family)  
  - `has_garage` flag  

- **Categorical encoding:**  
  - One-hot encoding for `airconditioningtypeid`, `heatingorsystemtypeid`, `fips`, `regionidcounty`  
  - Top-K encoding for land use, land use type, city, ZIP, and neighborhood  

- **Final dataset:** numeric-only, 213 engineered features  

---

## 🔧 Modeling & Evaluation  

1. **Ridge Regression (GridSearchCV)**  
   - Pipeline: `StandardScaler → Ridge`  
   - GridSearchCV over α = {150, 180, 200, 230, 250}  

2. **Random Forest (RandomizedSearchCV)**  
   - Tuned over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`  
   - 5-fold CV, 25 randomized candidates
   - **Best Params (Random Forest):**  
      `n_estimators=959, max_depth=10, max_features=0.7, min_samples_split=6, min_samples_leaf=2, bootstrap=True`
 

3. **LightGBM (RandomizedSearchCV)**  
   - Tuned over `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`  
   - Search in progress  

4. **Evaluation Metrics**  
   - **R² (train/test)**  
   - **Adjusted R² (test)**  
   - **RMSE, MAE, Median AE**  
   - **MAPE (%)**  
   - **% of predictions within $10k of true value**  

---

### 📊 Model Performance Comparison (Test Set)

| Model          | R² (Test) | Adj R² (Test) | RMSE     | MAE     | MAPE   | Within $10k | Within $20k | Within $50k | Within 5% | Within 10% |
|----------------|-----------|---------------|----------|---------|--------|-------------|-------------|-------------|-----------|------------|
| **Ridge**      | 0.6962    | 0.6919        | 422,633  | 117,826 | 52.37% | 8.85%       | 16.98%      | 40.10%      | 18.84%    | 34.36%     |
| **RandomForest** | 0.8743  | 0.8725        | 271,898  | 10,989  | 2.06%  | 85.23%      | 95.13%      | 98.60%      | 93.85%    | 97.77%     |
| **LightGBM**   | 0.8401    | 0.8378        | 306,639  | 24,478  | 5.74%  | 53.30%      | 78.40%      | 94.17%      | 72.80%    | 88.02%     |


---

### Explainability (SHAP)  
- **Beeswarm Plot:** Confirms key drivers are price per sqft, finished square footage, and location  
- **Dependence Plots:** Show diminishing returns for square footage and variability in lot size effect  
- **Waterfall Plot:** Provides transparency for individual homes by showing feature contributions  

📊 Example SHAP Visuals:  
![Beeswarm](https://github.com/TINYRAINYLIN/Zillow_Property_Price_Prediction/blob/main/reports/figures/old_shap_beeswarm.png))  
![Dependence](https://github.com/TINYRAINYLIN/Zillow_Property_Price_Prediction/blob/main/reports/figures/bees_space.png)  
![Waterfall](https://github.com/TINYRAINYLIN/Zillow_Property_Price_Prediction/blob/main/reports/figures/waterfall.png)  

---

### Geospatial Analysis  
- **Heatmap:** Clusters of high-value properties around Santa Monica, Beverly Hills, and coastal LA  
- **CircleMarker Map:** Property-level predictions with interactive color coding  

🌍 Example Maps:  
![Heatmap](https://github.com/TINYRAINYLIN/Real_Estate_Investment_Valuation_App/blob/main/reports/figures/Heatmap.png)
![CircleMarker](https://github.com/TINYRAINYLIN/Real_Estate_Investment_Valuation_App/blob/main/reports/figures/CircleMarker.png)


---

## 🛠️ Tech Stack  
- **Python**: Pandas, NumPy, Scikit-learn, LightGBM  
- **ML Models**: Ridge, Random Forest, LightGBM  
- **Explainability**: SHAP  
- **Visualization**: Matplotlib, Seaborn, Folium  
- **Deployment**: Streamlit, **AWS (EC2, S3, Docker, optional SageMaker)**  
