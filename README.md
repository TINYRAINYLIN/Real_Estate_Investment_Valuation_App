# 🏡 Zillow Property Value Prediction  

## 📌 Project Status  
⚠️ **Work in Progress**  

- ✅ Phase 1: Data Prep & EDA  
- ✅ Phase 2: Feature Engineering  
- ✅ Phase 3: Modeling (Ridge + Random Forest)  
- 🔄 Phase 3: LightGBM (in progress)  
- ⏳ Phase 4: Explainability (SHAP planned)  
- ⏳ Phase 5: Deployment (Streamlit on AWS)  
- ⏳ Phase 6: Packaging & Storytelling  

---

## 📌 Overview  
This project builds an end-to-end ML pipeline to **predict residential property values** and explain the drivers of home prices.  

- Cleaned and engineered Zillow dataset  
- Developed regression models (**Ridge, Random Forest, LightGBM**)  
- Ridge implemented with **scaling** (no log transform)  
- Random Forest & LightGBM tuned with **RandomizedSearchCV (5-fold CV)**  
- **SHAP explainability** planned for feature importance and local explanations  
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

## 📈 Model Performance  

### Main Metrics  
| Model                 | R² (Test) | RMSE   | MAE   |
|-----------------------|-----------|--------|-------|
| Ridge (scaled target) | 🔄 Pending | —      | —     |
| Random Forest         | 🔄 Pending | —      | —     |
| LightGBM              | 🔄 Pending | —      | —     |

<details>
  <summary>🔎 Additional Metrics</summary>

| Model                 | Adj R² | Median AE | MAPE  | Within $10k |
|-----------------------|--------|-----------|-------|-------------|
| Ridge (scaled target) | 🔄 —   | —         | —     | —           |
| Random Forest         | 🔄 —   | —         | —     | —           |
| LightGBM              | 🔄 —   | —         | —     | —           |

</details>  

---

## 📊 Explainability (SHAP)  
- **Planned:**  
  - Apply SHAP TreeExplainer to Random Forest & LightGBM  
  - Generate **beeswarm summary plots** (global feature importance)  
  - Generate **waterfall plots** (local explanations)  

- **Future extension:**  
  - Build **geospatial Folium map** for predicted values by ZIP code  

---

## 🛠️ Tech Stack  
- **Python**: Pandas, NumPy, Scikit-learn, LightGBM  
- **ML Models**: Ridge, Random Forest, LightGBM  
- **Explainability**: SHAP  
- **Visualization**: Matplotlib, Seaborn, Folium  
- **Deployment**: Streamlit, **AWS (EC2, S3, Docker, optional SageMaker)**  

---

## ☁️ Deployment (Planned on AWS)  
- Build interactive **Streamlit app** for property value prediction  
- Host on **AWS** with:  
  - **Model artifacts** stored on S3  
  - **App containerized with Docker** for portability  
  - Public URL for recruiters to test predictions  
- Features:  
  - Sidebar inputs → sqft, bedrooms, bathrooms, year built, ZIP code  
  - Output → **Predicted price** (formatted as \$123,456)  
  - Display **top 5 features influencing prediction** (via SHAP values)  

---

## 🚀 Next Steps  
- [ ] Finish LightGBM tuning & test evaluation  
- [ ] Evaluate Ridge, RF, and LGBM on **test set** (final metrics)  
- [ ] Add model comparison bar chart (**R² / RMSE across Ridge, RF, LGBM**)  
- [ ] Apply SHAP analysis (RF + LGBM, global + local)  
- [ ] Build geospatial map of predicted values by ZIP code  
- [ ] Deploy Streamlit app on **AWS**  

---
