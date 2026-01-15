# 🏡 PropertyIQ

## 📌 Current Status  
✅ **Production Ready - Live on Streamlit Cloud**

- ✅ Data Preparation, Feature Engineering, Modeling, Evaluation  
- ✅ Explainability (SHAP global + local interpretability)  
- ✅ Geospatial Analysis (Folium heatmaps + property-level maps)  
- ✅ Deployment Complete (Interactive Streamlit app live and running)

🌐 **Live Demo:** [PropertyIQ](https://realestateinvestmentvaluationapp-ac5hrwunnzbgqxbxlkuwjz.streamlit.app/)

---

## 📌 Overview  
An end-to-end machine learning pipeline for predicting residential property values using Zillow housing data.  
- Improved **R² from 0.69 → 0.87** with ensemble methods  
- Reduced **RMSE by ~150k** compared to baseline Ridge Regression  
- Delivered **explainable AI** insights with SHAP (global + local explanations)  
- Built **geospatial visualizations** (heatmaps & property-level maps) to highlight regional pricing trends  
- **Live on Streamlit Cloud** with Docker + AWS deployment options

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

1. **Random Forest (RandomizedSearchCV)** ⭐ **Primary Model**
   - Tuned over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`  
   - 5-fold CV, 25 randomized candidates
   - **Best Params (Random Forest):**  
      `n_estimators=959, max_depth=10, max_features=0.7, min_samples_split=6, min_samples_leaf=2, bootstrap=True`

2. **LightGBM (RandomizedSearchCV)**  
   - Tuned over `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`  
   - 5-fold CV optimization

3. **Evaluation Metrics**  
   - **R² (train/test)**  
   - **Adjusted R² (test)**  
   - **RMSE, MAE, Median AE**  
   - **MAPE (%)**  
   - **% of predictions within $10k/$20k/$50k of true value**  

---

### 📊 Model Performance Comparison (Test Set)

| Model          | R² (Test) | Adj R² (Test) | RMSE     | MAE     | MAPE   | Within $10k | Within $20k | Within $50k | Within 5% | Within 10% |
|----------------|-----------|---------------|----------|---------|--------|-------------|-------------|-------------|-----------|------------|
| **RandomForest** | 0.8743  | 0.8725        | 271,898  | 10,989  | 2.06%  | 85.23%      | 95.13%      | 98.60%      | 93.85%    | 97.77%     |
| **LightGBM**   | 0.8401    | 0.8378        | 306,639  | 24,478  | 5.74%  | 53.30%      | 78.40%      | 94.17%      | 72.80%    | 88.02%     |
| **Ridge**      | 0.6962    | 0.6919        | 422,633  | 117,826 | 52.37% | 8.85%       | 16.98%      | 40.10%      | 18.84%    | 34.36%     |

**🏆 Random Forest** selected as primary model for deployment due to superior accuracy and precision.

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

**Folium Interactive Maps:**  
- **Heatmap:** Visualizes property density and value distribution across LA, Orange, and Ventura counties  
- **Property-Level Map:** Individual markers with popup details (price, sqft, bedrooms, bathrooms)  
- **Insights:** Clear geographic clustering of high-value properties in coastal areas and affluent neighborhoods  

📍 Example Maps:  
![Heatmap](https://github.com/TINYRAINYLIN/Real_Estate_Investment_Valuation_App/blob/main/reports/figures/Heatmap.png)  

---

## 🛠️ Tech Stack  
- **Python**: Pandas, NumPy, Scikit-learn, LightGBM  
- **ML Models**: Random Forest (Primary), LightGBM  
- **Web App**: Streamlit with interactive UI
- **Deployment**: Streamlit Cloud (Live) + AWS EC2 option
- **Containerization**: Docker + Docker Compose
- **Visualization**: Matplotlib, Seaborn, Folium  
- **Explainability**: SHAP

---

## ☁️ Deployment  

### 🌐 Streamlit Cloud (Live)
- **Interactive web app** for real-time property value predictions  
- **Free hosting** on Streamlit Cloud
- **Features:**  
  - Sidebar inputs → sqft, bedrooms, bathrooms, year built, ZIP code, garage, pool
  - Output → **Predicted price** (formatted as $123,456)  
  - **Random Forest model** (87.4% R²) with 213 engineered features
  - Property summary with price per sqft calculation

### 🏗️ AWS Deployment (Alternative)
- **Complete infrastructure** with Docker + EC2 + S3
- **Model artifacts** stored on S3  
- **Containerized app** for scalability
- **Production-ready** with monitoring and auto-scaling options

---

## 🚀 Quick Start

### Try the Live App
🌐 **[PropertyIQ - Live Demo](https://realestateinvestmentvaluationapp-ac5hrwunnzbgqxbxlkuwjz.streamlit.app/)**

### Run Locally
```bash
# Clone repository
git clone https://github.com/TINYRAINYLIN/Real_Estate_Investment_Valuation_App.git
cd Real_Estate_Investment_Valuation_App

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### Key Features
- 🏠 **Property Input:** Living area, bedrooms, bathrooms, year built, location
- 🎯 **AI Prediction:** Random Forest model (87.4% R²) 
- 📊 **Smart Features:** 213 engineered features with smart defaults
- 💰 **Instant Results:** Property value + price per sqft
- 📱 **Responsive:** Works on desktop and mobile

---


**🌟 Star this repo if it helped you!** | **🔗 [Live Demo - PropertyIQ](https://realestateinvestmentvaluationapp-ac5hrwunnzbgqxbxlkuwjz.streamlit.app/)**

