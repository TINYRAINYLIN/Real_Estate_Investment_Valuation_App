import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.predicting_pipeline_v2 import PredictPipeline, CustomData

# Page config
st.set_page_config(
    page_title="PropertyIQ - AI Property Valuation",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .info-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏡 PropertyIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered property valuation using Random Forest</div>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("🏠 Property Features")
st.sidebar.markdown("---")

# Basic features
st.sidebar.subheader("📐 Basic Information")
sqft = st.sidebar.number_input(
    "Living Area (sqft)", 
    min_value=300, 
    max_value=20000, 
    value=2000, 
    step=100,
    help="Total finished square footage"
)

bedrooms = st.sidebar.number_input(
    "Bedrooms", 
    min_value=1, 
    max_value=10, 
    value=3, 
    step=1
)

bathrooms = st.sidebar.number_input(
    "Bathrooms", 
    min_value=1.0, 
    max_value=10.0, 
    value=2.0, 
    step=0.5
)

year_built = st.sidebar.number_input(
    "Year Built", 
    min_value=1900, 
    max_value=2025, 
    value=2000, 
    step=1
)

st.sidebar.markdown("---")

# Location
st.sidebar.subheader("📍 Location")
county_map = {
    "Los Angeles County": 6037,
    "Orange County": 6059,
    "Ventura County": 6111
}
county_name = st.sidebar.selectbox(
    "County", 
    list(county_map.keys()),
    index=0
)
fips = county_map[county_name]

zip_code = st.sidebar.number_input(
    "ZIP Code", 
    min_value=90001, 
    max_value=99999, 
    value=96023,
    step=1,
    help="Property ZIP code"
)

st.sidebar.markdown("---")

# Additional features
st.sidebar.subheader("🏗️ Additional Features")

has_garage = st.sidebar.checkbox("Has Garage", value=True)
garage_sqft = st.sidebar.number_input(
    "Garage Area (sqft)", 
    min_value=0, 
    max_value=2000, 
    value=400 if has_garage else 0, 
    step=50,
    disabled=not has_garage
)

has_pool = st.sidebar.checkbox("Has Pool", value=False)
pool_sqft = st.sidebar.number_input(
    "Pool Area (sqft)", 
    min_value=0, 
    max_value=1000, 
    value=200 if has_pool else 0, 
    step=50,
    disabled=not has_pool
)

lot_size = st.sidebar.number_input(
    "Lot Size (sqft)", 
    min_value=0, 
    max_value=100000, 
    value=int(sqft * 5), 
    step=500,
    help="Total lot size (leave 0 for auto-estimate)"
)

st.sidebar.markdown("---")

# Model info
st.sidebar.subheader("🤖 Model")
st.sidebar.info("**Random Forest**\n\n87.4% R² Score\n\n85.2% within $10k")

# Predict button
predict_button = st.sidebar.button("🔮 Predict Property Value", type="primary", use_container_width=True)

# Main content
if not predict_button:
    # Show instructions when no prediction yet
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>📊 How It Works</h3>
        <p>1. Enter property details in the sidebar</p>
        <p>2. Select a prediction model</p>
        <p>3. Click "Predict Property Value"</p>
        <p>4. Get instant price estimate!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Model Performance</h3>
        <p><strong>Random Forest</strong></p>
        <p>• R² Score: 87.4%</p>
        <p>• 85.2% within $10k</p>
        <p>• 95.1% within $20k</p>
        <p>• Training: 61,393 properties</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
        <h3>🏘️ Coverage Area</h3>
        <p>• Los Angeles County</p>
        <p>• Orange County</p>
        <p>• Ventura County</p>
        <p>• 50+ cities and neighborhoods</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👈 Enter property details in the sidebar to get started!")

else:
    # Make prediction
    try:
        with st.spinner("🔮 Predicting property value..."):
            # Create custom data object
            property_data = CustomData(
                calculatedfinishedsquarefeet=sqft,
                bedroomcnt=bedrooms,
                bathroomcnt=bathrooms,
                yearbuilt=year_built,
                fips=fips,
                regionidzip=zip_code,
                garagetotalsqft=garage_sqft,
                poolsizesum=pool_sqft,
                lotsizesquarefeet=lot_size if lot_size > 0 else None
            )
            
            # Get feature DataFrame
            features_df = property_data.get_data_as_dataframe()
            
            # Make prediction with Random Forest
            rf_pipeline = PredictPipeline(model_type="randomforest")
            prediction = rf_pipeline.predict(features_df)[0]
            
            # Display prediction
            st.success("✅ Prediction Complete!")
            
            st.markdown(f"""
            <div class="prediction-box">
            <h3>🏡 Estimated Property Value</h3>
            <div class="prediction-value">${prediction:,.0f}</div>
            <p style="color: #666; margin-top: 0.5rem;">Random Forest Model (87.4% R²)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display property summary
            st.markdown("---")
            st.subheader("📋 Property Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = 2025 - year_built
                st.markdown(f"""
                **Basic Information:**
                - Living Area: {sqft:,} sqft
                - Bedrooms: {bedrooms}
                - Bathrooms: {bathrooms}
                - Year Built: {year_built} ({age} years old)
                - Price per sqft: ${prediction/sqft:,.0f} (estimated)
                """)
            
            with col2:
                garage_text = f"{garage_sqft:,} sqft" if has_garage else "None"
                pool_text = f"{pool_sqft:,} sqft" if has_pool else "None"
                lot_text = f"{lot_size:,} sqft" if lot_size > 0 else "Auto-estimated"
                
                st.markdown(f"""
                **Location & Features:**
                - County: {county_name}
                - ZIP Code: {zip_code}
                - Garage: {garage_text}
                - Pool: {pool_text}
                - Lot Size: {lot_text}
                """)
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("Please check your inputs and try again.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Built with:** Streamlit + scikit-learn")
with col2:
    st.markdown("**Model:** Random Forest (87.4% R²)")
with col3:
    st.markdown("**Data:** 61,393 properties | 213 features")