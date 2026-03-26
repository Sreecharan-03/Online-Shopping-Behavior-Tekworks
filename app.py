import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Ad Click Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border-left: 5px solid #667eea;
    }
    h1 { color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

with st.sidebar:
    st.markdown("### 🎯 Ad Click Prediction System")
    st.markdown("---")
    selected = st.radio(
        "Navigation",
        ["🏠 Prediction", "📊 Model Info", "ℹ️ About"],
        index=0
    )

if selected == "🏠 Prediction":
    st.markdown("<h1 style='color: white;'>🎯 Predict Ad Clicks</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    if not model_loaded:
        st.error("❌ Model file not found!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⏱️ Session Duration")
            time_on_site = st.slider(
                "Time on Site (minutes)",
                min_value=1.0,
                max_value=20.0,
                value=10.0,
                step=0.1,
                key="time_slider"
            )
            st.metric("Selected Time", f"{time_on_site:.2f} min")
        
        with col2:
            st.markdown("### 🛒 Shopping Activity")
            items_in_cart = st.slider(
                "Items in Cart",
                min_value=0,
                max_value=9,
                value=4,
                step=1,
                key="items_slider"
            )
            st.metric("Selected Items", f"{items_in_cart} items")
        
        st.markdown("---")
        
        if st.button("🚀 Predict Now", key="predict_btn"):
            input_data = np.array([[time_on_site, items_in_cart]])
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            click_prob = probabilities[1] * 100
            no_click_prob = probabilities[0] * 100
            confidence = max(probabilities) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div style="text-align:center;">
                        <h3 style="color: #10b981;">✅ Will Click</h3>
                        <p style="font-size: 24px; color: #10b981; font-weight: bold;">YES</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align:center;">
                        <h3 style="color: #ef4444;">❌ Won't Click</h3>
                        <p style="font-size: 24px; color: #ef4444; font-weight: bold;">NO</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            with col3:
                st.metric("Model", "Random Forest")
            
            st.markdown("### 📊 Probability Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.progress(click_prob / 100)
                st.metric("Click Probability", f"{click_prob:.1f}%")
            
            with col2:
                st.progress(no_click_prob / 100)
                st.metric("No Click Probability", f"{no_click_prob:.1f}%")

elif selected == "📊 Model Info":
    st.markdown("<h1 style='color: white;'>📊 Model Information</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Algorithm:** Random Forest
        **Trees:** 200
        **Max Depth:** 15
        """)
    
    with col2:
        st.info("""
        **CV Accuracy:** 52.00%
        **CV F1-Score:** 54.94%
        **Test F1-Score:** 52.12%
        """)
    
    st.markdown("### 📚 Features Used")
    st.markdown("""
    | Feature | Range |
    |---------|-------|
    | Time on Site | 1.0 - 20.0 min |
    | Items in Cart | 0 - 9 items |
    """)

elif selected == "ℹ️ About":
    st.markdown("<h1 style='color: white;'>ℹ️ About This App</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    Predicts whether users will click advertisements based on their website behavior.
    
    ### 🛠️ Technology Stack
    - **ML:** Scikit-learn
    - **Algorithm:** Random Forest (200 trees)
    - **Frontend:** Streamlit
    - **Preprocessing:** StandardScaler + Stratified Sampling
    
    ### 📊 Model Performance
    - **Training Data:** 625 samples
    - **Test Data:** 157 samples
    - **Validation:** 5-fold Stratified KFold
    - **Best Metrics:** CV F1-Score 54.94%
    
    ### 🔬 Key Insights
    ✅ Random Forest captures non-linear patterns
    ✅ Session duration strongly influences clicks
    ✅ Cart activity shows user engagement
    ✅ Provides probability estimates for each prediction
    
    ---
    **Built with ❤️ using Streamlit and Scikit-learn**
    """)
