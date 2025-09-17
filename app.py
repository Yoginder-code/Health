import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image  # For adding images if needed

# Load the saved model and scaler
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names from your feature importance
feature_names = [
    'heartRate_num', 'spo2_num', 'temperature_num', 'systolicBP', 'diastolicBP',
    'age', 'stepsTaken', 'calorieIntake', 'sleepHours', 'waterIntakeMl',
    'mealsSkipped', 'exerciseMinutes', 'bathroomVisits'
]

# Sidebar with project details
with st.sidebar:
    st.title("📊 Project Details")
    st.markdown("---")
    
    st.header("About This App")
    st.write("""
    This Health Risk Prediction App uses a **Random Forest Classifier** to predict 
    whether a person is **Stable (0)** or **At Risk (1)** based on various health metrics.
    """)
    
    st.header("🔬 Model Information")
    st.write("""
    - **Algorithm**: Random Forest Classifier
    - **Input Features**: 13 health parameters
    - **Output**: Binary classification (0 = Stable, 1 = At Risk)
    - **Training**: Model trained on comprehensive health dataset
    """)
    
    st.header("📈 Features Used")
    st.write("""
    The model considers multiple health dimensions:
    - **Vital Signs**: Heart rate, SpO2, Temperature, Blood Pressure
    - **Lifestyle**: Steps, Exercise, Sleep, Nutrition
    - **Hydration**: Water intake, Bathroom visits
    - **Demographics**: Age
    """)
    
    st.header("⚙️ How It Works")
    st.write("""
    1. Input health metrics in the form
    2. Data is scaled using pre-fitted scaler
    3. Random Forest model makes prediction
    4. Results show prediction with probabilities
    5. Feature importance is displayed
    """)
    
    st.header("⚠️ Important Notes")
    st.write("""
    - This is a predictive tool, not medical advice
    - Consult healthcare professionals for medical decisions
    - Model includes adjusted threshold (0.4) for risk detection
    - Results should be interpreted by qualified personnel
    """)
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and Scikit-learn")

# Main content area
st.title("🏥 Health Risk Prediction App")
st.write("Enter the health metrics below to predict if a person is **Stable (0)** or **At Risk (1)**.")

# Add some visual separation
st.markdown("---")

# Input form for features
st.header("📋 Input Features")
user_data = {}

with st.form(key='input_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vital Signs")
        user_data['heartRate_num'] = st.slider("Heart Rate (bpm)", 40.0, 200.0, 70.0, 1.0, 
                                              help="Normal resting heart rate: 60-100 bpm")
        user_data['spo2_num'] = st.slider("SpO2 (%)", 70.0, 100.0, 95.0, 1.0,
                                         help="Normal oxygen saturation: 95-100%")
        user_data['temperature_num'] = st.slider("Temperature (°C)", 35.0, 42.0, 37.0, 0.1,
                                                help="Normal body temperature: 36.5-37.5°C")
        user_data['systolicBP'] = st.slider("Systolic BP (mmHg)", 80.0, 200.0, 120.0, 1.0,
                                           help="Normal systolic BP: 90-120 mmHg")
        user_data['diastolicBP'] = st.slider("Diastolic BP (mmHg)", 50.0, 120.0, 80.0, 1.0,
                                            help="Normal diastolic BP: 60-80 mmHg")
        
        st.subheader("Demographics")
        user_data['age'] = st.slider("Age (years)", 0.0, 100.0, 30.0, 1.0)
    
    with col2:
        st.subheader("Lifestyle Metrics")
        user_data['stepsTaken'] = st.slider("Steps Taken", 0.0, 20000.0, 5000.0, 100.0,
                                           help="Recommended: 7,000-10,000 steps daily")
        user_data['calorieIntake'] = st.slider("Calorie Intake (kcal)", 0.0, 5000.0, 2000.0, 100.0,
                                              help="Average adult needs 2000-2500 kcal/day")
        user_data['sleepHours'] = st.slider("Sleep Hours", 0.0, 24.0, 7.0, 0.5,
                                           help="Recommended: 7-9 hours for adults")
        user_data['waterIntakeMl'] = st.slider("Water Intake (mL)", 0.0, 5000.0, 2000.0, 100.0,
                                              help="Recommended: 2000-3000 mL daily")
        user_data['mealsSkipped'] = st.slider("Meals Skipped", 0.0, 5.0, 0.0, 1.0)
        user_data['exerciseMinutes'] = st.slider("Exercise Minutes", 0.0, 180.0, 30.0, 5.0,
                                                help="Recommended: 150+ minutes weekly")
        user_data['bathroomVisits'] = st.slider("Bathroom Visits", 0.0, 15.0, 5.0, 1.0,
                                               help="Normal: 4-10 times daily")
    
    submit_button = st.form_submit_button(label="🚀 Predict Health Status", 
                                         help="Click to analyze health risk")

# Process prediction when form is submitted
if submit_button:
    st.markdown("---")
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_data], columns=feature_names)
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Predict and get probabilities
    prediction = rf_model.predict(input_scaled)[0]
    prob = rf_model.predict_proba(input_scaled)[0]
    class_label = "Stable (0)" if prediction == 0 else "At Risk (1)"
    
    # Display prediction with color coding
    st.header("📊 Prediction Result")
    
    if prediction == 0:
        st.success(f"✅ **Prediction**: The person is **{class_label}**.")
    else:
        st.error(f"⚠️ **Prediction**: The person is **{class_label}**.")
    
    # Progress bars for probabilities
    col_prob1, col_prob2 = st.columns(2)
    
    with col_prob1:
        st.metric("Probability of Stable (0)", f"{prob[0]:.2%}")
        st.progress(prob[0])
    
    with col_prob2:
        st.metric("Probability of At Risk (1)", f"{prob[1]:.2%}")
        st.progress(prob[1])
    
    # Adjusted threshold for misprediction
    if prediction == 0 and prob[1] > 0.3:
        st.warning("""
        **Note**: Model predicted 'Stable' but risk probability is significant.
        Consider reviewing inputs or consulting healthcare professional.
        """)
        
        st.write("**Adjusted Threshold Analysis (0.4 threshold):**")
        prediction_adj = 1 if prob[1] > 0.4 else 0
        class_label_adj = "At Risk (1)" if prediction_adj == 1 else "Stable (0)"
        
        if prediction_adj == 1:
            st.error(f"**Adjusted Prediction**: The person is **{class_label_adj}**.")
        else:
            st.success(f"**Adjusted Prediction**: The person is **{class_label_adj}**.")
    
    # Display feature importance
    st.header("📈 Feature Importance Analysis")
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                 title='Most Influential Features in Prediction',
                 color='Importance', color_continuous_scale='Viridis')
    fig.update_layout(yaxis={'autorange': 'reversed'})
    st.plotly_chart(fig)
    
    # Show top 3 most important features
    st.subheader("🎯 Key Influencing Factors")
    top_features = feature_importance.head(3)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        st.write(f"{i}. **{row['Feature']}** - {row['Importance']:.3f} importance")

# Add footer
st.markdown("---")
st.caption("""
**Disclaimer**: This application is for educational and demonstration purposes only. 
It should not be used for medical diagnosis or treatment decisions. 
Always consult with qualified healthcare professionals for medical advice.
""")