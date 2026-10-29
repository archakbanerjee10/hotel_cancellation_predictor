import streamlit as st
import pandas as pd
import joblib
import os

# Dynamically construct paths relative to app/app.py -> going up to root, then into models/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/hotel_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, '../models/model_features.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, '../models/encoders.pkl')

# Cache the asset loading to improve app performance
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else None
    return model, features, encoders

model, features, encoders = load_assets()

# Page Configuration
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="centered"
)

st.title("🏨 Hotel Booking Cancellation Predictor")
st.write("Provide the reservation details below to predict whether the booking is likely to be canceled.")

# Organize user input fields into a clean form layout
with st.form("prediction_form"):
    st.subheader("Reservation Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
        lead_time = st.number_input("Lead Time (Days in advance)", min_value=0, max_value=730, value=30)
        stays_weekend = st.number_input("Weekend Nights", min_value=0, max_value=10, value=1)
        stays_week = st.number_input("Week Nights", min_value=0, max_value=30, value=2)
        adults = st.slider("Adults", min_value=1, max_value=5, value=2)
        children = st.slider("Children", min_value=0, max_value=4, value=0)
        babies = st.slider("Babies", min_value=0, max_value=2, value=0)
        
    with col2:
        market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate"])
        customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
        is_repeated = st.selectbox("Is Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        prev_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=20, value=0)
        booking_changes = st.number_input("Booking Changes", min_value=0, max_value=10, value=0)
        adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, max_value=1000.0, value=100.0)
        parking = st.slider("Required Car Parking Spaces", min_value=0, max_value=8, value=0)
        special_requests = st.slider("Total Special Requests", min_value=0, max_value=5, value=0)

    submit_button = st.form_submit_button(label="Predict Cancellation")

# Handle Prediction Logic when button is clicked
if submit_button:
    # Transform categorical inputs using saved encoders if available
    if encoders and 'hotel' in encoders:
        hotel_val = encoders['hotel'].transform([hotel])[0]
        market_val = encoders['market_segment'].transform([market_segment])[0]
        customer_val = encoders['customer_type'].transform([customer_type])[0]
    else:
        # Fallback mapping if encoders.pkl is missing
        hotel_val = 0 if hotel == "City Hotel" else 1
        market_mapping = {"Corporate": 0, "Direct": 1, "Groups": 2, "Offline TA/TO": 3, "Online TA/TO": 4}
        market_val = market_mapping.get(market_segment, 4)
        customer_mapping = {"Contract": 0, "Group": 1, "Transient": 2, "Transient-Party": 3}
        customer_val = customer_mapping.get(customer_type, 2)

    # Construct input dataframe matching exact model features order
    input_data = pd.DataFrame([[
        hotel_val, lead_time, stays_weekend, stays_week,
        adults, children, babies, is_repeated,
        prev_cancellations, booking_changes, adr, parking,
        special_requests, market_val, customer_val
    ]], columns=features)

    # Predict outcome and probability
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Cancellation!** (Probability: {prediction_proba[1]*100:.2f}%)")
        st.write("This booking exhibits patterns typical of cancellations. Consider proactive follow-up or flexible policy checks.")
    else:
        st.success(f"✅ **Low Risk of Cancellation** (Probability of staying: {prediction_proba[0]*100:.2f}%)")
        st.write("This booking is stable and likely to check in as scheduled.")