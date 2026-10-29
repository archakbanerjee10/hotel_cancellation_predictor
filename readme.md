#  Hotel Booking Cancellation Predictor

An end-to-end Machine Learning web application built using **Python**, **Scikit-Learn**, and **Streamlit** to predict whether a hotel reservation is likely to be canceled (`is_canceled = 1`) or fulfilled (`is_canceled = 0`) using the famous Kaggle Hotel Booking Demand dataset.

---

##  Project Workflow & Pipeline

This project follows a structured data science lifecycle:
1. **Exploratory Data Analysis (EDA):** Performed in-depth data exploration and visualization inside Jupyter notebooks to understand cancellation trends, lead times, customer types, and seasonal patterns.
2. **Data Preprocessing & Cleaning:** Handled missing values, dropped duplicate records, and removed anomalies/outliers to ensure high data quality.
3. **Model Training:** Trained a robust **Random Forest Classifier** using Scikit-Learn to capture complex patterns in booking behaviors.
4. **Interactive Deployment:** Built a responsive, user-friendly frontend UI using **Streamlit** allowing users to query live cancellation predictions.

---

## 📁 Repository Structure
```text
HOTEL-BOOKING-PREDICTOR/
│
├── .venv/                  # Python virtual environment
├── app/
│   └── app.py              # Streamlit web application interface
├── data/
│   └── hotel_bookings.csv  # Raw Kaggle dataset
├── models/
│   ├── hotel_model.pkl     # Trained Random Forest model artifact
│   └── model_features.pkl  # Saved feature column mapping list
├── notebooks/
│   └── analysis.ipynb      # EDA, data cleaning, and model training notebook
├── src/                    # Helper scripts and modules
└── requirements.txt        # Project dependencies