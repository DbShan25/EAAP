import streamlit as st
import numpy as np
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler

# Setting up the page config
st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")

@st.cache_resource
def load_model():
    """Load the trained model from a pickle file."""
    try:
        return joblib.load("c:/Users/Hxtreme/Jupyter_Notebook_Learning/Project4_V1/emp_att_model_rfc.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please check the path.")
        return None

@st.cache_resource
def load_scaler():
    """Load the pre-trained scaler for input normalization."""
    try:
        return joblib.load("c:/Users/Hxtreme/Jupyter_Notebook_Learning/Project4_V1/scaler.pkl")
    except FileNotFoundError:
        st.error("Scaler file not found. Please check the path.")
        return None

# Load model and scaler
attrition_model = load_model()
scaler = load_scaler()

# Sidebar for navigation
with st.sidebar:
    selected_page = option_menu('Employee Attrition Analysis', 
                    ['Home', 'Predict Employee Attrition'])


if selected_page == 'Home':
    st.title("🏢 **Employee Insights Dashboard**")

    # Create three columns for dropdowns
    col1, col2, col3 = st.columns(3)

    with col1:
        attrition_dropdown = st.selectbox("🚨 High-Risk Employees", ["Show", "Hide"], key="attrition_dropdown")

    with col2:
        satisfaction_dropdown = st.selectbox("😊 High Job Satisfaction", ["Show", "Hide"], key="satisfaction_dropdown")

    with col3:
        performance_dropdown = st.selectbox("🏆 High Performance Score", ["Show", "Hide"], key="performance_dropdown")

    # Load dataset
    try:
        df = pd.read_excel(r"C:\Users\Hxtreme\Jupyter_Notebook_Learning\Project4_V1\data_ea.xlsx")
        #st.success("✅ Employee data loaded successfully!")
    except FileNotFoundError:
        st.error("🚨 Employee dataset not found!")
        df = None

    if df is not None:
        # Create three columns for the tables
        table_col1, table_col2, table_col3 = st.columns(3)

        # Display High-Risk Employees if selected
        if attrition_dropdown == "Show":
            with table_col1:
                # Select relevant features for Attrition Prediction
                features = ["over_time", "stock_option_level", "marital_status", "job_satisfaction",
                            "monthly_income", "distance_from_home", "job_involvement", "years_in_current_role"]

                # Encode categorical variables
                df["over_time"] = df["over_time"].map({"No": 0, "Yes": 1})
                df["marital_status"] = df["marital_status"].map({"Single": 0, "Married": 1, "Divorced": 2})

                # Scale numeric features using pre-trained scaler
                df_scaled = scaler.transform(df[features])

                # Predict attrition probabilities
                df["attrition_risk"] = attrition_model.predict_proba(df_scaled)[:, 1]  

                # Get Top 10 High-Risk Employees
                high_risk_employees = df.nlargest(15, "attrition_risk")

                # Display Data
                #st.subheader("🚨 High-Risk Employees")
                st.dataframe(high_risk_employees[["employee_no", "attrition_risk", "performance_score_percent"]])

        # Display High Job Satisfaction Employees if selected
        if satisfaction_dropdown == "Show":
            with table_col2:
                high_satisfaction_employees = df.nlargest(15, "job_satisfaction")
                #st.subheader("😊 High Job Satisfaction Employees")
                st.dataframe(high_satisfaction_employees[["employee_no", "job_satisfaction", "attrition_risk"]])

        # Display High Performance Score Employees if selected
        if performance_dropdown == "Show":
            with table_col3:
                high_performance_employees = df.nlargest(15, "performance_score_percent")  # Ensure this column exists
                #st.subheader("🏆 High Performance Employees")
                st.dataframe(high_performance_employees[["employee_no", "performance_score_percent","job_satisfaction"]])


elif selected_page == 'Predict Employee Attrition':
    st.header('Predict Employee Attrition')

    # User Inputs
    over_time = st.selectbox("Over Time", ["No", "Yes"])
    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    job_satisfaction = st.slider("Job Satisfaction", 1, 4)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, step=500)
    distance_from_home = st.number_input("Distance From Home (miles)", min_value=1, max_value=50, step=1)
    job_involvement = st.slider("Job Involvement", 1, 4)
    years_in_current_role = st.slider("Years in Current Role", 0, 15)

    # **Encoding categorical variables**
    over_time = 1 if over_time == "Yes" else 0  # Convert "Yes"/"No" to 1/0
    marital_status_map = {"Single": 0, "Married": 1, "Divorced": 2}
    marital_status = marital_status_map[marital_status]  # Convert categories to numbers

    # **Prepare DataFrame**
    input_data = pd.DataFrame([[over_time, stock_option_level, marital_status, job_satisfaction, 
                                monthly_income, distance_from_home, job_involvement, years_in_current_role]],
                            columns=["over_time", "stock_option_level", "marital_status", "job_satisfaction", 
                                     "monthly_income", "distance_from_home", "job_involvement", "years_in_current_role"])

    # **Scale input data using the pre-trained scaler**
    if scaler is not None:
        input_data_scaled = scaler.transform(input_data)
        st.write("Scaled Input Data:", input_data_scaled)

    # **Prediction Button**
    if st.button("🔍 Predict Attrition"):
        if attrition_model is not None and scaler is not None:
            # Make prediction
            prediction_prob = attrition_model.predict_proba(input_data_scaled)[:, 1]  # Get probability
            prediction = (prediction_prob > 0.50).astype(int)  # Set custom threshold

            # Display Results
            st.write("Raw Prediction:", prediction)
            st.write("Prediction Probability:", prediction_prob)

            if prediction[0] == 1:
                st.error(f"🚨 Employee **likely to leave** (Attrition risk: {prediction_prob[0]*100:.2f}%)")
            else:
                st.success(f"✅ Employee **likely to stay** (Attrition risk: {prediction_prob[0]*100:.2f}%)")
