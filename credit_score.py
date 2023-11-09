import pandas as pd
import streamlit as st
import xgboost
from xgboost import XGBClassifier
import pickle
import numpy as np

# Load the trained XGBoost model (best_xgb_model)
# You can replace 'best_xgb_model.pkl' with the actual file path
import pickle

with open('xgb_model.pkl', 'rb') as model_file:
    best_xgb_model = pickle.load(model_file)

# Define all possible dummy variable column names
dummy_columns = [
    'Occupation_Architect', 'Occupation_Developer', 'Occupation_Doctor',
    'Occupation_Engineer', 'Occupation_Entrepreneur', 'Occupation_Journalist',
    'Occupation_Lawyer', 'Occupation_Manager', 'Occupation_Mechanic',
    'Occupation_Media_Manager', 'Occupation_Musician', 'Occupation_Scientist',
    'Occupation_Teacher', 'Occupation_Writer', 'Credit_Mix_Good',
    'Credit_Mix_Standard', 'Payment_of_Min_Amount_No', 'Payment_of_Min_Amount_Yes',
    'Payment_Behaviour_High_spent_Medium_value_payments',
    'Payment_Behaviour_High_spent_Small_value_payments',
    'Payment_Behaviour_Low_spent_Large_value_payments',
    'Payment_Behaviour_Low_spent_Medium_value_payments',
    'Payment_Behaviour_Low_spent_Small_value_payments',
    'ToL_No Data', 'ToL_auto loan', 'ToL_credit-builder loan',
    'ToL_debt consolidation loan', 'ToL_home equity loan', 'ToL_mortgage loan',
    'ToL_not specified', 'ToL_payday loan', 'ToL_personal loan', 'ToL_student loan'
]
occupations = ['Occupation_Architect', 'Occupation_Developer', 'Occupation_Doctor',
    'Occupation_Engineer', 'Occupation_Entrepreneur', 'Occupation_Journalist',
    'Occupation_Lawyer', 'Occupation_Manager', 'Occupation_Mechanic',
    'Occupation_Media_Manager', 'Occupation_Musician', 'Occupation_Scientist',
    'Occupation_Teacher', 'Occupation_Writer']
occupation_names = {
    'Occupation_Architect': 'Architect',
    'Occupation_Developer': 'Developer',
    'Occupation_Doctor': 'Doctor',
    'Occupation_Engineer': 'Engineer',
    'Occupation_Entrepreneur': 'Entrepreneur',
    'Occupation_Journalist': 'Journalist',
    'Occupation_Lawyer': 'Lawyer',
    'Occupation_Manager': 'Manager',
    'Occupation_Mechanic': 'Mechanic',
    'Occupation_Media_Manager': 'Media Manager',
    'Occupation_Musician': 'Musician',
    'Occupation_Scientist': 'Scientist',
    'Occupation_Teacher': 'Teacher',
    'Occupation_Writer': 'Writer'}
loan_types = ['ToL_No Data', 'ToL_auto loan', 'ToL_credit-builder loan',
    'ToL_debt consolidation loan', 'ToL_home equity loan', 'ToL_mortgage loan',
    'ToL_not specified', 'ToL_payday loan', 'ToL_personal loan', 'ToL_student loan']
correct_column_order = [
    'Occupation_Architect', 'Occupation_Developer', 'Occupation_Doctor',
    'Occupation_Engineer', 'Occupation_Entrepreneur', 'Occupation_Journalist',
    'Occupation_Lawyer', 'Occupation_Manager', 'Occupation_Mechanic',
    'Occupation_Media_Manager', 'Occupation_Musician', 'Occupation_Scientist',
    'Occupation_Teacher', 'Occupation_Writer', 'Credit_Mix_Good',
    'Credit_Mix_Standard', 'Payment_of_Min_Amount_No', 'Payment_of_Min_Amount_Yes',
    'Payment_Behaviour_High_spent_Medium_value_payments',
    'Payment_Behaviour_High_spent_Small_value_payments',
    'Payment_Behaviour_Low_spent_Large_value_payments',
    'Payment_Behaviour_Low_spent_Medium_value_payments',
    'Payment_Behaviour_Low_spent_Small_value_payments', 'Age', 'Annual_Income',
    'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
    'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
    'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
    'Amount_invested_monthly', 'Monthly_Balance', 'ToL_No Data', 'ToL_auto loan',
    'ToL_credit-builder loan', 'ToL_debt consolidation loan', 'ToL_home equity loan',
    'ToL_mortgage loan', 'ToL_not specified', 'ToL_payday loan', 'ToL_personal loan',
    'ToL_student loan']
loan_type_names = {
    'ToL_No Data': 'No Data',
    'ToL_auto loan': 'Auto Loan',
    'ToL_credit-builder loan': 'Credit Builder Loan',
    'ToL_debt consolidation loan': 'Debt Consolidation Loan',
    'ToL_home equity loan': 'Home Equity Loan',
    'ToL_mortgage loan': 'Mortgage Loan',
    'ToL_not specified': 'Not Specified',
    'ToL_payday loan': 'Payday Loan',
    'ToL_personal loan': 'Personal Loan',
    'ToL_student loan': 'Student Loan'
}

# Initialize user input dictionary with all columns set to 0
user_input = {column: 0 for column in dummy_columns}

# Collect user inputs for numeric features
numeric_features = [
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
    'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
    'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month',
    'Amount_invested_monthly', 'Monthly_Balance'
]

feature_names = {
    'Age': 'Age',
    'Annual_Income': 'Annual Income',
    'Monthly_Inhand_Salary': 'Monthly In-hand Salary',
    'Num_Bank_Accounts': 'Number of Bank Accounts',
    'Num_Credit_Card': 'Number of Credit Cards',
    'Interest_Rate': 'Interest Rate',
    'Num_of_Loan': 'Number of Loans',
    'Delay_from_due_date': 'Delay from Due Date',
    'Num_of_Delayed_Payment': 'Number of Delayed Payments',
    'Changed_Credit_Limit': 'Changed Credit Limit',
    'Num_Credit_Inquiries': 'Number of Credit Inquiries',
    'Outstanding_Debt': 'Outstanding Debt',
    'Credit_Utilization_Ratio': 'Credit Utilization Ratio',
    'Credit_History_Age': 'Credit History Age',
    'Total_EMI_per_month': 'Total EMI per Month',
    'Amount_invested_monthly': 'Amount Invested Monthly',
    'Monthly_Balance': 'Monthly Balance'}

# Create a Streamlit web application
st.title("Credit Score Prediction")

# Collect user inputs for numeric features
st.sidebar.title("Client profile")

for feature in numeric_features:
    value = st.sidebar.number_input(
        f"Enter {feature_names[feature]}:")
    user_input[feature] = value

# Collect user input for Occupation (choose one)
st.sidebar.title("Select Occupation")
occupation_choices = list(occupation_names.values())
selected_occupation = st.sidebar.selectbox("Choose your occupation:", occupation_choices)
for key, value in occupation_names.items():
    if value == selected_occupation:
        user_input[key] = 1

st.sidebar.title("Select Loan Types (Multiple Choices Allowed)")
loan_type_choices = list(loan_type_names.values())
selected_loan_types = st.sidebar.multiselect("Choose your loan types:", loan_type_choices)
for key, value in loan_type_names.items():
    if value in selected_loan_types:
        user_input[key] = 1


# Set selected loan types to 1 in user input
for loan_type in selected_loan_types:
    user_input[loan_type] = 1

# Create a DataFrame from user input
user_df = pd.DataFrame(user_input, index=[0])

# Reorder the columns in the user_df DataFrame
user_df = user_df[correct_column_order]

# Make a prediction using the trained model
user_prediction = best_xgb_model.predict(user_df)

# Map the predicted integer value to the corresponding label
original_labels = ['Good', 'Poor', 'Standard']
predicted_label = original_labels[user_prediction[0]]

label_colors = {
    'Good': 'green',
    'Poor': 'red',
    'Standard': 'orange'
}

# Calculate progress percentage based on the predicted label
progress_percentage = {
    'Good': 100,
    'Poor': 0,
    'Standard': 50
}.get(predicted_label, 0)

st.markdown(f"<h2 style='color: {label_colors[predicted_label]};'>{predicted_label}</h2>", unsafe_allow_html=True)

# Display progress bar and legend
st.progress(progress_percentage / 100.0)

st.image("rmobile.webp", use_column_width=True)

