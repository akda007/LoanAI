import pandas as pd
import numpy as np
from keras import models
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib  # Used to save and load scalers and encoders

# Load the trained model
model = models.load_model("loan_approval_model.h5")

# Load the previously saved scaler and label encoders
scaler = joblib.load("scaler.pkl")
encoder_gender = joblib.load("encoder_gender.pkl")
encoder_defaults = joblib.load("encoder_defaults.pkl")

def preprocess_input(data):
    # Apply the same LabelEncoder used during training
    data['person_gender'] = encoder_gender.transform(data['person_gender'])
    data['previous_loan_defaults_on_file'] = encoder_defaults.transform(data['previous_loan_defaults_on_file'])
    
    # Apply the same StandardScaler used during training
    num_columns = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 
                   'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
    data[num_columns] = scaler.transform(data[num_columns])
    
    return data.values

# Example input data (from user)
input_data = pd.DataFrame([{
    "person_age": 30.0,
    "person_gender": "male",
    "person_education": "Master",
    "person_income": 50000.0,
    "person_emp_exp": 5,
    "person_home_ownership": "OWN",
    "loan_amnt": 20000.0,
    "loan_intent": "PERSONAL",
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.4,
    "cb_person_cred_hist_length": 10,
    "credit_score": 700,
    "previous_loan_defaults_on_file": "No"
}])

# Preprocess the input data
processed_input = preprocess_input(input_data)

# Make the prediction
prediction = model.predict(processed_input)
result = (prediction > 0.5).astype(int)  # Convert to binary output

# Output the result
print("Loan Approval Prediction:", "Approved" if result[0][0] == 1 else "Not Approved")
