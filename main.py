import pandas as pd
from keras import models
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib  # Used to save and load scalers and encoders

# Load the trained model, scaler, and encoder
model = models.load_model("loan_approval_model.h5")
scaler = joblib.load('scaler.pkl')  # Load the saved scaler
encoder_gender = joblib.load('encoder_gender.pkl')  # Load the saved gender encoder
encoder_defaults = joblib.load('encoder_defaults.pkl')  # Load the saved loan defaults encoder

def preprocess_input(data):
    # Apply one-hot encoding to categorical columns
    data = pd.get_dummies(
        data,
        columns=[
            'person_education',
            'person_home_ownership',
            'loan_intent',
        ],
        drop_first=True
    )

    # Apply LabelEncoder to gender and previous loan defaults
    data['person_gender'] = encoder_gender.transform(data['person_gender'])
    data['previous_loan_defaults_on_file'] = encoder_defaults.transform(data["previous_loan_defaults_on_file"])

    num_columns = [
        'person_age',
        'person_income',
        'person_emp_exp',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'credit_score'
    ]

    # Standardize numerical columns using the pre-fitted scaler
    data[num_columns] = scaler.transform(data[num_columns])

    bool_columns = data.select_dtypes(include=[bool]).columns
    data[bool_columns] = data[bool_columns].astype(int)

    # Ensure all columns from training data are in the input data
    expected_columns = [
        'person_age', 'person_gender', 'person_income', 'person_emp_exp', 'loan_amnt', 
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 
        'previous_loan_defaults_on_file', 'person_education_Bachelor', 'person_education_Doctorate',
        'person_education_High School', 'person_education_Master', 'person_home_ownership_OTHER', 
        'person_home_ownership_OWN', 'person_home_ownership_RENT', 'loan_intent_EDUCATION',
        'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE'
    ]

    # Add missing columns with 0 if not present
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0

    # Return the data as a numpy array
    return data[expected_columns].to_numpy()

# Example input data (from user)
input_data = pd.DataFrame([{
    "person_age": None,  # Age is not provided, but will assume a default of 30 for this example
    "person_gender": "male",  # Gender is not an issue for loan eligibility
    "person_education": "High School",  # Education level could limit income potential, but not a deal breaker
    "person_income": 59864,  # Decent income, but on the lower end for larger loans
    "person_emp_exp": 0,  # No employment experience, which is a concern for loan approval
    "person_home_ownership": "RENT",  # Renting indicates a lack of home ownership, which could be a negative sign
    "loan_amnt": 25000,  # Large loan amount relative to income
    "loan_intent": "EDUCATION",  # Education is a valid reason for a loan and often considered favorably
    "loan_int_rate": 10.99,  # High interest rate, indicating a higher risk
    "loan_percent_income": 0.42,  # A relatively high percentage of income spent on loan, which might raise concerns
    "cb_person_cred_hist_length": 2,  # Short credit history, which is not ideal for assessing reliability
    "credit_score": 648,  # Moderate credit score — this could be acceptable, but still a concern for larger loans
    "previous_loan_defaults_on_file": "No"  # No loan defaults, which is a positive sign
}])

# Preprocess the input data
processed_input = preprocess_input(input_data)

# Make the prediction
prediction = model.predict(processed_input)
result = (prediction > 0.5).astype(int)  # Convert to binary output

# Output the result
print("Loan Approval Prediction:", "Approved" if result[0][0] == 1 else "Not Approved")
