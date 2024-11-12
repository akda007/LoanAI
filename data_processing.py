import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_dataset():
    dataset = pd.read_csv("loan_data.csv", sep=',')

    # One-hot encode the categorical columns
    dataset = pd.get_dummies(
        dataset,
        columns=[
            'person_education',
            'person_home_ownership',
            'loan_intent',
        ],
        drop_first=True
    )

    # Initialize the encoders
    encoder_gender = LabelEncoder()
    encoder_defaults = LabelEncoder()

    # Fit the encoders and transform the relevant columns
    dataset['person_gender'] = encoder_gender.fit_transform(dataset['person_gender'])
    dataset['previous_loan_defaults_on_file'] = encoder_defaults.fit_transform(dataset['previous_loan_defaults_on_file'])

    # Define numerical columns to standardize
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

    # Initialize the scaler and fit it on numerical columns
    scaler = StandardScaler()
    dataset[num_columns] = scaler.fit_transform(dataset[num_columns])

    # Convert boolean columns to integers
    bool_columns = dataset.select_dtypes(include=[bool]).columns
    dataset[bool_columns] = dataset[bool_columns].astype(int)

    # Print dataset columns (optional)
    [print(x) for x in dataset.columns]

    # Define features (X) and target (Y)
    X = dataset.drop('loan_status', axis=1).values
    Y = dataset['loan_status'].values

    # Save the encoders and scaler to disk
    joblib.dump(encoder_gender, 'encoder_gender.pkl')
    joblib.dump(encoder_defaults, 'encoder_defaults.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Return train-test split
    return train_test_split(X, Y, test_size=0.2, random_state=42)


load_dataset()