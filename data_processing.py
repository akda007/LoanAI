import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_dataset():
    dataset = pd.read_csv("loan_data.csv", sep=',')

    dataset = pd.get_dummies(
        dataset,
        columns=[
            'person_education',
            'person_home_ownership',
            'loan_intent',
            ],
        drop_first=True
    )

    encoder = LabelEncoder()

    # 0 - Female | 1 - Male
    dataset['person_gender'] = encoder.fit_transform(dataset['person_gender'])

    # 0 - No | 1 - Yes
    dataset['previous_loan_defaults_on_file'] = encoder.fit_transform(dataset['previous_loan_defaults_on_file'])


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

    scaler = StandardScaler()

    dataset[num_columns] = scaler.fit_transform(dataset[num_columns])

    bool_columns = dataset.select_dtypes(include=[bool]).columns
    dataset[bool_columns] = dataset[bool_columns].astype(int)

    X = dataset.drop('loan_status', axis=1).values
    Y = dataset['loan_status'].values

    return train_test_split(X, Y, test_size=0.2, random_state=42)

