# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:52:59 2024

@author: kaust
"""

import joblib
import numpy as np
import pandas as pd

# Define the preprocess function here, as used in the original pipeline
education_mapping = {
    '12TH': 2, 'GRADUATE': 3, 'SSC': 1, 'POST-GRADUATE': 4,
    'UNDER GRADUATE': 3, 'OTHERS': 1, 'PROFESSIONAL': 3
}
education_col_idx = 38
maritalstatus_col_idx = 37
gender_col_idx = 39
last_prod_enq2_col_idx = 40
first_prod_enq2_col_idx = 41
expected_columns = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 38, '37_Single', '39_M', '40_CC', '40_ConsumerLoan',
    '40_HL', '40_PL', '40_others', '41_CC', '41_ConsumerLoan', '41_HL',
    '41_PL', '41_others'
]
import pandas as pd

# Load the Excel file
file_path = 'cibil.xlsx'
df = pd.read_excel(file_path)

def get_credit_score(prospect_id):
    # Filter the DataFrame to get the row with the specified prospect_id
    result = df[df['PROSPECTID'] == prospect_id]
    
    if not result.empty:
        # Get the credit score from the first matching row
        credit_score = result.iloc[0]['Credit_Score']
        return credit_score
    else:
        return "Prospect ID not found."

# Example usage:

credit_score = get_credit_score(4)




# Preprocessing function
def preprocess(df):
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
    df.columns = list(range(df.shape[1]))  # Set default column indices

    # Apply mappings
    df[education_col_idx] = df[education_col_idx].map(education_mapping)

    # Dummy encode categorical columns
    df = pd.get_dummies(df, columns=[maritalstatus_col_idx, gender_col_idx, last_prod_enq2_col_idx, first_prod_enq2_col_idx], drop_first=True)

    # Ensure consistent columns with expected training data
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Convert all columns to float
    df = df.astype(float)

    return df[expected_columns]

# Load the joblib model
pipeline = joblib.load("model.joblib")

# Define the test input array
test = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 16, 16, 424, 38, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    38, 0, 0, 108, 0, 18000, 122, 0, 0, 0, 0, 0, 0, 'Married', 'SSC', 'F', 'ConsumerLoan', 'others'
], dtype=object).reshape(1, 42)

# Convert test data to DataFrame format
test_df = pd.DataFrame(test, columns=list(range(42)))

# Apply preprocessing manually since we're testing outside the pipeline
test_df_processed = preprocess(test_df)

# Predict using the loaded pipeline
prediction = pipeline.predict(test_df_processed)

# Output the prediction

if(prediction[0]==1):
    print("Approved flag p")
elif(prediction[0]==0):
    print("Approved flag p1")
elif(prediction[0]==2):
    print("Approved flag p2")
elif(prediction[0]==3):
    print("Approved flag p3")

    

print(f"Credit Score {credit_score}")




