

import pandas as pd 
import numpy as np 
def preprocess(df):
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