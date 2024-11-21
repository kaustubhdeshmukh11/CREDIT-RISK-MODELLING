from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from utils import preprocess
import joblib
import pandas as pd
import uvicorn
import numpy as np
import logging


app = FastAPI()


# Set up logging
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
# Load the model pipeline
#pipeline = joblib.load("model.joblib")
pipeline = joblib.load("model.joblib")

# Load the CSV data at the start
df = pd.read_csv('loan.csv')  # Replace with your actual CSV file path

# Define mappings and constants


# Preprocessing function

# Function to get credit score for a given uid
def get_credit_score(uid):
    # Filter the DataFrame to get the row with the specified uid
    result = df[df['uid'] == uid]
    
    if not result.empty:
        # Get the credit score from the first matching row
        credit_score = result.iloc[0]['Credit_Score']
        return credit_score
    else:
        return None

# Define the request model
class UIDRequest(BaseModel):
    uid: int

# FastAPI app setup


@app.post("/predict")
async def predict(request: UIDRequest):
    uid = request.uid  # Extract the UID from the request body
    
    # Filter rows corresponding to the UID
    filtered_data = df[df['uid'] == uid]

    if filtered_data.empty:
        raise HTTPException(status_code=404, detail="No data found for the provided UID")

    # Exclude both the 'uid' column and the last column from 'filtered_data'
    filtered_data = filtered_data.drop(columns=['uid']).iloc[:, :-1]

    # Preprocess the data before making predictions
    test_df_processed = preprocess(filtered_data)

    # Predict using the loaded pipeline
    prediction = pipeline.predict(test_df_processed)

    # Interpret the prediction result
    prediction_label = {
        0: "Approved flag p1",
        1: "Approved flag p0",
        2: "Approved flag p2",
        3: "Approved flag p3"
    }

    prediction_message = prediction_label.get(prediction[0], "customer category")

    # Get the credit score
    credit_score = get_credit_score(uid)

    if credit_score is None:
        credit_score_message = "Credit score not found."
    else:
        credit_score_message = f"Credit score: {credit_score}"

    logger.info(f"UID: {uid}, Prediction: {prediction_message}, {credit_score_message}")

    # Return the prediction and credit score in the response
    return {
        "uid": uid,
        "prediction": prediction_message,
        "credit_score": credit_score_message
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("testapi:app", host="127.0.0.1", port=8000, reload=True)


