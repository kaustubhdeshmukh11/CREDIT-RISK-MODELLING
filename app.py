from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import uvicorn
from utlis1111 import preprocess  # Ensure this import works as in your original code

# 1. Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="API for predicting loan approval flags and retrieving credit scores.",
    version="1.0"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Load Resources (Model & Data)
# It's best practice to load these at startup, but global variable works for simple scripts
try:
    pipeline = joblib.load("model_1.joblib")
    df = pd.read_csv('loan.csv') # Simulating a database
    logger.info("Model and Data loaded successfully.")
except Exception as e:
    logger.error(f"Error loading resources: {e}")
    raise RuntimeError("Could not load model or data file.")

# 3. Define Input Schema using Pydantic (This is a key FastAPI feature)
class RiskRequest(BaseModel):
    uid: float  # Using float as per your code logic

# Helper function (Same logic as your Flask code)
def get_credit_score(uid):
    result = df[df['uid'] == uid]
    if not result.empty:
        return result.iloc[0]['Credit_Score']
    return None

# 4. Define the Prediction Endpoint
@app.post("/predict")
async def predict(request: RiskRequest):
    uid = request.uid
    
    # Check if UID exists in our "Database" (CSV)
    filtered_data = df[df['uid'] == uid]
    
    if filtered_data.empty:
        raise HTTPException(status_code=404, detail="No data found for the provided UID")

    # Exclude 'uid' and the target column (last column) just like in your logic
    # Note: Ensure your CSV structure allows this drop. 
    # If 'uid' is not a column in the CSV but just an index, remove .drop(columns=['uid'])
    try:
        features_df = filtered_data.drop(columns=['uid']).iloc[:, :-1]
    except KeyError:
        # Fallback if uid is not a column
        features_df = filtered_data.iloc[:, :-1]

    # Preprocess
    try:
        test_df_processed = preprocess(features_df)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    # Predict
    try:
        prediction = pipeline.predict(test_df_processed)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")

    # Map Result
    prediction_label = {
        0: "Approved flag p1",
        1: "Approved flag p0",
        2: "Approved flag p2",
        3: "Approved flag p3"
    }
    prediction_message = prediction_label.get(prediction[0], "Unknown Category")

    # Get Credit Score
    credit_score = get_credit_score(uid)
    credit_score_message = f"Credit score: {credit_score}" if credit_score else "Credit score not found."

    logger.info(f"UID: {uid}, Prediction: {prediction_message}, {credit_score_message}")

    return {
        "uid": uid,
        "prediction": prediction_message,
        "credit_score": credit_score_message
    }

# Entry point for debugging
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
