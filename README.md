# CREDIT-RISK-MODELLING
Overview
Credit risk modeling plays a critical role in the banking industry by assessing the likelihood that a borrower will default on a loan. Banks rely on these models to minimize financial risk, optimize lending strategies, and protect against potential losses. The goal of this project is to develop a predictive model that can accurately assess the credit risk associated with loan applicants, helping banks make informed decisions about who to lend to,his project is designed to predict the creditworthiness of loan applicants by categorizing them into four risk categories (P1, P2, P3, P4), where P1 represents the best customers, and P4 represents the highest risk customers. The business team uses these categories to decide whether to approve or deny a loan application.

# In financial terms:

Loan as an Asset: A loan is considered an asset for the bank, as it represents a promise of repayment along with interest. However, this asset is at risk if the borrower defaults.

NPA (Non-Performing Asset): If a loan is not repaid according to the terms (e.g., the borrower misses payments), it becomes a Non-Performing Asset (NPA). A loan is typically classified as NPA after 90 days of delinquency. Managing NPAs is crucial to maintaining a healthy balance sheet for the bank.

Delinquency Days: The number of days an applicant has delayed in repaying previous loans can significantly impact their creditworthiness. Higher delinquency days can increase the likelihood of default, signaling higher credit risk.


# Timeframe
Start Date: September 2024
End Date: November 2024

# Key Features:
1.Data Preprocessing: The dataset was cleaned and preprocessed, handling missing values, categorical variables, and scaling numerical features.

2.Feature Engineering: Statistical techniques like ANOVA were used to determine significant features and improve model performance.

3.Model Development: Several baseline models (e.g., decision trees, gradient boosting, random forests) were experimented with, and the final model was optimized using XGBoost, a gradient boosting algorithm known for its performance on structured data.

4.Hyperparameter Tuning: XGBoost hyperparameters were optimized using techniques such as grid search and random search to find the best-performing configuration.

5.Model Evaluation: The model’s performance was evaluated using metrics like AUC, precision, recall, and accuracy to ensure its effectiveness for real-world credit risk assessment.

6.Deployment: The trained model was deployed using FlaskAPI, providing an API for real-time predictions and seamless integration with banking systems.
# Dataset
The project uses two datasets for training and evaluation:

Internal Bank Dataset: Real-world financial data containing information on customer credit behavior, loan amounts, repayment history, etc.
CIBIL Dataset: A publicly available dataset on credit information of individuals, typically used in financial assessments in India.
