import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Add a custom style for the app
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
    color: #333;
}
.sidebar .sidebar-content {
    background-color: #fafafa;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

st.write("""
# Simple Loan Prediction App
This app predicts the **loan approval**!
""")

st.sidebar.header('User Input')

def user_input_features():
    no_of_dependents = st.sidebar.text_input('Number of dependents', '5')
    
    education_options = ['Graduate', 'Not Graduate']
    education = st.sidebar.selectbox('Education', education_options)
    
    self_employed_options = ['Yes', 'No']
    self_employed = st.sidebar.selectbox('Self Employed', self_employed_options)
    
    income_annum = st.sidebar.text_input('Annual Income', '2500000')
    loan_amount = st.sidebar.text_input('Loan Amount', '500000')
    loan_term = st.sidebar.text_input('Loan Term (in months)', '30')
    cibil_score = st.sidebar.text_input('Credit Score', '100')
    total_asset_valuation = st.sidebar.text_input('Total Asset Valuation', '5000000')
    
    data = {
        'no_of_dependents': [int(no_of_dependents)],
        'education_Graduate': [int(education == 'Graduate')],
        'education_Not_Graduate': [int(education == 'Not Graduate')],
        'self_employed_ No': [int(self_employed == 'No')],
        'self_employed_ Yes': [int(self_employed == 'Yes')],
        'income_annum': [int(income_annum)],
        'loan_amount': [int(loan_amount)],
        'loan_term': [int(loan_term)],
        'cibil_score': [int(cibil_score)],
        'total_asset_valuation': [int(total_asset_valuation)],
    }
    
    return data

# Load the data
loan_raw = pd.read_csv('loan_data_2_final.csv')

# Select only first 13 columns (features) and assign to X
X = loan_raw.drop("loan_status", axis=1)

# Select last column (target) and assign to y
y = loan_raw["loan_status"]

# We want a split size of 80%-20%
size = 0.20

# A random seed number will just maintain repeatability for every run
# This ensures that each time you run your code with the same seed value, you will get the same data split.
seed = 7

# train_test_split() will split the data into 4 sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=seed)

CART = DecisionTreeClassifier()
CARTT = CART.fit(X_train, y_train)

# Get user input and create the user_input_df DataFrame
user_input_df = user_input_features()
user_input_df = pd.DataFrame(user_input_df)
user_input_df = user_input_df[X_train.columns]

# Predict the loan approval using the trained model
prediction = CARTT.predict(user_input_df)
prediction_prob = CARTT.predict_proba(user_input_df)

st.subheader('Prediction Probability')
st.write("""
This shows the probabilities of loan "approval" (class 1) and loan "rejected" (class 0), which are for example, [0.2, 0.8], respectively.
""")
st.write(prediction_prob)

# Display the prediction result
st.subheader('Prediction:')
st.write(f"The loan is {prediction[0]}")

# Add a footer
st.markdown("""
<hr style="border:1px solid #ccc">
<p style="text-align:center;">😊</p>
""", unsafe_allow_html=True)
