import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set Streamlit page config first
st.set_page_config(page_title="Transaction Prediction App", layout="wide")

# Load data
data = pd.read_csv("transaction_dataset.csv", index_col=0)

# Preprocess data
categories = data.select_dtypes('O').columns.astype('category')
data.drop(data[categories], axis=1, inplace=True)
data.fillna(data.median(), inplace=True)
data.drop(columns=data.columns[16:], axis=1, inplace=True)

x = data.drop(columns=['FLAG', 'Index'], axis=1)
y = data['FLAG']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

# Train model
scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)

model_std1 = xgb.XGBClassifier()
model_std1.fit(x_train_std, y_train)

# Get accuracy (optional)
training_predict1 = model_std1.predict(x_train_std)
training_predict_accuracy = accuracy_score(training_predict1, y_train)

# Streamlit user interface
st.title("🔮 Transaction Prediction Web App")

st.markdown("### Model Accuracy on Training Data")
st.write(f"**Training Accuracy**: {training_predict_accuracy:.4f}")

st.markdown("""
This web application allows you to input transaction data and get a prediction of the transaction status (Flag). 
Please input the values for the features in the form below.
""")

# Input fields for user input (with feature names)
st.markdown("### Enter Feature Values")

feature_names = [
    "Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5", "Feature 6", "Feature 7", 
    "Feature 8", "Feature 9", "Feature 10", "Feature 11", "Feature 12", "Feature 13", "Feature 14"
]

input_data = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    input_data.append(value)

# Prepare the input data and make a prediction
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
input_data_std = scaler.transform(input_data_reshaped)

# Button to make prediction
if st.button('Predict Transaction Flag'):
    prediction = model_std1.predict(input_data_std)
    st.subheader("Prediction Result")
    if prediction[0] == 0:
        st.write("**Predicted FLAG: 0** (Transaction Failed)")
    else:
        st.write("**Predicted FLAG: 1** (Transaction Successful)")


