import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

scaler = joblib.load("scaler.pkl")
# Set page config as the FIRST and ONLY call
st.set_page_config(layout="wide")

# Title and description
st.title("Restaurant Rating Prediction App")
st.caption("This app predicts restaurant rating.")
st.divider()

# Input fields
averagecost = st.number_input("Please enter the estimated average cost for two people")
tablebooking = st.selectbox("Restaurant has table booking?", ["Yes", "No"])
onlinedelivery = st.selectbox("Restaurant has online delivery?", ["Yes", "No"])
pricerange = st.selectbox("What is the price range? (1 = Cheapest, 4 = Most Expensive)", [1, 2, 3, 4])

predictbutton = st.button("Predict the review!")

st.divider()

model = joblib.load("model.pkl")

bookingstatus = 1 if tablebooking=="Yes" else 0

deliverystatus = 1 if onlinedelivery=="Yes" else 0

values = [[averagecost, bookingstatus, deliverystatus, pricerange]]
my_values = np.array(values)

X = scaler.transform(my_values)

if predictbutton:
    st.snow()

    prediction = model.predict(X)

    if prediction < 2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction <4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")

