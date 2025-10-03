import streamlit as st
import pandas as pd
import joblib

st.title("🏖 Tourism Package Prediction")

model = joblib.load("deployment/best_model.pkl")

# Input form
age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Monthly Income", 1000, 1000000, 50000)
passport = st.selectbox("Passport", [0,1])
owncar = st.selectbox("Own Car", [0,1])

df = pd.DataFrame([[age, income, passport, owncar]], 
                  columns=["Age","MonthlyIncome","Passport","OwnCar"])

if st.button("Predict"):
    pred = model.predict(df)[0]
    st.write("✅ Will Purchase Package" if pred==1 else "❌ Will Not Purchase Package")
