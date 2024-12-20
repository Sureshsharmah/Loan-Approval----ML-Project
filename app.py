import streamlit as st
import pandas as pd
import pickle as pk
import warnings
warnings.filterwarnings("ignore")

model = pk.load(open('model.pkl','rb'))
scaler = pk.load(open('scaler.pkl','rb'))

st.header("Loan Prediction App")

no_of_dept = st.slider("Choose No of dependents", 0, 5)
grad = st.selectbox("Choose Education", ["Graduated", "Non Graduated"])
self_emp = st.selectbox("Choose Self_Employed ?", ["Yes", "No"])
annual_income = st.slider("Choose Annual Income", 0, 10000000)
loan_amount = st.slider("Choose Loan Amount", 0, 10000000)
loan_dur = st.slider("Choose Loan Duration ", 0, 20)
cibil = st.slider("Choose Cibil Score", 0, 1000)
assets = st.slider("Choose Assets", 0, 10000000)

if grad == 'Graduated':
    grad_s = 0
else:
    grad_s = 1


if self_emp == 'No':
    emp_s = 0
else:
    emp_s = 1

if st.button("Predict"):
    pred_df = pd.DataFrame([[no_of_dept, grad_s, emp_s, annual_income, loan_amount, loan_dur, cibil, assets]],
                       columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'])
    pred_df = scaler.transform(pred_df)
    predict = model.predict(pred_df)
    if predict[0] == 1:
        st.markdown("Loan Is Approved")
        
    else:
        st.markdown("Loan Is Rejected")









