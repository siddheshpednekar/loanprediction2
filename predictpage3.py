# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 09:32:37 2022

@author: siddh
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 00:40:00 2022

@author: siddh
"""





import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np

def load_model():
    with open('trainedmodel.sav', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

lr_model = data["model"]
oe_dependants = data["oe_dependants"]
oe_grad = data["oe_grad"]
oe_prop = data["oe_prop"]
sc = data["scalar"]

def show_predict_page():
    st.title("Loan Prediction")
    
    st.write("""### We need some information to predict Loan Status""")
    
    depcat = ['3+','2', '1', '0']
    grad = ['Not Graduate', 'Graduate']
    prop = ['Rural', 'Semiurban','Urban']
    
    gend = ['Male', 'Female']
    mar = ['Yes', 'No']
    selem = ['Yes', 'No']
    
    credhis = ['1.0', '0.0']
    
    dependants = st.selectbox('dependants', list(range(len(depcat))), format_func=lambda x: depcat[x])
    education = st.selectbox('education', list(range(len(grad))), format_func=lambda x: grad[x])
    property_area = st.selectbox('Property_Area', list(range(len(prop))), format_func=lambda x: prop[x])
    
    credit_history = st.selectbox('Credit History', credhis)
      
    gender = st.selectbox('Gender', gend)
    married = st.selectbox('Married', mar)
    self_emp = st.selectbox('Self Employed', selem)
    
    appl_inc = st.slider('Applicant Income', 0, 100000, 0 )
    co_appl_inc = st.slider('Coapplicant Income', 0, 100000, 0 )
    loan_amt = st.slider('Loan Amount', 0, 1000, 0)
    
    '''cat_list = ['Male', 'Female', 'Married', 'Unmarried', 'Not_semp', 'emp']         
    for i in cat_list:
           exec("%s = %d" % (i,0))'''
    Male = 0
    Female = 0 
    Married = 0
    Unmarried = 0
    Not_semp = 0
    emp = 0
    
    if gender=='Male':
            Male = 1
    elif gender=='Female':
            Female = 1
    else:
        pass
    
    if married == 'Yes':
        Married = 1
    elif married == 'No':
        Unmarried = 1
    else:
        pass
    
    if self_emp == 'Yes':
        emp = 1
    elif self_emp == 'No':
        Not_semp = 1
    else:
        pass
    
    
    prediction = st.button('Predict')
    
    if prediction:
        
       
        
        lst = [dependants, education, appl_inc,
                      co_appl_inc, loan_amt,
                      float(credit_history), property_area, Unmarried,
                      Married, Female, Male, Not_semp, emp]
        X = pd.DataFrame(lst)
        X = X.T
       
        X = np.array(X)
        mn = sc.transform(X)
        
        loan_status = lr_model.predict(mn)
        if loan_status[0] == "Y":
            st.write('### You are eligible for home loan')
        else:
            st.write('### You are not eligible for home loan')


