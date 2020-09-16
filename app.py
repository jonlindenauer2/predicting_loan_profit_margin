#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 2020, rev Sep 15 2020
This script contains Python code for deploying the Loan Yield Percent Predictor model on streamlit.io
@author: Jon Lindenauer
"""
import streamlit as st
import pickle
from PIL import Image

import sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

# page heading
st.title('')
st.header('Loan Profit Margin Predictor')

# main image background
background = Image.open('cash_money.jpg')
st.image(background, width=300)

# load pickled model, vectorizer and factorizer
model = pickle.load(open("my_pickled_model.p", "rb"))

# set up slide bars and parameters
app_score = st.sidebar.slider('Credit Score', min_value=600, max_value=920, value=600, step=10)
yrs_in_biz = st.sidebar.slider('Years in Business', min_value=0, max_value=30, value=0, step=1)
num_contracts = st.sidebar.slider('Number of Contracts', min_value=0, max_value=5, value=0, step=1)
delinq_61 = st.sidebar.slider('Delinquent > 60 Days', min_value=0, max_value=45, value=0, step=1)
equipment_cost = st.sidebar.slider('Equipment Cost ($)', min_value=1000, max_value=200000, value=1000, step=1000)
#late_charges = st.sidebar.slider('Late Charges', min_value=0, max_value=1, value=0, step=1)
#industry_TruckingandCourierServicesExceptAir = st.sidebar.slider('Trucking and Courier (no Air)', min_value=0, max_value=1, value=0, step=1)
#industry_LocalTrucking = st.sidebar.slider('Local Trucking', min_value=0, max_value=1, value=0, step=1)
#industry_OfficesandClinicsofDoctorsofMedicine = st.sidebar.slider('Doctors Office', min_value=0, max_value=1, value=0, step=1)
#industry_MiscellaneousAmusementandRecreation = st.sidebar.slider('Amusement and Rec', min_value=0, max_value=1, value=0, step=1)
#industry_OfficesandClinicsofDentists = st.sidebar.slider('Dentists Office', min_value=0, max_value=1, value=0, step=1)
#industry_EngineeringArchitecturalandSurveying = st.sidebar.slider('Engineering, Architectural and Surveying', min_value=0, max_value=1, value=0, step=1)


option = st.selectbox(
'Select the Industry:',
('None',
'Trucking',
'Dentists',
'Engineering',
'Contractors'
'Doctors')
)

if option == 'Trucking':
    industry_TruckingandCourierServicesExceptAir = 1
    industry_GeneralBuildingContractors_Nonresidential = 0
    industry_OfficesandClinicsofDoctorsofMedicine = 0
    industry_OfficesandClinicsofDentists = 0
    industry_EngineeringArchitecturalandSurveying = 0
elif option == 'Contractors':
    industry_TruckingandCourierServicesExceptAir = 0
    industry_GeneralBuildingContractors_Nonresidential = 1
    industry_OfficesandClinicsofDoctorsofMedicine = 0
    industry_OfficesandClinicsofDentists = 0
    industry_EngineeringArchitecturalandSurveying = 0
elif option == 'Doctors':
    industry_TruckingandCourierServicesExceptAir = 0
    industry_GeneralBuildingContractors_Nonresidential = 0
    industry_OfficesandClinicsofDoctorsofMedicine = 1
    industry_OfficesandClinicsofDentists = 0
    industry_EngineeringArchitecturalandSurveying = 0
elif option == 'Dentists':
    industry_TruckingandCourierServicesExceptAir = 0
    industry_GeneralBuildingContractors_Nonresidential = 0
    industry_OfficesandClinicsofDoctorsofMedicine = 0
    industry_OfficesandClinicsofDentists = 1
    industry_EngineeringArchitecturalandSurveying = 0
elif option == 'Engineering':
    industry_TruckingandCourierServicesExceptAir = 0
    industry_GeneralBuildingContractors_Nonresidential = 0
    industry_OfficesandClinicsofDoctorsofMedicine = 0
    industry_OfficesandClinicsofDentists = 0
    industry_EngineeringArchitecturalandSurveying = 1
else:
    industry_TruckingandCourierServicesExceptAir = 0
    industry_GeneralBuildingContractors_Nonresidential = 0
    industry_OfficesandClinicsofDoctorsofMedicine = 0
    industry_OfficesandClinicsofDentists = 0
    industry_EngineeringArchitecturalandSurveying = 0

st.write('You selected:', option)

# create features dict
input_dct = {'app_score': [app_score],
             'yrs_in_biz': [yrs_in_biz],
             'industry_TruckingandCourierServicesExceptAir': [industry_TruckingandCourierServicesExceptAir],
             'industry_OfficesandClinicsofDentists': [industry_OfficesandClinicsofDentists],
             'industry_EngineeringArchitecturalandSurveying': [industry_EngineeringArchitecturalandSurveying],
             'num_contracts': [num_contracts],
             'industry_GeneralBuildingContractors_Nonresidential': [industry_GeneralBuildingContractors_Nonresidential],
             'delinq_61': [delinq_61],
             'industry_OfficesandClinicsofDoctorsofMedicine': [industry_OfficesandClinicsofDoctorsofMedicine],
             'equipment_cost': [equipment_cost]
            }

input_df = pd.DataFrame.from_dict(input_dct)

prediction = model.predict(input_df.iloc[0:10])
pred_rnd = int(np.round(prediction + 3 * np.random.uniform(-1,1)))  # add random value so don't output exact values
pred_str = str(pred_rnd)
print_str = 'Predicted Profit is:' + ' ' + pred_str + '%'

rmse_str = '\u00B1' + '3% error'

# make the output message a title so it is BIG
st.title(print_str)

st.write(rmse_str)
