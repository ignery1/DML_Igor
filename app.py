
import pandas as pd
import streamlit as st
from pycaret.regression import *
import numpy as np

#Load the trained model
model = load_model('model')

#Load the dataset
dataset = pd.read_csv('estudantes.csv')

#Create a sidebar with input fields for the student's attributes
st.sidebar.subheader('Define the student attributes for math score prediction')

reading_score = st.sidebar.number_input('Reading score')
writing_score = st.sidebar.number_input('Writing score')
gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))
ethnicity = st.sidebar.selectbox('Ethnicity', ('A', 'B', 'C', 'D', 'E'))
parental_education = st.sidebar.selectbox('Parental education', ('BD', 'SC', 'MD', 'AD', 'HS', 'SHS'))
test_preparation_course = st.sidebar.selectbox('Test preparation course', ('None', 'Complete'))
lunch = st.sidebar.selectbox('Lunch', ('Free/Reduced', 'Standard'))

#Convert the input data to a Pandas DataFrame
data_test = pd.DataFrame(dataset, columns=['reading_score', 'writing_score', 'gender', 'ethnicity', 'parental_education', 'test_preparation_course', 'lunch'])

#Make a prediction
prediction = model.predict(data_test)

#Display the prediction
st.subheader('Predicted math score:')
st.write(round(prediction[0], 2))
