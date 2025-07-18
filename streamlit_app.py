import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Load the trained pipeline
import pickle

try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except Exception as e:
    st.error("Failed to load the model.")
    st.text(str(e))

# Team and city options
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England',
         'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London',
          'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 'St Lucia', 'Wellington',
          'Lauderhill', 'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai',
          'Nottingham', 'Southampton', 'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore',
          'Delhi', 'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
          'Christchurch', 'Trinidad']

# App Title
st.title('🏏 Cricket Score Predictor')

# Input widgets
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

city = st.selectbox('Select City', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score', min_value=0)
with col4:
    overs = st.number_input('Overs Completed (works for > 5 overs)', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10)

last_five = st.number_input('Runs Scored in Last 5 Overs', min_value=0)

# Prediction
if st.button('Predict Score'):
    if overs <= 0:
        st.error("Overs must be greater than 0 for prediction.")
    else:
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs

        # DataFrame for prediction
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [current_score],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'crr': [crr],
            'last_five': [last_five]
        })

        # Make prediction
        predicted_score = int(pipe.predict(input_df)[0])
        st.header(f"🏆 Predicted Final Score: {predicted_score}")
