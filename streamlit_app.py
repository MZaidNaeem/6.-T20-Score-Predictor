import streamlit as st
import pandas as pd
import pickle

# Set up the page
st.set_page_config(page_title="🏏 Cricket Score Predictor", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f7f7f7;
    }
    .stButton>button {
        background-color: #009999;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 40px;
        width: 100%;
    }
    .stNumberInput>div>div>input {
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title('🏏 T20 Cricket Score Predictor')

# Load the model
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except Exception as e:
    st.error("🚨 Failed to load the model.")
    st.text(str(e))

teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England',
         'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London',
          'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 'St Lucia', 'Wellington',
          'Lauderhill', 'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai',
          'Nottingham', 'Southampton', 'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore',
          'Delhi', 'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
          'Christchurch', 'Trinidad']

# Form for input
with st.form(key='prediction_form'):
    st.subheader("📋 Match Details")

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('🏏 Batting Team', sorted(teams))
    with col2:
        possible_bowling_teams = [team for team in teams if team != batting_team]
        bowling_team = st.selectbox('🎯 Bowling Team', sorted(possible_bowling_teams))

    city = st.selectbox('📍 Match City', sorted(cities))

    st.subheader("🧮 Match Situation")

    col3, col4, col5 = st.columns(3)
    with col3:
        current_score = st.number_input('Current Score', min_value=0)
    with col4:
        overs = st.number_input('Overs Completed', min_value=5.0, max_value=20.0, step=0.1)
    with col5:
        wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10)

    last_five = st.number_input('Runs in Last 5 Overs', min_value=0)

    submit_button = st.form_submit_button('Predict Score')

# Prediction logic
if submit_button:
    if overs <= 0:
        st.error("⛔ Overs must be greater than 0 for prediction.")
    elif wickets >= 10:
        st.error("❌ All wickets have fallen! Cannot predict further.")
    else:
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs

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

        try:
            predicted_score = int(pipe.predict(input_df)[0])
            st.success(f"🎯 Predicted Final Score: {predicted_score}")
        except Exception as e:
            st.error("🚫 Error in making prediction.")
            st.text(str(e))
