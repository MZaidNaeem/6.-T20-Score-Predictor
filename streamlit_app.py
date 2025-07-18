import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# ✅ Redefine transformer and scaler structure manually
categorical_features = ['batting_team', 'bowling_team', 'city']
transformer = ColumnTransformer(
    transformers=[('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

scaler = StandardScaler(with_mean=False)
model = joblib.load("xgb_model.pkl")

# ✅ Rebuild pipeline
pipe = Pipeline(steps=[
    ('step1', transformer),
    ('step2', scaler),
    ('step3', model)
])

# Teams and cities
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
         'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']
cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
          'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
          'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
          'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
          'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi',
          'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
          'Christchurch', 'Trinidad']

st.title('🏏 Cricket Score Predictor')

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

city = st.selectbox('Select City', sorted(cities))

col3, col4, col5 = st.columns(3)
with col3:
    current_score = st.number_input('Current Score', min_value=15)
with col4:
    overs = st.number_input('Overs Completed', min_value=5.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=9)

last_five = st.number_input('Runs Scored in Last 5 Overs', min_value=15)

if st.button('Predict Final Score'):
    if overs < 5:
        st.warning('⚠️ Prediction works best when overs > 5.')
    elif bowling_team == batting_team:
        st.warning('⚠️ Bowling and batting team cannot be the same.')
    elif overs == 0:
        st.error("❌ Cannot divide by zero overs.")
    else:
        balls_left = int(120 - (overs * 6))
        wickets_left = int(10 - wickets)
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
            prediction = pipe.predict(input_df)[0]
            st.success(f"🏆 Predicted Final Score: {int(prediction)}")
        except Exception as e:
            st.error(f"🚫 Prediction failed: {str(e)}")
