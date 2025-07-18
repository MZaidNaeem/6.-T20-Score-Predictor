import streamlit as st
import cloudpickle



import pandas as pd

# Load model
with open('final_model_pipeline.pkl', 'rb') as f:
    pipe = cloudpickle.load(f)


# Teams and cities (can be reduced if model was trained on fewer categories)
teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur',
    'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
    'Christchurch', 'Trinidad'
]

# App title
st.title('🏏 Cricket Score Predictor')

# Team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

# City selection
city = st.selectbox('Select City', sorted(cities))

# Match input stats
col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score', min_value=15)
with col4:
    overs = st.number_input('Overs Completed', min_value=5.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10)

# Runs in last 5 overs
last_five = st.number_input('Runs Scored in Last 5 Overs', min_value=15)

# Prediction button
if st.button('Predict Final Score'):
    if overs < 5:
        st.warning('⚠️ Prediction works best when over > 5.')
    elif bowling_team == batting_team:
        st.warning('⚠️ Bowling and batting team cannot be the same.')
    elif overs == 0:
        st.error("❌ Cannot divide by zero overs.")
    else:
        # Feature engineering
        balls_left = int(120 - (overs * 6))
        wickets_left = int(10 - wickets)
        crr = current_score / overs

        # DataFrame in correct order
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
            result = pipe.predict(input_df)
            st.success(f"🏆 Predicted Final Score: {int(result[0])}")
        except ValueError as e:
            st.error(f"🚫 Error: {str(e)}")
