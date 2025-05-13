# AI-Based Cloud Learning Platform for Personalized Education Delivery
# Core Simulation in Python using Streamlit

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("AI-Based Cloud Learning Platform")
st.subheader("Personalized Education Delivery Simulation")

# Simulated dataset for students (features: time spent, quiz score, past performance)
students_data = pd.DataFrame({
    'time_spent': np.random.randint(10, 60, 100),
    'quiz_score': np.random.randint(0, 100, 100),
    'past_performance': np.random.randint(0, 100, 100),
    'level': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], 100)
})

# Encode target labels
students_data['level_code'] = students_data['level'].map({
    'Beginner': 0,
    'Intermediate': 1,
    'Advanced': 2
})

X = students_data[['time_spent', 'quiz_score', 'past_performance']]
y = students_data['level_code']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# User input for simulation
st.sidebar.header("Simulate Student Input")
time_spent = st.sidebar.slider("Time Spent (minutes)", 0, 120, 30)
quiz_score = st.sidebar.slider("Quiz Score", 0, 100, 50)
past_perf = st.sidebar.slider("Past Performance", 0, 100, 60)

input_df = pd.DataFrame({
    'time_spent': [time_spent],
    'quiz_score': [quiz_score],
    'past_performance': [past_perf]
})

# Predict level
prediction = model.predict(input_df)[0]
level_map = {0: 'Beginner', 1: 'Intermediate', 2: 'Advanced'}
predicted_level = level_map[prediction]

st.write("### Predicted Learning Level:", predicted_level)

# Recommend resources
st.write("### Recommended Content:")
if predicted_level == 'Beginner':
    st.markdown("- Introduction to Concepts")
    st.markdown("- Beginner Exercises")
elif predicted_level == 'Intermediate':
    st.markdown("- Applied Practice Problems")
    st.markdown("- Mini Projects")
else:
    st.markdown("- Advanced Research Papers")
    st.markdown("- Capstone Project Guidance")

st.success("Simulation complete! Adjust values in the sidebar to test personalization.")
