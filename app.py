import streamlit as st
import pandas as pd
import joblib

#Load trained model
loaded_model = joblib.load(r"C:\Users\HAI\Downloads\LogisticRegression.pkl")

st.title("Student Performance Prediction")

# ---- User Input Fields ----
gender = st.selectbox("Gender", ["female", "male"])
race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.number_input("Reading Score", min_value=0, max_value=100)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100)

# ---- Encoding Mappings ----
gender_map = {'female': 0, 'male': 1}
race_ethnicity_map = {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4}
parental_education_map = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}
lunch_map = {'standard': 1, 'free/reduced': 0}
test_prep_map = {'none': 0, 'completed': 1}

# ---- Apply Encoding ----
input_data = pd.DataFrame({
    'gender': [gender_map[gender]],
    'race_ethnicity': [race_ethnicity_map[race_ethnicity]],
    'parental_level_of_education': [parental_education_map[parental_level_of_education]],
    'lunch': [lunch_map[lunch]],
    'test_preparation_course': [test_prep_map[test_preparation_course]],
    'reading_score': [reading_score],
    'writing_score': [writing_score]
})

st.write("Encoded Input Data", input_data)

# ---- Predict Button ----
if st.button("Predict"):
    try:
        prediction = loaded_model.predict(input_data)[0]
        st.success(f"Predicted Math Score: {prediction:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
