import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

st.set_page_config(page_title="Car Price Predictor", page_icon=":car:", layout="wide")

# Load the model and label encoders
try:
    with open('gradient_boosting_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoders.pkl', 'rb') as encoder_file:
        label_encoders = pickle.load(encoder_file)
except FileNotFoundError as e:
    st.error(f'Error loading files: {e}')
    st.stop()

# Load car data for OEM-model mapping
car_data_path = r'd:\Data Science\Python.VS\sownd\project 3 cardekho\final_cleaned_data.csv'  # Replace with your data path
car_data = pd.read_csv(car_data_path)

# Extract Feature Names
feature_names = model.feature_names_in_

# OEM-Model Mapping
oem_models = car_data.groupby('oem')['model'].unique().to_dict()

# UI Enhancements
st.title('Car Price Prediction App :car:')
st.markdown("Enter the car details below to predict its price.")
st.markdown("---")  # Separator

# Input Form (Dynamic)
input_values = {}

# OEM Dropdown
oem_options = sorted(car_data['oem'].unique())
selected_oem = st.selectbox('Select OEM (Manufacturer)', options=oem_options)
input_values['oem'] = selected_oem

# Filter car data based on OEM
filtered_data = car_data[car_data['oem'] == selected_oem]

# Model Dropdown (Dynamic)
model_options = oem_models.get(selected_oem, [])
selected_model = st.selectbox('Select Model', options=model_options)
input_values['model'] = selected_model

# Features to convert to dropdowns
dropdown_features = [
    'Registration Year', 'Year of Manufacture', 'modelYear',
    'Top Speed', 'Engine Displacement', 'ownerNo', 'Acceleration',
    'Insurance Validity', 'Fuel Type', 'Transmission', 'Ownership',
    'Color', 'Seating Capacity'
]

for feature in feature_names:
    if feature not in ['oem', 'model', 'Kms Driven']:
        if feature in dropdown_features:
            if feature in ['Registration Year', 'Year of Manufacture']:
                filtered_data[feature] = filtered_data[feature].astype(int) #convert to int.
            options = sorted(filtered_data[feature].unique())
            input_values[feature] = st.selectbox(f'Select {feature}', options=options)
        elif feature in label_encoders:  # Categorical feature
            options = list(label_encoders[feature].classes_)
            input_values[feature] = st.selectbox(f'Select {feature}', options=options)
        else:  # Numerical feature (you might need to refine this)
            input_values[feature] = st.number_input(f'Enter {feature}', value=0.0)


# Kms Driven Slider
min_kms = int(filtered_data['Kms Driven'].min())
max_kms = int(filtered_data['Kms Driven'].max())
selected_kms = st.slider('Kms Driven', min_value=min_kms, max_value=max_kms, value=min_kms)
input_values['Kms Driven'] = selected_kms

if st.button('Predict Price'):
    try:
        # Create Input DataFrame
        input_data = pd.DataFrame([input_values])

        # Reorder columns
        input_data = input_data[feature_names]

        # Label Encoding
        categorical_cols = [col for col in feature_names if col in label_encoders]
        for col in categorical_cols:
            if col in input_data.columns and col in label_encoders:
                input_data[col] = input_data[col].apply(lambda x: label_encoders[col].transform([x])[0])

        # Ensure all values are numeric.
        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(-1)

        # Progress bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)

        # Make Prediction
        prediction = model.predict(input_data)
        st.success(f'Predicted Car Price: ₹ {prediction[0]:,.2f}')

    except Exception as e:
        st.error(f'Prediction error: {e}')