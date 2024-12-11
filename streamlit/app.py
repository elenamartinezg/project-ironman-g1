import streamlit as st
import pandas as pd
import pickle

from pathlib import Path

# Cargar los modelos
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error al cargar el modelo {model_path}: {e}")
        return None

# Cargar el archivo CSV y obtener valores únicos de una columna
@st.cache_data
def load_unique_values(csv_path, column_name):
    try:
        df = pd.read_csv(csv_path)
        return df[column_name].dropna().unique()
    except Exception as e:
        st.error(f"Error al cargar valores únicos de '{column_name}' desde {csv_path}: {e}")
        return []

# Predicción basada en modelos
def predict_time(model, age, event, gender, country):
    if model is None:
        return "Modelo no disponible"
    try:
        # Crear un DataFrame con las entradas
        input_data = pd.DataFrame({
            "age": [age], 
            "event": [event],
            "gender": [gender],
            "country": [country]
        })
        # Realizar predicción
        prediction = model.predict(input_data)
        return round(prediction[0], 2)
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        return "Error en la predicción"

# Cargar los modelos
model_swim = load_model('model_xgb_swim.pkl')
model_bike = load_model('model_xgb_bike.pkl')
model_run = load_model('model_rf_run.pkl')

# Configurar la app de Streamlit
st.title("Predicción de tiempos para Ironman 70.3")

# Cargar valores únicos de las columnas requeridas
event_locations = load_unique_values('df_merged_small.csv', 'EventLocation')
genders = load_unique_values('df_merged_small.csv', 'Gender')
countries = load_unique_values('df_merged_small.csv', 'Country')

# Interfaz de usuario
st.header("Introduce los detalles")
age = st.slider("Edad", 18, 70, 30)
event = st.selectbox("Carrera", event_locations)
gender = st.selectbox("Género", genders)
country = st.selectbox("País", countries)

if st.button("Predecir tiempos"):
    st.subheader("Resultados")
    swim_time = predict_time(model_swim, age, event, gender, country)
    bike_time = predict_time(model_bike, age, event, gender, country)
    run_time = predict_time(model_run, age, event, gender, country)

    st.write(f"*Tiempo natación:* {swim_time} minutos")
    st.write(f"*Tiempo bicicleta:* {bike_time} minutos")
    st.write(f"*Tiempo carrera:* {run_time} minutos")
