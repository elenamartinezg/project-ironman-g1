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

def get_df_model(df_data, features):
    # Read DataFrames
    df_country_freqs = pd.read_csv("df_country_freqs.csv")
    df_event_location_freq = pd.read_csv("df_event_location_freq.csv")
    df_event_country_freq = pd.read_csv("df_event_country_freq.csv")
    df_types_per_location = pd.read_csv('df_types_per_location.csv')
    df_types_per_location = df_types_per_location.drop(columns=['Unnamed: 0'], errors='ignore')
    df_merge = pd.read_csv("df_merge_final.csv")

    df_country_of_location = df_merge[['EventCountry', 'EventLocation']].drop_duplicates().sort_values('EventCountry')
    df_data['Gender_M'] = df_data['Gender'].replace({"F": 0, "M": 1})
    df_data = pd.merge(df_data, df_country_freqs[['Country', 'Country_Encoded']], on='Country', how='left')
    df_data = pd.merge(df_data, df_event_location_freq[['EventLocation', 'EventLocation_Encoded']], on='EventLocation', how='left')
    df_data['EventCountry'] = df_country_of_location[df_country_of_location['EventLocation'] == event]['EventCountry'].values[0]
    df_data = pd.merge(df_data, df_event_country_freq[['EventCountry', 'EventCountry_Encoded']], on='EventCountry', how='left')

    age_limits = [0, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    age_groups = ['00', '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89']

    def get_age_band_and_group(age):
        age_band = max([limit for limit in age_limits if limit <= age])
        idx = age_limits.index(age_band)
        age_group = age_groups[idx]
        
        return age_band, age_group

    df_data[['AgeBand', 'AgeGroup']] = df_data['Age'].apply(lambda age: pd.Series(get_age_band_and_group(age)))
    df_data['AgeBand'] = df_data['Elite'].apply(lambda x: 0 if x == True else df_data['AgeBand'])
    df_data['AgeGroup'] = df_data['Elite'].apply(lambda x: '00' if x == True else df_data['AgeGroup'])

    df_filter = df_merge[['Country','Location', 'Swim Type', 'Bike Type', 'Run Type', 'Latitude', 'Longitude', 'Altitude (m)', 'Air Temperature (°C)', 'Water Temperature (°C)', 'EventCountry', 'Distance from Country Center (km)', 'Distance from Country Center (m)', 'EventLocation']].drop_duplicates()
    df_model = pd.merge(df_data, df_filter, on=['EventLocation', 'Country', 'EventCountry'], how='inner')
    df_model = pd.merge(df_model, df_types_per_location, on='EventLocation_Encoded', how='inner')
    df_model['Is_Local'] = (df_model['Country'] == df_model['EventCountry']).astype(int)

    return df_model[features]

def seconds_to_hms(seconds):
    return pd.to_datetime(seconds, unit='s').strftime('%H:%M:%S')

# Predicción basada en modelos
def predict_time(model, df_data):
    if model is None:
        return "Modelo no disponible"
    try:
        features = model.get_booster().feature_names
        input_data = get_df_model(df_data, features)
        # Realizar predicción
        prediction = model.predict(input_data)
        return round(prediction[0], 2)
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        return "Error en la predicción"

# Cargar los modelos
model_swim = load_model('model_xgb_swim.pkl')
model_bike = load_model('model_xgb_bike.pkl')
model_run = load_model('model_xgb_run.pkl')
model_finishactivetime = load_model('model_xgb_finishactivetime.pkl')

# Configurar la app de Streamlit
st.title("Predicción de tiempos para Ironman 70.3")

# Cargar valores únicos de las columnas requeridas
event_locations = load_unique_values('df_merged_small.csv', 'EventLocation')
genders = load_unique_values('df_merged_small.csv', 'Gender')
countries = load_unique_values('df_merged_small.csv', 'Country')

# Interfaz de usuario
st.header("Introduce los detalles")
age = st.slider("Edad", 18, 70, 30)
elite = st.checkbox("¿Eres atleta de élite?")
event = st.selectbox("Carrera", event_locations)
gender = st.selectbox("Género", genders)
country = st.selectbox("País", countries)

if st.button("Predecir tiempos"):
    st.subheader("Resultados")
    df_data = pd.DataFrame({'Age': [age], 'Elite': [elite], 'EventLocation': [event], 'Gender': [gender], 'Country': [country]})

    swim_time = predict_time(model_swim, df_data)
    bike_time = predict_time(model_bike, df_data)
    run_time = predict_time(model_run, df_data)
    finishactive_time = predict_time(model_finishactivetime, df_data)

    st.write(f"*Tiempo Natación:* {seconds_to_hms(swim_time)}")
    st.write(f"*Tiempo Bicicleta:* {seconds_to_hms(bike_time)}")
    st.write(f"*Tiempo Carrera:* {seconds_to_hms(run_time)}")
    st.write(f"*Tiempo Total (Suma):* {seconds_to_hms(swim_time+bike_time+run_time)}")
    st.write(f"*Tiempo Total (Modelo):* {seconds_to_hms(finishactive_time)}")
