import streamlit as st
import pandas as pd
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from src.utils import *

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

    df_filter = df_merge[['Location', 'Swim Type', 'Bike Type', 'Run Type', 'Latitude', 'Longitude', 'Altitude (m)', 'Air Temperature (°C)', 'Water Temperature (°C)', 'EventCountry', 'EventLocation']].drop_duplicates()
    df_model = pd.merge(df_data, df_filter, on=['EventLocation', 'EventCountry'], how='inner')
    df_model = pd.merge(df_model, df_types_per_location, on='EventLocation_Encoded', how='inner')
    df_model['Is_Local'] = (df_model['Country'] == df_model['EventCountry']).astype(int)
    df_model['Distance from Country Center (m)'] = df_model.apply(calculate_distance_meters, axis=1, df=df_model)

    return df_model[features]

def seconds_to_hms(seconds, format='%H:%M:%S'):
    return pd.to_datetime(seconds, unit='s').strftime(format)

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
st.title("Predicción de tiempos para Ironman 70.3 🏊🏽‍♀️🚴🏽‍♀️🏃🏽‍♀️")

# Cargar valores únicos de las columnas requeridas
event_locations = sorted(load_unique_values('df_merged_filtered.csv', 'EventLocation'))
genders = sorted(load_unique_values('df_merged_filtered.csv', 'Gender'))
countries = sorted(load_unique_values('df_merged_filtered.csv', 'Country'))

# Interfaz de usuario
st.subheader("Introduce los detalles")
age = st.slider("Edad", 18, 75, 30)
elite = st.checkbox("¿Eres atleta de élite?")
event = st.selectbox("Carrera", event_locations, index=31)
gender = st.selectbox("Género", genders, index=1)
country = st.selectbox("País", countries, index=83)

df_merged = pd.read_csv("df_merged_filtered.csv")

if st.button("Predecir tiempos"):

    df_event = df_merged[df_merged.EventLocation==event][['EventLocation', 'Location', 'Swim Type', 'Bike Type', 'Run Type',
                                                                            'Latitude', 'Longitude', 'Altitude (m)', 'Air Temperature (°C)',
                                                                            'Water Temperature (°C)', 'EventCountry']].drop_duplicates()    
    st.header(f"{event.upper()}")
    st.markdown(f"Location: {df_event['Location'].values[0]} | Country: {df_event['EventCountry'].values[0].upper()} | Altitude (m): {df_event['Altitude (m)'].values[0]} m")
    st.map(df_event, latitude="Latitude", longitude="Longitude")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Swim", f"{df_event['Swim Type'].values[0].upper()}")
    col2.metric("Bike", f"{df_event['Bike Type'].values[0].upper()}")
    col3.metric("Run", f"{df_event['Run Type'].values[0].upper()}")
    col4.metric("☀️ Avg. Air Temp", f"{df_event['Air Temperature (°C)'].values[0]} °C")
    col5.metric("💧 Avg. Water Temp", f"{df_event['Water Temperature (°C)'].values[0]} °C")

    st.subheader("Resultados ⏱️")
    df_data = pd.DataFrame({'Age': [age], 'Elite': [elite], 'EventLocation': [event], 'Gender': [gender], 'Country': [country]})

    swim_time = predict_time(model_swim, df_data)
    bike_time = predict_time(model_bike, df_data)
    run_time = predict_time(model_run, df_data)
    finishactive_time = predict_time(model_finishactivetime, df_data)

    st.markdown("🏊🏽‍♀️ **Natación** | Distancia: 1900 m")
    st.success(f"Tiempo: **{seconds_to_hms(swim_time)}**")
    st.success(f"Ritmo Medio: **{seconds_to_hms((swim_time / 1900) *100, '%M:%S')} /100m**")
               
    st.markdown("🚴🏽‍♀️ **Bicicleta** | Distancia: 90 km")
    st.success(f"Tiempo: **{seconds_to_hms(bike_time)}**")
    st.success(f"Velocidad Media: **{round(90 / (bike_time/3600), 1)} km/h**")

    st.markdown("🏃🏽‍♀️ **Carrera** | Distancia: 21.1 km")
    st.success(f"Tiempo: **{seconds_to_hms(run_time)}**")
    st.success(f"Ritmo Medio: **{seconds_to_hms(run_time/21.1, '%M:%S')} /km**")
    
    st.markdown(f"🏁 **Total** (Suma) | Distancia: 113 km")
    st.success(f"Tiempo: **{seconds_to_hms(swim_time+bike_time+run_time)}**")

    st.markdown(f"🏁 **Total** (Modelo) | Distancia: 113 km")
    st.success(f"Tiempo: **{seconds_to_hms(finishactive_time)}**")

    df_merged['FinishActiveTime'] = df_merged['RunTime'] + df_merged['SwimTime'] + df_merged['BikeTime']

    sns.set(style="white")

    fig = plt.figure(figsize=(10, 6))

    sns.histplot(data=df_merged[df_merged['EventLocation'] == event], x='SwimTime',  color='lightblue', alpha=0.5, label='SwimTime')
    p_swim = round(percentileofscore(df_merged[df_merged['EventLocation'] == event]['SwimTime'], swim_time), 1)
    plt.axvline(swim_time, color='lightblue', linestyle='--', label=f'p{p_swim}: {swim_time:.2f}')
    sns.histplot(data=df_merged[df_merged['EventLocation'] == event], x='BikeTime', color='orange', alpha=0.5, label='BikeTime')
    p_bike = round(percentileofscore(df_merged[df_merged['EventLocation'] == event]['BikeTime'], bike_time), 1)
    plt.axvline(bike_time, color='darkorange', linestyle='--', label=f'p{p_bike}: {bike_time:.2f}')
    sns.histplot(data=df_merged[df_merged['EventLocation'] == event], x='RunTime', color='green', alpha=0.5, label='RunTime')
    p_run = round(percentileofscore(df_merged[df_merged['EventLocation'] == event]['RunTime'], run_time), 1)
    plt.axvline(run_time, color='darkgreen', linestyle='--', label=f'p{p_run}: {run_time:.2f}')
    plt.xlabel("SwimTime, RunTime, BikeTime")
    plt.title("Tus tiempos respecto al resto de participantes")
    plt.legend()
    # plt.show()
    st.pyplot(fig)

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(data=df_merged[df_merged['EventLocation'] == event], x='FinishActiveTime', palette='purple', alpha=0.5, label='FinishActiveTime')
    p_finishactive = round(percentileofscore(df_merged[df_merged['EventLocation'] == event]['FinishActiveTime'], finishactive_time), 1)
    plt.axvline(finishactive_time, color='purple', linestyle='--', label=f'p{p_finishactive}: {finishactive_time:.2f}')
    plt.title("Tu tiempo total respecto al resto de participantes")
    plt.xlabel("FinishActiveTime")
    plt.legend()
    # plt.show()
    st.pyplot(fig)

else: 
    st.map(df_merged[['Latitude', 'Longitude']].drop_duplicates(), latitude="Latitude", longitude="Longitude")


