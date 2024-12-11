import streamlit as st

# Título de la app
st.title("Calculadora de tiempos para Ironman")

# Mensaje inicial
st.write("Introduce tus datos para estimar el tiempo de cada segmento y el total.")

# Inputs del usuario
st.header("Datos de rendimiento")
swim_pace = st.text_input("Ritmo natación (min:seg / 100m):", value="2:00")  # Ritmo natación
bike_pace = st.text_input("Ritmo ciclismo (min/km):", value="4:00")  # Ritmo ciclismo
run_pace = st.text_input("Ritmo carrera (min/km):", value="5:00")  # Ritmo carrera

t1_time = st.number_input("Tiempo de transición T1 (min):", min_value=0.0, max_value=30.0, step=0.1)
t2_time = st.number_input("Tiempo de transición T2 (min):", min_value=0.0, max_value=30.0, step=0.1)

# Distancias estándar de un Ironman
swim_distance = 3.8  # en km
bike_distance = 180  # en km
run_distance = 42.2  # en km

# Función para convertir ritmo (min:seg) a tiempo en horas
def convert_pace_to_hours(pace, distance_m):
    try:
        minutes, seconds = map(int, pace.split(":"))
        total_seconds_per_100m = minutes * 60 + seconds
        total_time_seconds = (distance_m / 100) * total_seconds_per_100m
        return total_time_seconds / 3600  # Convertir segundos a horas
    except:
        st.error("Por favor, introduce el ritmo en formato correcto (min:seg).")
        return None

# Función para convertir el ritmo ciclismo/carrera (min/km) a tiempo en horas
def convert_pace_to_hours_for_bike_run(pace, distance_km):
    try:
        minutes, seconds = map(int, pace.split(":"))
        total_seconds_per_km = minutes * 60 + seconds
        total_time_seconds = distance_km * total_seconds_per_km
        return total_time_seconds / 3600  # Convertir segundos a horas
    except:
        st.error("Por favor, introduce el ritmo en formato correcto (min:seg).")
        return None

# Función para convertir tiempo en horas a horas y minutos
def convert_hours_to_hm(hours):
    hrs = int(hours)  # Extraer la parte entera (horas)
    mins = round((hours - hrs) * 60)  # Convertir la parte decimal a minutos
    return hrs, mins

# Cálculos de tiempo
if st.button("Calcular tiempo total"):
    swim_time = convert_pace_to_hours(swim_pace, swim_distance * 1000)  # Convertir distancia a metros
    if swim_time is not None:
        bike_time = convert_pace_to_hours_for_bike_run(bike_pace, bike_distance)  # Cálculo de tiempo ciclismo
        run_time = convert_pace_to_hours_for_bike_run(run_pace, run_distance)  # Cálculo de tiempo carrera

        # Convertir transiciones a horas
        t1_time_in_hours = t1_time / 60
        t2_time_in_hours = t2_time / 60

        # Sumar todos los tiempos en horas
        total_time = swim_time + bike_time + run_time + t1_time_in_hours + t2_time_in_hours

        # Convertir el tiempo total a horas y minutos
        total_hours, total_minutes = convert_hours_to_hm(total_time)
        swim_hours, swim_minutes = convert_hours_to_hm(swim_time)
        bike_hours, bike_minutes = convert_hours_to_hm(bike_time)
        run_hours, run_minutes = convert_hours_to_hm(run_time)
        t1_hours, t1_minutes = convert_hours_to_hm(t1_time_in_hours)
        t2_hours, t2_minutes = convert_hours_to_hm(t2_time_in_hours)

        # Mostrar resultados
        st.header("Resultados")
        st.write(f"Tiempo natación: {swim_hours} horas {swim_minutes} minutos")
        st.write(f"Tiempo ciclismo: {bike_hours} horas {bike_minutes} minutos")
        st.write(f"Tiempo carrera: {run_hours} horas {run_minutes} minutos")
        st.write(f"Tiempo de transición T1: {t1_hours} horas {t1_minutes} minutos")
        st.write(f"Tiempo de transición T2: {t2_hours} horas {t2_minutes} minutos")
        st.write(f"*Tiempo total estimado: {total_hours} horas {total_minutes} minutos*")
