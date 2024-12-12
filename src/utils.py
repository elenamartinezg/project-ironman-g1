from pathlib import Path
import pickle

def create_dir(dir):
    """Create directory if it does not exists

    Args:
        dir: Directory or folder path
    """
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)

def load_pickle(pickle_fpath):
    """Load Pickle File

    Args:
        pickle_fpath (Path): Input Pickle Filepath

    Returns:
        data: Loaded file
    """
    with open(pickle_fpath, "rb") as file:
        data = pickle.load(file)
    return data


def dump_pickle(data, pickle_fpath):
    """Dump Pickle File

    Args:
        data: Loaded file
        pickle_fpath (Path): Output Pickle Filepath

    """
    create_dir(pickle_fpath.parent)
    with open(pickle_fpath, "wb") as f:
        pickle.dump(data, f)

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# 1. Crear un diccionari amb els centres geogràfics de cada país
def get_country_centers(countries):
    """
    Obté el centre geogràfic (latitud, longitud) per a una llista de països utilitzant Geopy.
    """
    geolocator = Nominatim(user_agent="geoapi")
    country_centers = {}
    for country in countries:
        try:
            location = geolocator.geocode(country)
            if location:
                country_centers[country] = (location.latitude, location.longitude)
        except Exception as e:
            print(f"Error al trobar el centre geogràfic de {country}: {e}")
            country_centers[country] = None
    return country_centers

def calculate_distance_meters(row, df):
    """
    Calcula la distància (en km) entre el centre del país del participant i la ubicació de l'esdeveniment.
    """
    participant_country = row['Country']
    event_coords = (row['Latitude'], row['Longitude'])

    # 2. Llista única de països del DataFrame
    unique_countries = df['Country'].unique()

    # 3. Obtenir els centres geogràfics per als països del DataFrame
    country_centers = get_country_centers(unique_countries)
    
    # Obtenim el centre del país
    country_center = country_centers.get(participant_country)
    
    # Si no hi ha centre definit, retornem NaN
    if not country_center:
        return np.nan
    
    # Calculem la distància geodèsica
    return geodesic(country_center, event_coords).meters
        