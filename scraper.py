import logging 
import pandas as pd
import re
import time

from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from src.utils import *

# Configure logging
logging.basicConfig(level=logging.INFO, format="{asctime} | {levelname} | {message}", style="{", datefmt="%Y-%m-%d %H:%M:%S")

# Configure Chrome WebDriver 
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Execute without opening window in web browser
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# URL base, se asume que las páginas están numeradas como 'page=1', 'page=2', etc.
base_url = 'https://www.ironman.com/im703-races?page='

# Número de páginas a recorrer (esto lo puedes ajustar según sea necesario)
total_pages = 10  # Ajusta este número según el número real de páginas

# Listas para almacenar los datos
race_names = []
locations = []
swim_types = []
bike_types = []
run_types = []
air_temps = []
water_temps = []
airports = []

def extract_data_from_page(html_content):
    """Extract relevant data from HTML content

    Args:
        html_content: Full source of the current HTML page
    """
    # Regex for data extraction
    race_names.extend(re.findall(r'<h3>(.*?)</h3>', html_content))
    locations.extend(re.findall(r'<p class="race-location">(.*?)</p>', html_content))
    swim_types.extend(re.findall(r'<div class="swim-type .*?"><p>Swim <br><b>(.*?)</b></p></div>', html_content))
    bike_types.extend(re.findall(r'<div class="bike-type .*?"><p>Bike <br><b>(.*?)</b></p></div>', html_content))
    run_types.extend(re.findall(r'<div class="run-type .*?"><p>Run <br><b>(.*?)</b></p></div>', html_content))
    air_temps.extend(re.findall(r'<div class="airTemp"><p>Avg. Air Temp <br><b>(.*?)</b></p></div>', html_content))
    water_temps.extend(re.findall(r'<div class="waterTemp"><p>Avg. Water Temp <br><b>(.*?)</b></p></div>', html_content))
    airports.extend(re.findall(r'<div class="airport"><p>Airport <br><b>(.*?)</b></p></div>', html_content))

# Scroll through all pages
for page_number in range(1, total_pages + 1):
    url = f"{base_url}{page_number}"
    logging.info(f"Processing page #{page_number}: {url}")

    # Open web
    driver.get(url)

    # Wait until web is available
    time.sleep(5)

    # Obtain HTML full content 
    html_content = driver.page_source

    # Extract data from web
    extract_data_from_page(html_content)

race_names = [race_name for race_name in race_names if race_name != ''] # Filter empty race_names

# Create dict with data
data = {
    "Race Name": race_names,
    "Location": locations,
    "Swim Type": swim_types,
    "Bike Type": bike_types,
    "Run Type": run_types,
    "Air Temperature": air_temps,
    "Water Temperature": water_temps,
    "Airport": airports
}

# Create DataFrame
df = pd.DataFrame(data)

filepath = Path("outputs/im703-races-data.csv")
# Create outputs folder if it does not exists
create_dir(filepath.parent)
# Dump to csv as output
df.to_csv(filepath, index=False)
logging.info(f"Dumped IRONMAN 70.3 Data in: {filepath}")

# Cerrar el WebDriver
driver.quit()
