

def geocode(name):

    """
    Convert place to lat,lon,address
    """

    from geopy.geocoders import Nominatim
    loc = Nominatim(user_agent="someagent666")
    getLoc = loc.geocode(name)
    
    lat = getLoc.latitude
    lon = getLoc.longitude
    # address = getLoc.address

    return lat,lon

lat,lon = geocode('Γεωργίου Κονδύλη 20, Μοσχάτο')

import requests
import json
import streamlit as st

# Replace 'YOUR_API_KEY' with your actual API key from OpenWeatherMap
API_KEY = '7b1365a8dca91e4361b7d4e116f4ca60'

part = 'current,minutely,hourly,daily'
url = f'https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude={part}&appid={API_KEY}'

try:
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200:
        alerts = data.get('alerts')
        
        if alerts:
            st.text("Weather Alerts:")
            for alert in alerts:
                event = alert.get('event')
                description = alert.get('description')
                
                st.text(f"Event: {event}")
                st.text(f"Description: {description}")
        else:
            st.text("No weather alerts.")
    else:
        st.text(f"Request failed with status code {response.status_code}.")

except requests.exceptions.RequestException as e:
    st.text(f"An error occurred: {e}")