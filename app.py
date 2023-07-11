import pandas as pd
from datetime import datetime, date, timedelta
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyowm
import xarray as xr
from scipy.spatial import cKDTree
from shapely.geometry import Point
import geopandas as gpd
from googletrans import Translator
import pytz

class tepgacast:
    
    """
    Object for weather prediction 
    Nt = number of hourly time steps
    Area of interest is defined by minlat,minlon,maxlat,maxlon
    mintime = time now
    maxtime = time now + Nt number of hours
    """

    def __init__(self, lat, lon, date):

        self.lat = lat
        self.lon = lon
        self.date = date

    def wmo_translation(data):

        # Translate wmo codes
        wmo_codes = {0:'Clear sky',
                    1:'Mainly clear',
                    2:'Partly cloudy',
                    3:'Overcast',
                    45:'Fog',
                    48:'Depositing rime fog',
                    51:'Drizzle light intensity',
                    53:'Drizzle moderate intensity',
                    55:'Drizzle dense intensity',
                    56:'Freezing Drizzle light intensity',
                    57:'Freezing Drizzle dense intensity',
                    61:'Rain slight intensity',
                    63:'Rain moderate intensity',
                    65:'Rain heavy intensity',
                    66:'Freezing rain light intensity',
                    67:'Freezing rain heavy intensity',
                    71:'Snow fall slight intensity',
                    73:'Snow fall moderate intensity',
                    75:'Snowfall heavy intensity',
                    77:'Snow grains',
                    80:'Rain showers slight',
                    81:'Rain showers moderate',
                    82:'Rain showers violent',
                    85:'Snow showers slight',
                    86:'Snow showers heavy',
                    95:'Thunderstorm slight or moderate',
                    96:'Thunderstorm with slight hail',
                    99:'Thunderstorm with heavy hail'}
        
        for code in wmo_codes.keys():
            data.weathercode = data.weathercode.apply(lambda x: wmo_codes.get(code) if x==code else x)

        return data

    def json2df(url_current):

        """
        Convert to DataFrame
        """

        dfcur = pd.read_json(url_current)

        if 'daily' in dfcur.columns:
            df = dfcur.daily.to_frame().T
            Nrows = len(df.time[0])
        elif 'hourly' in dfcur.columns:
            df = dfcur.hourly.to_frame().T
            Nrows = len(df.time[0])
        else:
            df = dfcur.results.to_frame().T
            print(df)
            Nrows = len(df.results)
            
        arr = []
        for i in range(Nrows):
            col = []
            for column in df.columns:
                col.append(df[column][0][i])
            arr.append(col)
        data = pd.DataFrame(columns=df.columns, data=arr)
        data.time = (data['time']).apply(lambda d: pd.to_datetime(str(d)))

        return data
    
    def day_cast(self, model):

        """
        Method for weather prediction at a point 
        with coordinates lat,lon.

        model:  best_match
                ecmwf_ifs04
                gfs_seamless
                gfs_global
                icon_seamless
                gem_seamless
                jma_seamless
                meteofrance_seamless
        """
        forecast_days = (self.date - date.today() + timedelta(days=1)).days

        url = (f'https://api.open-meteo.com/v1/forecast?latitude={self.lat}&longitude={self.lon}'
               f'&daily=weathercode,temperature_2m_max,temperature_2m_min,sunrise,sunset&models={model}'
               f'&forecast_days={forecast_days}&timezone=Europe%2FMoscow')
        data = tepgacast.json2df(url)
        data = tepgacast.wmo_translation(data)

        data = data.rename(columns={'time':'date'})
        data.sunrise = (data['sunrise']).apply(lambda d: pd.to_datetime(str(d)))
        data.sunset = (data['sunset']).apply(lambda d: pd.to_datetime(str(d)))
        data['sunrise'] = data['sunrise'].dt.strftime('%H:%M')
        data['sunset'] = data['sunset'].dt.strftime('%H:%M')

        data = data.loc[[forecast_days-1]].reset_index(drop=True)

        return data
    
    def flood_cast(self):

        forecast_days = (self.date - date.today() + timedelta(days=1)).days
        url = (f'https://flood-api.open-meteo.com/v1/flood?latitude={self.lat}&longitude={self.lon}'
               f'&daily=river_discharge&past_days=356&forecast_days={forecast_days}')
        data = tepgacast.json2df(url)
        data = data.rename(columns={'time':'date'})
        
        return data
    
    def hourly_cast(self, model):

        """
        Method for weather prediction at a point 
        with coordinates lat,lon.

        model:  best_match
            ecmwf_ifs04
            gfs_seamless
            gfs_global
            icon_seamless
            gem_seamless
            jma_seamless
            meteofrance_seamless
        """

        forecast_days = (self.date - date.today() + timedelta(days=1)).days
        url = (f'https://api.open-meteo.com/v1/forecast?latitude={self.lat}&longitude={self.lon}'
               f'&hourly=weathercode,precipitation,windspeed_10m,windgusts_10m,winddirection_10m,relativehumidity_2m&models={model}'
               f'&forecast_days={forecast_days}&timezone=Europe%2FMoscow')
        data = tepgacast.json2df(url)
        data = tepgacast.wmo_translation(data)

        data['time2'] = pd.to_datetime(data['time'], format='%y-%m-%d').dt.date
        data = data.loc[data.time2==self.date].reset_index(drop=True)

        data['winddirection_10m'] = 270 - data.winddirection_10m
        data['winddirection_10m'] = data.winddirection_10m.apply(lambda x: x+360 if x<0 else x)

        return data
    
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
    
    def utc2moscow(str_object):
        dtime = datetime.fromisoformat(str_object)
        tz = 'Europe/Moscow'
        dtime = dtime.astimezone(pytz.timezone(tz))
        return dtime
    
    def wmo2emoji(weathercode):
        x = weathercode.lower().split()
        if any(word in 'rain drizzle' for word in x):
            emoji = '🌧'
        elif 'thunderstorm' in x:
            emoji = '⛈'
        elif 'snow' in x:
            emoji ='❄'
        elif 'clear' in x:
            emoji = '☀️'
        else:
            emoji = '⛅'
        return emoji
    
    def current_weather(self):

        weatherI = pyowm.OWM('7b1365a8dca91e4361b7d4e116f4ca60')
        weatherManager = weatherI.weather_manager()

        weatherAtCoords = weatherManager.weather_at_coords(self.lat,self.lon)
        weather = weatherAtCoords.weather

        df = pd.DataFrame(columns=['Χρόνος','Περιγραφή καιρού', 'Θερμοκρασία',
                                   'Ταχύτητα ανέμου', 'Κατεύθυνση ανέμου', 'Ριπές ανέμου', 
                                   'Σχετική υγρασία', 'Βροχόπτωση', 'Χιονόπτωση'])

        dtime = weather.reference_time('iso')
        dtime = tepgacast.utc2moscow(dtime)

        wcode = weather.detailed_status
        wcode_trans = tepgacast.en2gr(wcode)
        temp = weather.temperature('celsius')['temp']
        wind = weather.wind('km_hour')
        wspd = round(wind['speed'],2)
        wdeg = round(wind['deg'],2)
        try:
            wgst = round(wind['gust'],2)
        except Exception:
            wgst = None
        
        rhum = weather.humidity
        rain = weather.rain.get('1h')
        snow = weather.snow.get('1h')

        df.loc[0] = [dtime, wcode_trans, temp,
                     wspd,  wdeg, wgst, 
                     rhum,  rain, snow]  # adding a row
        
        df = df.rename(index={0: ''})

        df['Κατεύθυνση ανέμου'] = 270 - df['Κατεύθυνση ανέμου']
        df['Κατεύθυνση ανέμου'] = df['Κατεύθυνση ανέμου'].apply(lambda x: x+360 if x<0 else x)

        one_call = weatherManager.one_call(lat=self.lat, lon=self.lon)
        national_weather_alerts = one_call.national_weather_alerts

        df_alert = pd.Series(index=["sender", "title", "description", 
                                    "start_time", "end_time"])
        if national_weather_alerts != None:
            for alert in national_weather_alerts:
                df_alert.sender = alert.sender                      # issuing national authority
                df_alert.title = tepgacast.en2gr(alert.title)                       # brief description
                df_alert.description = tepgacast.en2gr(alert.description)         # long description
                start_time = tepgacast.utc2moscow(alert.start_time(timeformat='iso'))
                end_time = tepgacast.utc2moscow(alert.end_time(timeformat='iso'))
                df_alert.start_time = start_time           # start time in UNIX epoch
                df_alert.end_time = end_time  # end time in ISO format
        
        return df.dropna(axis=1),df_alert

    def get_bestmodel(self, variable, days):

        today = date.today()
        maxdate = today - timedelta(days=1)
        mindate = today - timedelta(days=days)
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={self.lat}&longitude={self.lon}"
               f"&start_date={str(mindate)}&end_date={str(maxdate)}&hourly={variable}&timezone=Europe%2FMoscow")
        df_true = tepgacast.json2df(url)
        df_true = df_true.rename(columns={variable: variable + "_true"})

        models = ["gfs_seamless","ecmwf_ifs04",
                  "icon_seamless","gem_seamless",
                  "jma_seamless","meteofrance_seamless"]

        rmses = []
        for model in models:

            url = (f"https://api.open-meteo.com/v1/forecast?latitude={self.lat}&longitude={self.lon}"
                   f"&hourly={variable}&models={model}&past_days={str(days)}&forecast_days=0&timezone=Europe%2FMoscow")
            df_preds = tepgacast.json2df(url)
            df_preds = df_preds.rename(columns={variable: variable + "_preds"})

            df = pd.merge(df_true,df_preds, on='time')
            df = df.dropna().reset_index(drop=True)

            if variable == "winddirection_10m":
                rmse = mean_squared_error(np.cos(np.deg2rad(df[variable + "_true"])), np.cos(np.deg2rad(df[variable + "_preds"])))
            else:
                rmse = mean_squared_error(df[variable + "_true"], df[variable + "_preds"])
            rmses.append(rmse)

        x = np.array(rmses)
        min_idx = x.argmin()
        best_model = models[min_idx]

        return best_model
    
    def df2gdf(df):
        """Convert DataFrame to GeoDataFrame""" 
        geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
        gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry) 
        return gdf

    def get_nearest(gdA,gdB):
        """Find nearest"""
        nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
        nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist, idx = btree.query(nA, k=1)
        gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
        gdf = pd.concat([gdA.reset_index(drop=True), 
                        gdB_nearest, 
                        pd.Series(dist, name='dist')], axis=1)
        return gdf
    
    def categorize_fwi(gdf):

        """
        Translate GeoDataFrame's fwi column to category acccording to
        https://effis.jrc.ec.europa.eu/about-effis/technical-background/fire-danger-forecast 
        and save result to a new cat_fwi column.
        """

        def filter(x):
            if x<5.2:
                return 'Πολύ χαμηλός'
            elif (x>=5.2 and x<11.2):
                return 'Χαμηλός'
            elif (x<=21.3 and x>=11.2):
                return 'Μεσαίος'
            elif (x>21.3 and x<=38):
                return 'Υψηλός'
            elif (x>38 and x<=50):
                return 'Πολύ υψηλός'
            elif  x>50:
                return 'Ακραίος'
            else:
                return 'Άγνωστος'

        gdf['cat_fwi'] = gdf['fwi'].apply(filter)
        return gdf
    
    def en2gr(x):
        translator = Translator()
        x_trans = translator.translate(x, src='en', dest='el').text
        return x_trans

### run streamlit run app.py in the cmd ###

import plotly.express  as px
import streamlit as st

st.set_page_config(page_title='ΠΕΚΣ', 
                   page_icon='🌀',
                   layout='wide')

# # ---- SIDEBAR ----
st.sidebar.header("🔴Γεωκωδικοποίηση")
city = st.sidebar.text_input("Περιοχή: ", "Γεωργίου Κονδύλη 20, Μοσχάτο")

forecast_date = st.sidebar.date_input('Ημερομηνία πρόγνωσης:', 
                                      date.today(), date.today(), 
                                      date.today()+timedelta(days=16))

# Forecast Model Selection
d_convert = {
             "gfs_seamless"         : "gfs",
             "ecmwf_ifs04"          : "ifs",
             "icon_seamless"        : "icon",
             "gem_seamless"         : "gem",
             "jma_seamless"         : "jma",
             "meteofrance_seamless" : "a&a"
             }
def convert_option(opt):
    return d_convert[opt]
model = st.sidebar.selectbox('Επιλογή προγνωστικού μοντέλου:', 
                             d_convert.keys(), key='port', format_func=convert_option)


### --- MAINCODE ---
lat,lon = tepgacast.geocode(city)
df_now, df_alert = tepgacast(lat, lon, forecast_date).current_weather()
df_hourly = tepgacast(lat, lon, forecast_date).hourly_cast(model)
df_daily =  tepgacast(lat, lon, forecast_date).day_cast(model)
df_pnt = pd.DataFrame(data={'lat': [lat], 'lon': [lon]})

# fwi stuff
try:
    ds = xr.open_dataset('geosfile')# FWI DataFrame
    df_fwi = ds.to_dataframe().reset_index().drop(columns=['time', 'GEOS-5_DC', 'GEOS-5_DMC', 'GEOS-5_FFMC', 
                                                           'GEOS-5_ISI', 'GEOS-5_BUI', 'GEOS-5_DSR']).rename(columns={"GEOS-5_FWI": "fwi"})
    # df_fwi = df_fwi.fillna(0)
    dx = 0.2
    df_fwi = df_fwi.loc[(df_fwi.lat>lat-dx)&(df_fwi.lat<lat+dx)&(df_fwi.lon>lon-dx)&(df_fwi.lon<lon+dx)].reset_index(drop=True)
    gdf_fwi = tepgacast.df2gdf(df_fwi) # Convert DataFrames to GeoDataFrames
    gdf_pnt = tepgacast.df2gdf(df_pnt)
    gdf = tepgacast.get_nearest(gdf_pnt,gdf_fwi) # Get FWI value for location
    gdf = tepgacast.categorize_fwi(gdf)
except Exception:
    gdf = pd.DataFrame(data={"cat_fwi": ["error"]})

# create a map
st.sidebar.map(df_pnt)

# ---- MAINPAGE ----
st.title("🌀Πίνακας Ελέγχου Καιρικών Συνθηκών")
st.markdown("---")

st.subheader("Ειδοποίηση εθνικού φορέα")
df_alert.rename(index={"sender"      : "Αποστολέας",
                       "title"       : "Τίτλος",
                       "description" : "Περιγραφή",
                       "start_time"  : "Ώρα έναρξης",
                       "end_time"    : "Ώρα λήξης"}, inplace=True)

st.dataframe(df_alert.to_frame(name=""), use_container_width=True)
st.markdown("---")

left_col, right_col = st.columns(2)
 
with left_col:
    left_col.subheader("Καιρός τώρα")
    left_col.dataframe(df_now.T, use_container_width=True, 
                       hide_index=False, column_config=None)
with right_col:
    right_col.subheader('Ακριβέστερο μοντέλο ανά παράμετρο')
    d_convert2 = {
                  "do not run"          : " ",
                  "temperature_2m"      : "Θερμοκρασία",
                  "windspeed_10m"       : "Ταχύτητα ανέμου",
                  "winddirection_10m"   : "Κατεύθυνση ανέμου",
                  "precipitation"       : "Κατακρήμνιση",
                  "relativehumidity_2m" : "Σχετική υγρασία"
                }

    def convert_option(opt):
        return d_convert2[opt]

    variable = right_col.selectbox('Get best model:', 
                                    d_convert2.keys(), key='port2', 
                                    format_func=convert_option, label_visibility="collapsed")

    days = right_col.slider(label='ισχύς υπολογισμού', min_value=5, max_value=90, value=20, step=1)
    if variable != "do not run":
        with st.spinner('Υπολογισμός...'):
            best_model = tepgacast(lat, lon, forecast_date).get_bestmodel(variable=variable, days=days)
            right_col.success(d_convert[best_model])

st.markdown("---")

emoji = tepgacast.wmo2emoji(df_daily.weathercode.values[0])
st.subheader(f"Ημερήσια πρόγνωση {emoji}")


val = df_daily.weathercode.values[0]
val_trans = tepgacast.en2gr(val)
tval = str(df_daily.temperature_2m_min.values[0]) + " - " + \
       str(df_daily.temperature_2m_max.values[0])
df_daily_new = pd.Series({"Καιρός"             : val_trans,
                          "Θερμοκρασία"        : tval,
                          "Ανατολή"            : df_daily.sunrise.values[0],
                          "Δύση"               : df_daily.sunset.values[0],
                          "Κίνδυνος πυρκαγιάς" : gdf.cat_fwi[0]})
st.dataframe(df_daily_new.to_frame().T, hide_index=True, use_container_width=True)

# first_column, second_column, third_column, fourth_column, fifth_column = st.columns(5)
# with first_column:
#     val = df_daily.weathercode.values[0]
#     val_trans = tepgacast.en2gr(val)
#     st.markdown(f"Καιρός: {val_trans}")
# with second_column:
#     st.markdown(f"Θερμοκρασία: {df_daily.temperature_2m_min.values[0]} - {df_daily.temperature_2m_max.values[0]} C")
# with third_column:
#     st.markdown(f"Ανατολή: {df_daily.sunrise.values[0]}")
# with fourth_column:
#     st.markdown(f"Δύση: {df_daily.sunset.values[0]}")
# with fifth_column:
#     st.markdown(f"Κίνδυνος πυρκαγιάς: {gdf.cat_fwi[0]}")

# --- GRAPHS --- #

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots()
trace1 = go.Barpolar(r=df_hourly['windspeed_10m'], 
                     theta=df_hourly['winddirection_10m'], 
                     name='Άνεμος', marker_color='darkorange')
fig.add_trace(trace1)
try:
    trace2 = go.Scatterpolar(r=df_hourly['windgusts_10m'], 
                             theta=df_hourly['winddirection_10m'], 
                             name='Ριπές', marker_color='darkred', mode='markers')
    fig.add_trace(trace2)
except Exception:
    pass
fig['layout'].update(title='Άνεμος (km/h)🍃')

fig.update_layout(legend=dict(yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01))

fig.update_traces(width=5, selector=dict(type='barpolar'))

# df_hourly.precipitation=df_hourly.precipitation.shift(-1)
rhum_chart = px.box(df_hourly,
                    y='relativehumidity_2m',
                    title='Σχετική υγρασία (%)💧', template='seaborn').update_layout(yaxis_title=None)
# rain_chart.add_vline(datetime.utcnow(), fillcolor='seagreen')

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig, use_container_width=True)
right_column.plotly_chart(rhum_chart, use_container_width=True)

# # flood

# tomorrow = date.today()+timedelta(days=1)
# tomorrow = datetime.combine(tomorrow, datetime.min.time())

df_flood = tepgacast(lat, lon, forecast_date).flood_cast()
val_2day = df_flood.river_discharge.iloc[-1]

flood_hist = px.histogram(df_flood.river_discharge, log_y=True)
flood_hist.add_vline(val_2day, fillcolor='red')
flood_hist.update_layout(xaxis_title='Απορροή (m<sup>3</sup>/s)', 
                         yaxis_title="log(N)", 
                         title='Κίνδυνος πλημμύρας🌊', 
                         showlegend=False)

rain_chart = px.bar(df_hourly,'time','precipitation',title='Κατακρήμνιση (mm/h)☔',
                    color='precipitation',color_continuous_scale=px.colors.sequential.Blues).update_layout(yaxis_title=None,
                                                                                                           xaxis_title=None)
rain_chart.update_coloraxes(showscale=False)

left_column, right_column = st.columns(2)
left_column.plotly_chart(rain_chart, use_container_width=True, template='ggplot2')
right_column.plotly_chart(flood_hist, use_container_width=True, template='ggplot2')

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)
