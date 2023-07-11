from datetime import date,timedelta
import urllib.request
import os
        
"""
Method to retrieve fire risk parameters. Keeps only fwi.
"""

# Retrieve global fwi values. First look if a forecast has been made
# today for today. If not, look if a forecast has been made yesterday
# for today, etc until a forecast for today is retrieved. Save it to
# file named geosfile.

print("Get FWI for today...")

today = date.today()
day = -1
while True:
    try:
        day += 1
        imerominia = today - timedelta(days=int(day))
        url = f'https://portal.nccs.nasa.gov/datashare/GlobalFWI/v2.0/fwiCalcs.GEOS-5/Default/GEOS-5/{str(imerominia.year)}/{imerominia.strftime("%Y%m%d")}00/FWI.GEOS-5.Daily.Default.{imerominia.strftime("%Y%m%d")}00.{today.strftime("%Y%m%d")}.nc'
        if os.path.exists("geosfile"):
            os.remove("geosfile")
        urllib.request.urlretrieve(url, 'geosfile')
        break
    except:
        continue
print("done!")
