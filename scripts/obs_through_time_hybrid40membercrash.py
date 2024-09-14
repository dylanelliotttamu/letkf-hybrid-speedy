import numpy as np
#import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
#from netCDF4 import Dataset
#import cartopy as cart
#import cartopy.crs as ccrs
#from cartopy.util import add_cyclic_point
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from cartopy.mpl.geoaxes import GeoAxes
#import xarray as xr
#import glob
from datetime import datetime, timedelta 
#from dateutil.relativedelta import *
#from numba import jit
#import calendar
#from mpl_toolkits.axes_grid1 import AxesGrid
#import seaborn as sns
#import re 
import pandas as pd 

#load in data
output_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/dylan_nohup_files/single_crash_nohup.txt'
mylines = []                             
with open(output_file, 'rt') as myfile:  
    start_extraction = False             
    for myline in myfile:               
        if myline.startswith('== NUMBER OF OBSERVATIONS =============================================='):
            start_extraction = True      
        if start_extraction:
            mylines.append(myline.rstrip('\n'))
        if myline.startswith('========================================================================'):
            start_extraction = False     
#print(mylines[:100:])                           
bad_string = '== NUMBER OF OBSERVATIONS =============================================='
bad_string2 = '========================================================================' 
bad_string3 = '           U           V           T           Q          PS          RH' 
mylines = [string for string in mylines if string != bad_string]
mylines = [string for string in mylines if string != bad_string2]
mylines = [string for string in mylines if string != bad_string3]
#print(len(mylines))
mylines_corrected = mylines[::2]  
#print(len(mylines_corrected))
#print(mylines_corrected)
length = len(mylines_corrected)
converted_data = []
for element in mylines_corrected:
    values = element.split()
    values =[int(value) for value in values]
    converted_data.append(values)
df = pd.DataFrame(converted_data, columns=['U', 'V', 'T', 'Q', 'PS', 'RH'])

df['Datetime'] = pd.to_datetime(df.index, format='%Y-%m-%d %H')
df.set_index('Datetime', inplace=True)

startdate = datetime(2011,1,1,0)
enddate = datetime(2011,5,29,0)
time_slice = slice(startdate,enddate)
base = datetime(2011,1,1,0)
date_list = [base + timedelta(days=x/4) for x in range(length)]

df['Datetime'] = date_list
df.set_index('Datetime',inplace=True)
print(df)
#print(df.T) This makes a transpose of the matrix NOT TEMPERATURE

plt.plot(df.U,label='U_wind')
plt.plot(df.V,label='V_wind')
plt.plot(df['T'],label='Temperature')
plt.plot(df.Q,label='Specific Humidity')
plt.plot(df.PS,label='Pressure Surface')
plt.plot(df.RH,label='Relative Humidity')
plt.title('Quantity of Observations\nHybrid LETKF 40 member crash')
plt.xlabel('Date')
plt.ylabel('Observations')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.xlim([datetime(2011, 1, 1,0), datetime(2011, 5, 1,0)])
plt.ylim(0,4000)
plt.show()

#plot last few days
plt.plot(df.U,label='U_wind')
plt.plot(df.V,label='V_wind')
plt.plot(df['T'],label='Temperature')
plt.plot(df.Q,label='Specific Humidity')
plt.plot(df.PS,label='Pressure Surface')
plt.plot(df.RH,label='Relative Humidity')
plt.title('Quantity of Observations\nHybrid LETKF 40 member crash')
plt.xlabel('Date')
plt.ylabel('Observations')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim([datetime(2011, 5, 25,0), datetime(2011, 5, 30,0)])
plt.ylim(0,4000)
plt.show()












#end
