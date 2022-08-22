#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 10:26:39 2022

@author: krishna
"""

'''  Housing Prices in California in 1990
       1) Depicting colour-coded areas  according to house value ranges
''' 

import pandas as pd 
''' 1990 Housing Prices 
    The housing price dataset for California may be obtained from https://github.com/krispad/Machine_Learning/houses.txt
'''

cal_houses1990 = pd.read_csv('https://github.com/krispad/Machine_Learning/houses.txt', sep = ' ') 
import geopandas as gpd # using the geospatial version of pandas

''' 1) link: https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html
                     On the page,  go to Counties > County Subdivisions and choose the state of California in the (shapefile) rectangular box 
                     Unzip the files and place them in a directory and generate the geospatial dataframe e.g. 
                          cal _shp2021 = gpd.read_file('~/Documents/'your_path'/cb_2021_06_cousub_500k.shp') 

    2) Note that the shape file cal_shp2021 is  a GeoDataFrame
'''

cal_shp2021 = gpd.read_file('"your path"/cb_2021_06_cousub_500k.shp') # A geodataframe (SHAPEFILE) for California Counties- u.s. geographic services ( internet link?



'''
     Constructing a GeoDataFrame from the cal_houses1990 dataset 
'''
geocal_houses = gpd.GeoDataFrame(data = cal_houses1990, geometry = gpd.points_from_xy(cal_houses1990.longitude, cal_houses1990.latitude), crs = 'EPSG:4269')
# Note that  crs = coordinate reference system 

geocal_houses_join = geocal_houses.sjoin(cal_shp2021, how = 'left') # houses data for California merged with California Shape file. 

'''
   Metropolitan Statistical Area(MSA) for California --- obtained from internet 
    Follow this link: https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html
       Go to Metropolitan and Micropolitan Statistical Areas > Places and then choose the California shapefile from the rectangular drop down box.
'''
msa_cal = gpd.read_file('"your path"/cb_2021_06_place_500k/cb_2021_06_place_500k.shp', crs = 'EPSG:4269')

#subsetting the data
msa_cal  = msa_cal[msa_cal['LSAD'] == '25']
msa_cal.columns = msa_cal.columns.str.lower()

# Calculating the range of house values in California
min_geocal, max_geocal = int(min(geocal_houses_join['house_value'])), int(max(geocal_houses_join['house_value']))
step = int((max_geocal - min_geocal)/4) # 5 value groups ; 4 gaps 

# Plot of California Housing Values.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (7 ,7))

cal_shp2021.plot(ax = ax, edgecolor = 'k', linewidth = 1, alpha = 1) # Shapefile california - 
msa_cal.plot(ax = ax, color = 'w', markersize = 3)

'''Extracting the msa's and selecting a subset that is to be identified on the map'''

msa_coords = msa_cal['geometry'].apply(lambda x: x.representative_point().coords[:])

a  = [x[0] for x in msa_coords]
msa_cal_gpd = msa_cal.copy()
msa_cal_gpd['coords'] = a
name_loc = [3, 11, 23, 50, 88, 406, 447, 380] # Selected cities in California 

# Numbering the selected cities in California
for idx, row in msa_cal_gpd.iloc[name_loc, :].iterrows():
    plt.annotate(text = idx, xy = row['coords'], color = 'xkcd:white', fontsize = 'small', fontstyle = 'italic', fontweight = 'extra bold')
    
# Colouring the groups of housing values
colouring =  ['k', 'r', 'g', 'm', 'y'] # of house values 
for i in range(min_geocal, max_geocal+1, step):
        
    a = (geocal_houses_join['house_value'] >= i) & (geocal_houses_join['house_value'] < i+step)
    dol1 = "{:,.2f}".format(i)
    dol2 = "{:,.2f}".format(i + step - 1)
    #print(sum(a== True))
    geocal_houses_join[a].plot(ax = ax, color = colouring[int((i-min_geocal)/step)], markersize = 1, label =f"From {dol1} to  {dol2}")
    
fig.suptitle('California - Housing Values 1990')
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.legend(loc = 'upper right', fontsize = 'small', frameon = False)

# Add an extra legend 
from matplotlib.legend import Legend
import matplotlib.patches as mpatches

dot = mpatches.Circle(xy = (0,0), color = 'w')
dot1 = mpatches.Circle(xy = (0, 0), radius = .5, color = 'xkcd:white')
names = list(map(lambda x : x, msa_cal_gpd.iloc[name_loc, :]['name']))
num_names = ['{}:{}'.format(x, y) for (x, y) in zip(msa_cal_gpd.iloc[name_loc, :].index, names)]

notes = ['Background: Blue'] + ['non-Blue colours represent Metro. Areas'] + ['Selected Cities'] +  num_names
hndles = [dot]*3 + [dot1]*8
leg = Legend(ax, handles = hndles, labels = notes, loc = 'lower left', frameon = False, fontsize = 'xx-small')
ax.add_artist(leg)
plt.show()


