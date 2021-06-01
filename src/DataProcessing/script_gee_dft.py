"""
Author: Alejandro Miranda
Contributor: Cristobal Pais
This script generates a fire dataset with positive and negative observations
Spatial information is extracted from all ares.
"""

# Importations
import ee
import os

# Initialize GEE session
ee.Initialize()

## Step 1
# Import wildfires, create table, and define the area of analysis
# Load cities and wildfires dataset
city_file = "users/alemiranda/buf_city_gcs"
fires_file = "users/alemiranda/FireDB_2013_2015_gcs"

city = ee.FeatureCollection(city_file)
cityg = city.geometry()
citydiss = cityg.dissolve()
citydiss2 = ee.FeatureCollection(citydiss).geometry()

fires = ee.FeatureCollection(fires_file)

# Generate random points within the buffer area 
# Modify the random seed to obtain different random points
random = ee.FeatureCollection.randomPoints(citydiss2, 10, 17)

# Area analysis: detect fires
# Radious = 500 m
radio = 500
print("Buffer radious [m]:", radio)

# Util: Calculate the buffer for each point
def buffun(feature):
    return feature.buffer(radio)

# Buffer
buf = random.map(buffun)

# SET ID
leng = buf.toList(10).length().subtract(1)
idList = ee.List.sequence(0,leng)
fc = buf.set('id', idList)


## Step 2: Extract images and feature collection
var igbpPalette = [
  'green',  # Forest
  'blue',   # Waterbodies
  'red',    # PeatA
  'yellow', # Steppe
  'black',  # Clearland
  'gray',   # Degradedforest
  'white',  # Salinewaterbodies
  'brown',  # Bush
  'purple'  # PeatB
]

# Load lcover layer
lcover_file = "users/alemiranda/lc2014_30m"
img = ee.Image(lcover_file) 


# IMAGE - COLLECTION 
# Util: Create image collection
def colFunc(feat):
    dis = feat.get("id")
    clipped = img.clip(feat).set("id", dis)
    return clipped

# Apply function
imcol = ee.ImageCollection(fc.map(colFunc))


# Export individual images
featlist = fc.toList(fc.size())
featlist_len = len(featlist.getInfo())

for f in range(1,featlist_len):
    feat = ee.Feature(featlist.get(f))
    dis = feat.get('system:index')
    disS = dis.getInfo()
    name = "ID"      
 
    export = ee.batch.Export.image.toDrive(image= img,
                                           description=name+disS,
                                           folder="DFTopology",
                                           fileNamePrefix=name + disS,
                                           region=feat.geometry().bounds().getInfo()['coordinates'],
                                           scale=img.projection().nominalScale().getInfo())
    export.start()


# Export Feature collection: shape file.
export = ee.batch.Export.table.toDrive(
                                       collection= fc,
                                       description='DFTopology',
                                       fileFormat= 'SHP')
export.start()