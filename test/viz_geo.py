from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

#map = Basemap(projection='ortho', lat_0=50, lon_0=-100,resolution='l', area_thresh=1000.0)
#map = Basemap(projection='robin', lat_0=0, lon_0=-100,resolution='l', area_thresh=1000.0)
dMap = Basemap(projection='merc',llcrnrlat=47,urcrnrlat=55,llcrnrlon=5.5,urcrnrlon=15,lat_ts=20,resolution='l')
dMap.drawcoastlines()
dMap.drawcountries()
dMap.drawstates()
dMap.fillcontinents(color='coral',alpha=0.3)
#map.bluemarble()
#dMap.shadedrelief()
dMap.etopo()
dMap.drawmapboundary()
dMap.drawmeridians(np.arange(47,55,1))
dMap.drawparallels(np.arange(15,20,1))
lons, lats = dMap.makegrid(5, 5) # get lat/lons of ny by nx evenly space grid.
x, y = dMap(lons, lats) # compute map proj coordinates.

plt.show()
# lons = [-135.3318, -134.8331, -134.6572]
# lats = [57.0799, 57.0894, 56.2399]
# x,y = map(lons, lats)
# map.plot(x, y, 'bo', markersize=18)

fig = plt.figure(figsize=(11.7,8.3))
#Custom adjust of the subplots
plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,wspace=0.15,hspace=0.05)
ax = plt.subplot(111)
#Let's create a basemap around Belgium
m = Basemap(resolution='i',projection='merc', llcrnrlat=49.0,urcrnrlat=52.0,llcrnrlon=1.,urcrnrlon=8.0,lat_ts=51.0)
m.drawcountries(linewidth=0.5)
m.drawcoastlines(linewidth=0.5)
 
m.drawparallels(np.arange(49.,53.,1.),labels=[1,0,0,0],color='black',dashes=[1,0],labelstyle='+/-',linewidth=0.2) # draw parallels
m.drawmeridians(np.arange(1.,9.,1.),labels=[0,0,0,1],color='black',dashes=[1,0],labelstyle='+/-',linewidth=0.2) # draw meridians
 
# Let's add some earthquakes (fake here) :
 
lon = np.random.random_integers(11,79,1000)/10.
lat = np.random.random_integers(491,519,1000)/10.
depth = np.random.random_integers(0,300,1000)/10.
magnitude = np.random.random_integers(0,100,1000)/10.
 
# I'm masking the earthquakes present in most of the regions (illustration of masks usage) :
import numpy.ma as ma
Mlon = ma.masked_outside(lon, 5.6, 7.5)
Mlat = ma.masked_outside(lat,49.6,50.6)
lat = ma.array(lat,mask=Mlon.mask+Mlat.mask).compressed()
lon = ma.array(lon,mask=Mlon.mask+Mlat.mask).compressed()
depth =ma.array(depth,mask=Mlon.mask+Mlat.mask).compressed()
magnitude = ma.array(magnitude,mask=Mlon.mask+Mlat.mask).compressed()
