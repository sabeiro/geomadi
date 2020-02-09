import math
import urllib3
from io import StringIO, BytesIO
from PIL import Image
import shutil
import requests

def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

def getImageCluster(lat_deg, lon_deg, delta_lat,  delta_long, zoom):
    smurl = r"http://a.tile.openstreetmap.org/{0}/{1}/{2}.png"
    xmin, ymax =deg2num(lat_deg, lon_deg, zoom)
    xmax, ymin =deg2num(lat_deg + delta_lat, lon_deg + delta_long, zoom)

    Cluster = Image.new('RGB',((xmax-xmin+1)*256-1,(ymax-ymin+1)*256-1) ) 
    for xtile in range(xmin, xmax+1):
        for ytile in range(ymin,  ymax+1):
            try:
                imgurl=smurl.format(zoom, xtile, ytile)
                resq = requests.get(imgurl)
                tile = Image.open(BytesIO(resq.content))
                Cluster.paste(tile, box=((xtile-xmin)*256 ,  (ytile-ymin)*255))
            except: 
                print("Couldn't download image")
                tile = None

    return Cluster



a = getImageCluster(38.5, -77.04, 0.02,  0.05, 13)
fig = plt.figure()
fig.patch.set_facecolor('white')
plt.imshow(np.asarray(a))
plt.show()

from osmapi import OsmApi
MyApi = OsmApi()
print(MyApi.NodeGet(123))

api = OsmApi(api="https://api06.dev.openstreetmap.org",username=cred['osm']['user'],password=cred['osm']['user'])



