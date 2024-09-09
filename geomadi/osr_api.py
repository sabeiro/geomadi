import os, sys, gzip, random, json, datetime, re, io
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import zipfile

baseDir = os.environ['LAV_DIR']

endpoint = ['directions','isochrone','matrix','geocode','pois','elevation/line','mapsurfer','optimization']

cred = json.load(open(baseDir + "/motion/credenza/geomadi.json"))
url = 'https://api.openrouteservice.org/v2/isochrones/driving-car'
headers = {'Content-Type':'application/json; charset=utf-8','Accept':'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8','Authorization':cred['osr']['token']}
data = {"locations":[[8.681495,49.41461],[8.686507,49.41943]],"range":[300,200]}
resq = requests.post(url=url,headers=headers,json=data)

## vroom

req = {
  "vehicles": [
    {"id": 1,"start": [2.35044, 48.71764],"end": [2.35044, 48.71764],"capacity": [4],"skills": [1, 14],"time_window": [28800, 43200]},
    {"id": 1,"start": [2.35044, 48.71764],"end": [2.35044, 48.71764],"capacity": [4],"skills": [1, 14],"time_window": [28800, 43200]}
  ],
  "jobs": [
    {"id": 1,"service": 300,"delivery": [1],"location": [1.98935, 48.701],"skills": [1],"time_windows": [[32400, 36000]]},
    {"id": 2,"service": 300,"delivery": [1],"location": [1.98935, 48.701],"skills": [1],"time_windows": [[32400, 36000]]}
  ],
  "shipments": [
    {"amount": [1],"skills": [2],"pickup": {"id": 4,"service": 300,"location": [2.41808, 49.22619]},
      "delivery": {"id": 3,"service": 300,"location": [2.39719, 49.07611]}
    }
  ]
}

sol = {
    "code": 0
    ,"summary": {"cost": 18671,"unassigned": 0,"delivery": [5],"pickup": [1],"service": 1800,"duration": 18671,"waiting_time": 0,"distance": 316682},
    "unassigned": [],
    "routes": [
        {"vehicle": 1,"cost": 6526,"delivery": [3],"pickup": [0],"service": 900,"duration": 6526,"waiting_time": 0,"distance": 99924
         ,"steps": [
             {"type": "start","location": [2.35044,48.71764],"load": [3],"arrival": 30137,"duration": 0,"distance": 0},
             {"type": "job","location": [1.98935,48.701],"job": 1,"service": 300,"waiting_time": 0,"load": [2],"arrival": 32400,"duration": 2263,"distance": 34937}],
         "geometry": ""
        }
  ]
}
