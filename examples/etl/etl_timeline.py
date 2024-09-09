#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt


headS = '<!DOCTYPE html><html lang="en" > <head>\n\
<meta charset="UTF-8">\n\
<title>Tank&Rast correlation evolution</title>\n\
<link rel="stylesheet" href="css/normalize.min.css">\n\
<link rel="stylesheet" href="css/timeline_style.css">\n\
</head>\n\
<body>\n\
<div class="timeline-container" id="timeline-1">\n\
<div class="timeline-header">\n\
<h2 class="timeline-header__title">Tank & Rast</h2>\n\
<h3 class="timeline-header__subtitle">KPI evolution</h3>\n\
</div>\n\
<div class="timeline">\n\n'
footS = '</div></div>\
<div class="demo-footer">\n\
<a href="http://172.25.100.21:31027/geomadi/motorway.html" target="_blank">test days delivery</a>\n\
<br>\n\
<a href="http://172.25.100.21:31027/geomadi/train_reference.html" target="_blank">year delivery</a>\n\
</div>\n\
<script src="js/jquery.min.js"></script>\n\
<script  src="js/timeline_index.js"></script>\n\
</body></html>\n'


df = pd.read_csv(baseDir + "raw/tank/timeline.csv")
df.loc[:,"link"] = df['img'].apply(lambda x: "f_mot/" + x)
itemS = ''

for i,l in df.iterrows():
    itemS += '<div class="timeline-item" data-text="'+l['label']+'">\n'
    itemS += '<a href='+l['link']+'><div class="timeline__content"><img class="timeline__img" src="f_mot/'+l['img']+'"/></a>\n'
    itemS += '<h2 class="timeline__content-title">'+l['date']+'</h2>\n'
    itemS += '<p class="timeline__content-desc">'+l['comment']+'</p>\n'
    itemS += '</div></div>\n\n'

with open(baseDir + "www/timeline.html","w") as f:
    f.write(headS + itemS + footS)


headS = '<!DOCTYPE html><html lang="en" > <head>\n\
<meta charset="UTF-8">\n\
<title>Nissan product evolution</title>\n\
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/5.0.0/normalize.min.css">\n\
<link rel="stylesheet" href="css/timeline_style.css">\n\
</head>\n\
<body>\n\
<div class="timeline-container" id="timeline-1">\n\
<div class="timeline-header">\n\
<h2 class="timeline-header__title">Nissan</h2>\n\
<h3 class="timeline-header__subtitle">enter 2 exit</h3>\n\
</div>\n\
<div class="timeline">\n\n'
footS = '</div></div>\
<div class="demo-footer">\n\
<a href="http://172.25.100.21:31027/geomadi/route.html" target="_blank">project page</a>\n\
<br>\n\
<script src="js/jquery.min.js"></script>\n\
<script  src="js/timeline_index.js"></script>\n\
</body></html>\n'
    
df = pd.read_csv(baseDir + "raw/nissan/timeline.csv")
df.loc[:,"link"] = df['img'].apply(lambda x: "f_route/" + x)
itemS = ''

for i,l in df.iterrows():
    itemS += '<div class="timeline-item" data-text="'+l['label']+'">\n'
    itemS += '<a href='+l['link']+'><div class="timeline__content"><img class="timeline__img" src="f_route/'+l['img']+'"/></a>\n'
    itemS += '<h2 class="timeline__content-title">'+l['date']+'</h2>\n'
    itemS += '<p class="timeline__content-desc">'+l['comment']+'</p>\n'
    itemS += '</div></div>\n\n'

with open(baseDir + "www/timeline_nissan.html","w") as f:
    f.write(headS + itemS + footS)

    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
