#%pylab inline
import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from datetime import datetime
import scipy.spatial as spatial
from collections import OrderedDict as odict
gradMeter = 111122.19769899677
def plog(text):
    print(text)

from shapely.geometry import Point, Polygon, LineString
from shapely import geometry
from shapely.ops import cascaded_union

import geopandas as gpd
import shapely
junct = gpd.read_file(baseDir + "gis/nissan/junction_area.geojson")
dirc  = gpd.read_file(baseDir + "gis/nissan/count_dir.geojson")
dirA = [np.arctan(0.),np.arctan(np.pi/2.),np.arctan(np.pi),np.arctan(3.*np.pi/2.),np.arctan(2.*np.pi)]
def getAng(dx,dy):
    ang = int(np.arctan2(dy,dx)*2./np.pi + 0.5)
    cordD = ["east","north","west","south"]
    return cordD[ang], cordD[abs(2-ang)]

def getDir(dtx,dty):
    cordD = [("east","west"),("north","south"),("west","east"),("south","north")]
    ang = 1
    if(dtx < 0):
        ang = 0
    elif(dtx > 0):
        ang = 2
    elif(dty < 0):
        ang = 3
    return cordD[ang]

cordD = [("east","west"),("north","south"),("west","east"),("south","north")]
exits = gpd.read_file(baseDir + "gis/nissan/motorway_exit_axes.geojson")
fluxC = pd.DataFrame(index=range(0,4*24))
exits.loc[:,'exit'] = 0
exits.loc[:,'enter'] = 0
for i,ex in exits.iterrows():
    l = ex['geometry']
    inTile = [a.contains(Point(l.xy[0][0],l.xy[1][0])) for a in dirc['geometry']]
    outTile = [a.contains(Point(l.xy[0][1],l.xy[1][1])) for a in dirc['geometry']]
    dircI = dirc.loc[inTile]
    dircO = dirc.loc[outTile]
#    ang1, ang2 = getAng(l.xy[1][1] - l.xy[1][0],l.xy[0][1] - l.xy[0][0])
    dtx = dircI.iloc[0]['col_id'] - dircO.iloc[0]['col_id']
    dty = dircI.iloc[0]['row_id'] - dircO.iloc[0]['row_id']
    ang1, ang2 = getDir(dtx,dty)
    if(dtx!=0):
        if(dtx < 0):
            ang1, ang2 = cordD[0]
        else: 
            ang1, ang2 = cordD[2]
        fluxC.loc[:,'enter_x_'+ex['id']] = [x for x in dircI[ang1 + '_in']]
        fluxC.loc[:,'exit_x_'+ex['id']] = [x for x in dircI[ang2 + '_out']]
        fluxC.loc[:,'enter2_x_'+ex['id']] = [x for x in dircO[ang1 + '_out']]
        fluxC.loc[:,'exit2_x_'+ex['id']] = [x for x in dircO[ang2 + '_in']]
        exits.loc[i,'exit'] = exits.loc[i,'exit'] + np.sum([x for x in dircI[ang1 + '_in']])
        exits.loc[i,'enter'] = exits.loc[i,'enter'] + np.sum([x for x in dircI[ang2 + '_out']])
    if(dty!=0):
        if(dty < 0):
            ang1, ang2 = cordD[1]
        else: 
            ang1, ang2 = cordD[3]
        fluxC.loc[:,'enter_y_'+ex['id']] = [x for x in dircI[ang1 + '_in']]
        fluxC.loc[:,'exit_y_'+ex['id']] = [x for x in dircI[ang2 + '_out']]
        fluxC.loc[:,'enter2_y_'+ex['id']] = [x for x in dircO[ang1 + '_out']]
        fluxC.loc[:,'exit2_y_'+ex['id']] = [x for x in dircO[ang2 + '_in']]
        exits.loc[i,'exit'] = exits.loc[i,'exit'] + np.sum([x for x in dircI[ang1 + '_in']])
        exits.loc[i,'enter'] = exits.loc[i,'enter'] + np.sum([x for x in dircI[ang2 + '_out']])

print(fluxC.head(1))
fluxC.to_csv(baseDir + "out/junction_country.csv")
axes = gpd.read_file(baseDir + "gis/nissan/motorway_axes.geojson")
fluxM = pd.DataFrame(index=range(0,4*24))
axes.loc[:,'exit'] = 0
axes.loc[:,'enter'] = 0
for i,ex in axes.iterrows():
    l = ex['geometry']
    inTile = [a.contains(Point(l.xy[0][0],l.xy[1][0])) for a in dirc['geometry']]
    outTile = [a.contains(Point(l.xy[0][1],l.xy[1][1])) for a in dirc['geometry']]
    dircI = dirc.loc[inTile]
    dircO = dirc.loc[outTile]
#    ang1, ang2 = getAng(l.xy[1][1] - l.xy[1][0],l.xy[0][1] - l.xy[0][0])
    dtx = dircI.iloc[0]['col_id'] - dircO.iloc[0]['col_id']
    dty = dircI.iloc[0]['row_id'] - dircO.iloc[0]['row_id']
    ang1, ang2 = getDir(dtx,dty)
    if(dtx!=0):
        if(dtx < 0):
            ang1, ang2 = cordD[0]
        else: 
            ang1, ang2 = cordD[2]
        fluxM.loc[:,'enter_x_'+ex['id']] = [x for x in dircI[ang1 + '_in']]
        fluxM.loc[:,'exit_x_'+ex['id']] = [x for x in dircI[ang2 + '_out']]
        fluxM.loc[:,'enter2_x_'+ex['id']] = [x for x in dircO[ang1 + '_out']]
        fluxM.loc[:,'exit2_x_'+ex['id']] = [x for x in dircO[ang2 + '_in']]
        axes.loc[i,'enter'] = axes.loc[i,'enter'] + np.sum([x for x in dircI[ang1 + '_in']])
        axes.loc[i,'exit'] = axes.loc[i,'exit'] + np.sum([x for x in dircI[ang2 + '_out']])
    if(dty!=0):
        if(dty < 0):
            ang1, ang2 = cordD[1]
        else: 
            ang1, ang2 = cordD[3]
        fluxM.loc[:,'enter_y_'+ex['id']] = [x for x in dircI[ang1 + '_in']]
        fluxM.loc[:,'exit_y_'+ex['id']] = [x for x in dircI[ang2 + '_out']]
        fluxM.loc[:,'enter2_y_'+ex['id']] = [x for x in dircO[ang1 + '_out']]
        fluxM.loc[:,'exit2_y_'+ex['id']] = [x for x in dircO[ang2 + '_in']]
        axes.loc[i,'enter'] = axes.loc[i,'enter'] + np.sum([x for x in dircI[ang1 + '_in']])
        axes.loc[i,'exit'] = axes.loc[i,'exit'] + np.sum([x for x in dircI[ang2 + '_out']])
        
print(fluxM.head(1))
fluxM.to_csv(baseDir + "out/junction_motorway.csv")

with open(baseDir + "gis/nissan/junction_country.geojson","w") as fo:
    fo.write(exits.to_json())
with open(baseDir + "gis/nissan/junction_motorway.geojson","w") as fo:
    fo.write(axes.to_json())

dirc.loc[:,'sum'] = dirc.loc[:,'north_in'] + dirc.loc[:,'south_in'] + dirc.loc[:,'east_in'] + dirc.loc[:,'west_in']
dirg = dirc[['tile_id','sum','north_in','south_in','east_in','west_in','north_out','south_out','east_out','west_out']].groupby(['tile_id']).agg(sum)
dirg = dirg.reset_index()
tileL = dirc[['tile_id','col_id','row_id','geometry']].groupby(['tile_id']).head(1)
dirg = pd.merge(dirg,tileL,left_on="tile_id",right_on="tile_id",how="left")
dirg = gpd.GeoDataFrame(dirg)
with open(baseDir + "gis/nissan/junction_tile.geojson","w") as fo:
    fo.write(dirg.to_json())
    
dirl = gpd.GeoDataFrame(columns=["in","out","dir","geometry"])
for i,a in dirg.iterrows():
    l = a['geometry'].boundary
    ll = LineString([(l.xy[0][0],l.xy[1][0]),(l.xy[0][1],l.xy[1][1])])
    dirl.loc[str(a['tile_id']) + 'a'] = [a['east_in'],a['east_out'],"e",ll]
    ll = LineString([(l.xy[0][1],l.xy[1][1]),(l.xy[0][2],l.xy[1][2])])
    dirl.loc[str(a['tile_id']) + 'b'] = [a['north_in'],a['north_out'],"n",ll]
    ll = LineString([(l.xy[0][2],l.xy[1][2]),(l.xy[0][3],l.xy[1][3])])
    dirl.loc[str(a['tile_id']) + 'c'] = [a['west_in'],a['west_out'],"w",ll]
    ll = LineString([(l.xy[0][3],l.xy[1][3]),(l.xy[0][4],l.xy[1][4])])
    dirl.loc[str(a['tile_id']) + 'd'] = [a['south_in'],a['south_out'],"s",ll]
                                         
dirl = gpd.GeoDataFrame(dirl)
with open(baseDir + "gis/nissan/junction_edge.geojson","w") as fo:
    fo.write(dirl.to_json())

# fluxSM = pd.DataFrame(index=range(0,4*24))
# strL = ["exit_","enter_"]
# for s in strL:
#     colL = [bool(re.match(s,x)) for x in fluxM.columns]
#     flux1 = fluxM.loc[:,colL]
#     numL = [re.sub("[a-z].*_[xy]_","",x) for x in flux1.columns]
#     numL = [re.sub("[a-z]","",x) for x in numL]
#     for i in [x for x in set(numL)]:
#         colL1 = [bool(re.search(i,x)) for x in flux1.columns]
#         flux10 = flux1.loc[:,colL1]
#         fluxSM.loc[:'exit_' + str(i)] = flux1[

#     fluxM.loc[:,colL].head()

if False:
    from matplotlib.patches import Polygon as plPoly
    fig, ax = plt.subplots(figsize=(8, 8))
    mpl_poly = plPoly(np.array(a.exterior), facecolor="g", lw=0, alpha=0.4)
    ax.add_patch(mpl_poly)
    ax.relim()
    ax.autoscale()
    plt.show()

    a.plot(ax=ax,color='red');
    l.plot(ax=ax, color='green', alpha=0.5);
    p.plot(ax=ax, color='blue', alpha=0.5);
    plt.show()

    print(junct.head())

    import geojsonio
    geojsonio.display(junct.to_json())
    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
    world.head()
    world.plot();

