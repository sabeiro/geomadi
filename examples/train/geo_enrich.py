"""
geo_enrich:
enrich poi information with additional geographical information
"""

import os, sys, gzip, random, csv, json, datetime, re
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import geopandas as gpd
import geomadi.geo_octree as octree
import shapely as sh
import shapely.speedups
import osmnx as ox
import networkx as nx
shapely.speedups.enable()

def addRegion(poi,regionF,field="GEN"):
    """assign a region to the poi"""
    region = gpd.GeoDataFrame.from_file(regionF)
    region.index = region[field]
    region = region['geometry']
    pL = poi[['x','y']].apply(lambda x: sh.geometry.Point(x[0],x[1]),axis=1)
    pnts = gpd.GeoDataFrame(geometry=pL)
    pnts = pnts.assign(**{key: pnts.within(geom) for key, geom in region.items()})
    for i in pnts.columns[1:]:
        #if (i%10) == 0: print("process %.2f\r" % (i/pnts.shape[0]),end="\r",flush=True)
        poi.loc[pnts[i],"region"] = i
    return poi['region']

def addZone(poi,max_d):
    """assign a zone to the poi, clustering"""
    Z = linkage(poi[['x','y']], 'ward')
    zoneL = fcluster(Z,max_d,criterion='distance')
    #newZone = np.isnan(poi['id_zone'])
    return zoneL

def interp2D(dens,x1,y1,z_col="Einwohner"):
    """ interpolate 2D and returns centroid value """
    x = [x.centroid.x for x in dens.geometry]
    y = [x.centroid.y for x in dens.geometry]
    z = [x for x in dens[z_col]]
    interp = sp.interpolate.Rbf(x,y,z,function='multiquadric',epsilon=0.03)
    if False:
        print(interp(x1,y1))
        print(np.mean(z))
        xi, yi = np.mgrid[min(x):max(x):50j,min(y):max(y):50j]
        zi = interp(xi,yi)
        fig, ax = plt.subplots()
        dens.plot(column=z_col,ax=ax,alpha=.5)
        ax.plot(x,y,'ko',alpha=.3)
        #        plt.colorbar()
        ax.imshow(zi,extent=[min(x),max(x),min(y),max(y)],cmap='gist_earth',alpha=.6,origin="lower")
        ax.scatter(p.geometry.centroid.x,p.geometry.centroid.y,color="red")
        plt.show()
    return interp(x1,y1)

def spaceDeg(poi,isPlot=False):
    """calculate spatial degeneracy"""
    param = [1.]
    def ser_fun(x,t,param):
        return x[0] + x[1] * t + x[2] * t * t
    def ser_fun_min(x,t,y,param):
        return ser_fun(x,t,param) - y
    x0 = [1.,1.,1.]
    if isPlot:
        degV = []
        degL = []
    degT = []
    for i,g in poi.iterrows():
        r = np.sqrt((poi['x'] - g['x'])**2 + (poi['y'] - g['y'])**2)
        r = r[r > 0.]
        h, t = np.histogram(r,bins=40,normed=False,range=(0.0,0.5))
        t = t[1:]
        res = sp.least_squares(ser_fun_min,x0,args=(t,h,param))
        y = ser_fun(res.x,t,param)
        if isPlot:
            degV.append(y)
            degL.append(h)
        degT.append(res.x[0])

    if isPlot:
        fig, ax = plt.subplots(1,2)
        for i in range(12):
            ax[0].plot(t,degL[i],alpha=.3)
            ax[1].plot(t,degV[i],alpha=.3)
        ax[0].set_title("histogram")
        ax[1].set_title("parabolic")
        ax[0].set_xlabel("grad distance")
        ax[1].set_xlabel("grad distance")
        ax[0].set_ylabel("density")        
        plt.show()
    return degT

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    km = 6371* c
    return km*.001

def localNetwork(xc,yc,dx,dy,collE,geo_idx="loc"):
    """ download local network from mongo collections"""
    BBox = [xc-dx,xc+dx,yc-dy,yc+dy]
    neiE = collE.find({geo_idx:{'$geoIntersects':{'$geometry':{
        "type":"Polygon"
        ,"coordinates":[
            [ [BBox[0],BBox[3]],[BBox[1],BBox[3]],[BBox[1],BBox[2]],[BBox[0],BBox[2]],[BBox[0],BBox[3]] ]
        ]}}}})
    pointL, listN, listE = [],[],[]
    G = nx.MultiDiGraph()
    G.graph['crs'] = {'init': 'epsg:4326'}
    G.graph['name'] = "local network"
    for neii in neiE:
        line = sh.geometry.LineString([(neii[geo_idx]['coordinates'][0][0],neii[geo_idx]['coordinates'][0][1]),(neii[geo_idx]['coordinates'][1][0],neii[geo_idx]['coordinates'][1][1])])
        dist = haversine(line.xy[0][0],line.xy[0][1],line.xy[1][0],line.xy[1][1])
        G.add_edge(neii['src'],neii['trg'],key=0,src=neii['src'],trg=neii['trg'],speed=neii['maxspeed'],highway=neii['highway'],geometry=line,length=dist)
        G.add_node(int(neii['src']),pos=(neii[geo_idx]['coordinates'][0][0],neii[geo_idx]['coordinates'][0][1]),x=neii[geo_idx]['coordinates'][0][0],y=neii[geo_idx]['coordinates'][0][1],key=0,osmid=int(neii['src']))
        G.add_node(int(neii['trg']),pos=(neii[geo_idx]['coordinates'][1][0],neii[geo_idx]['coordinates'][1][1]),x=neii[geo_idx]['coordinates'][1][0],y=neii[geo_idx]['coordinates'][1][1],key=0,osmid=int(neii['trg']))
    if False:
        ox.plot_graph(G)
    return G

def localPolygon(xc,yc,dx,dy,collE,geo_idx="loc"):
    """ download all the crossing polygons intersecting the bounding box"""
    BBox = [xc-dx,xc+dx,yc-dy,yc+dy]
    neiE = collE.find({geo_idx:{'$geoIntersects':{'$geometry':{
        "type":"Polygon"
        ,"coordinates":[
            [ [BBox[0],BBox[3]],[BBox[1],BBox[3]],[BBox[1],BBox[2]],[BBox[0],BBox[2]],[BBox[0],BBox[3]] ]
        ]}}}})
    featL = []
    for neii in neiE:
        feat = {}
        feat['type'] = "Feature"
        feat['geometry'] = neii['geom']
        del neii['geom']
        feat['properties'] = neii
        featL.append(feat)
    geoD = gpd.GeoDataFrame.from_features(featL)
    geoD = geoD.loc[[x.is_valid for x in geoD['geometry'].values]]
    del geoD['_id']
    if False:
        geoD.plot()
        plt.show()
    return geoD

def cellPolygon(celL,collE,geo_idx="geom"):
    """ download all the geo features from a list of id"""
    featL = []
    for cilac in celL:
        neiE = collE.find({"cell_ci":int(cilac.split("-")[0]),"cell_lac":int(cilac.split("-")[1])})
        neii = neiE[0]
        feat = {}
        feat['type'] = "Feature"
        feat['geometry'] = neii[geo_idx]
        del neii['geom']
        feat['properties'] = neii
        featL.append(feat)
    geoD = gpd.GeoDataFrame.from_features(featL)
    if False:
        geoD.plot()
        plt.show()
    return geoD

def getSampleGraph():
    """download a sample graph from openstreetmap"""
    G = ox.graph_from_place('Salò, BS, Italy',network_type='walk')
    return G

def isochrone(G,isPlot=False):
    """calculate the isocrone from a graph"""
    import osmnx as ox, networkx as nx, geopandas as gpd, matplotlib.pyplot as plt
    from shapely.geometry import Point, LineString, Polygon
    from descartes import PolygonPatch
    ox.config(log_console=True, use_cache=True)
    trip_times = [5, 10, 15, 20, 25] #in minutes
    travel_speed = 4.5 #walking speed in km/hour
    gdf_nodes = ox.graph_to_gdfs(G, edges=False)
    x, y = gdf_nodes['geometry'].unary_union.centroid.xy
    center_node = ox.get_nearest_node(G, (y[0], x[0]))
    G = ox.project_graph(G)
    meters_per_minute = travel_speed * 1000 / 60 #km per hour to m per minute
    for u, v, k, data in G.edges(data=True, keys=True):
        data['time'] = data['length'] / meters_per_minute
        data['time'] = data['length'] / data['speed'] #* 0.06 # m / km/h
    iso_colors = ox.get_colors(n=len(trip_times), cmap='Reds', start=0.3, return_hex=True)
    node_colors = {}
    isochrone_polys = []
    for trip_time, color in zip(sorted(trip_times, reverse=True), iso_colors):
        subgraph = nx.ego_graph(G, center_node, radius=trip_time, distance='time')
        for node in subgraph.nodes():
            node_colors[node] = color
        node_points = [Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]
        bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull
        isochrone_polys.append(bounding_poly)
    nc = [node_colors[node] if node in node_colors else 'none' for node in G.nodes()]
    ns = [20 if node in node_colors else 0 for node in G.nodes()]
    if isPlot:
        fig, ax1 = ox.plot_graph(G,fig_height=8,node_color=nc,node_size=ns,node_alpha=0.8,node_zorder=2,show=False,close=False)
        fig, ax = ox.plot_graph(G,fig_height=8,show=False,close=False,edge_color='k',edge_alpha=0.2,node_color='none')
        for polygon, fc in zip(isochrone_polys, iso_colors):
            patch = PolygonPatch(polygon, fc=fc, ec='none', alpha=0.6, zorder=-1)
        ax.add_patch(patch)
        plt.show()
    return isochrone_polys

