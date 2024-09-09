#%pylab inline
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
import os, sys, gzip, random, csv, json, re, time, csv
if False:
    sys.path.append(os.environ['LAV_DIR']+'/src/py/')
    baseDir = os.environ['LAV_DIR']
baseDir = ""
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import importlib
from io import StringIO
from pandas import *
import multiprocessing
from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_double
from datetime import datetime, timedelta
from multiprocessing.pool import Pool

def plog(labS):
    print(labS)

def worker():
    """worker function"""
    print('Worker')
    return

#get nearest node, if actual node is not in the graph
def getNearestNode(nodeid,plz_df):
    coord_x = plz_df.loc[plz_df['nearest_graph_node']==str(nodeid) ]['X'].item()
    coord_y = plz_df.loc[plz_df['nearest_graph_node']==str(nodeid) ]['Y'].item()
    coords = (coord_x, coord_y)
    nearest_node = ox.get_nearest_node(graph, coords, method='euclidean')
    return nearest_node

def progress(count, total, status='',delta_time=0): #progressbar
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s %.2fh\r' % (bar,percents,'%',status,delta_time))
    sys.stdout.flush()

def calcRoute(graph,start_node,start_zip,dest_list,plz_df):
    try: #calculate the route from the OSM node to all nodes in the network 
        routes_dij = nx.single_source_dijkstra_path(G=graph,source=start_node,weight='weight')
        ##route = nx.shortest_path(graph, origin_node, destination_node, weight='weight')
    except:
        print("Start node " + str(start_node) +" not found")
        nearest_node = getNearestNode(start_node,plz_df)
        print("Using nearest node instead, nodeid: " + str(nearest_node))
        routes_dij = nx.single_source_dijkstra_path(G=graph, source=nearest_node,weight='weight')

    routeL = []
    for route in list(routes_dij.items()):
        if route[0] not in dest_list:
            continue
        if len(route[1]) < 2: #to make sure it is a route       
            continue
        length = 0
        motorway_length = 0
        trunk_length = 0
        primary_length = 0
        secondary_length = 0
        first_junction = 0
        first_flag = False
        last_junction = 0
        for i in range(len(route[1])-1):
            edgeData = graph.get_edge_data(route[1][i],route[1][i+1])
            length = length + round(edgeData[0].get('length')/1000.)
            edgeType = edgeData[0].get('highway')
            segDist = round(edgeData[0].get('length')/1000.)
            if edgeType == "motorway_link":
                last_junction = route[1][i]
                if not first_flag:
                    first_junction = route[1][i]
                    first_flag = True
            if edgeType in ('motorway','motorway_link') :
                motorway_length = motorway_length + segDist
            elif edgeType  in ('trunk', 'trunk_link'):
                trunk_length = trunk_length + segDist
            elif edgeType in ('primary', 'primary_link'):
                primary_length = primary_length + segDist
            elif edgeType in ('secondary', 'secondary_link'):
                secondary_length = secondary_length + segDist
        routeL.append([route[0],length,motorway_length,primary_length,secondary_length,first_junction,last_junction])

    routeL = pd.DataFrame(routeL,columns=["end","length","motorway_length","primary_length","secondary_length","first_junction","last_junction"])
    routeL.loc[:,"start_zip"] = start_zip
    routeL.loc[:,"end_zip"] = pd.merge(routeL,plz_df,left_on="end",right_on='nearest_graph_node',how="left")['PLZ']
    del routeL['end']
    print('wrote node: ' + str(start_node))
    return routeL

if __name__ == '__main__':
    plog("------------------------load-map--------------------------------")
    start_time = datetime.now()
    ##graph1 = ox.load_graphml(filename="routes_germany.graphml",folder=baseDir+"gis/graph/")
    graph = ox.load_graphml(filename="allGermany_allstreetsUntilSec_proj.graphml",folder=baseDir+"gis/graph/")
    plog("-----------------------undirect-graph---------------------------")
    graph = graph.to_undirected()
    for edge in graph.edges():
        edgeD = graph.get_edge_data(edge[0],edge[1])
        for i,x in edgeD.items():
            weight = 1000
            if x.get('highway') == 'motorway' :
                weight = 1
            graph[edge[0]][edge[1]][i]["weight"] = weight
    plz_df = pd.read_csv(baseDir + 'gis/graph/zip_node.csv')
    start_list = plz_df['nearest_graph_node'].tolist()
    start_list = list(map(int,start_list))
    dest_list = start_list
    df = pd.DataFrame(columns=['distance','motorway_distance','primary_distance','secondary_distance','first_junction','last_junction','start_zip','end_zip'])
    df.to_csv(baseDir + "gis/graph/zip2zip.csv",index=False)
    plog("------------------------start-routing-------------------------")
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker)
        jobs.append(p)
        p.start()
     # pool = Pool(6)
     # segments = [(stations_with_coords, cilacs_map, id_cilac_map, start_index) for start_index in range(0, len(stations_with_coords), BUFFER)]
     # cell_map_distance = pool.map(get_cilacs_to_segment, segments)

    for count,start in plz_df.iterrows():
        end_time = datetime.now()
        delta_time = int((end_time-start_time).total_seconds())/3600.
        progress(count,len(start_list),status='',delta_time=delta_time)
        start_node = int(start['nearest_graph_node'])
        start_zip = int(start['PLZ'])
        routeL = calcRoute(graph,start_node,start_zip,dest_list,plz_df)
        routeL.to_csv(baseDir + "gis/graph/zip2zip.csv",mode="a",index=False,header=False)
        
    print ('origins: ' + str(len(start_list)))
    print ('time:    ' + str(delta_time))
    print ('origins per sec: ' + str(delta_time/len(start_list)))
    print ('rel:     '+ str((len(start_list)*(len(start_list)))))
    print ('rel per sec: ' + str(delta_time/(len(start_list)*(len(start_list)))))
    print('-----------------te-se-qe-te-ve-be-te-ne------------------------')



