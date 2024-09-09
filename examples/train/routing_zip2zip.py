#%pylab inline
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
import os, sys, gzip, random, csv, json, re, time, csv
if True:
    sys.path.append(os.environ['LAV_DIR']+'/src/')
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

def showSalo():
    G = ox.graph_from_place('Salò, Italy')
    ec = ox.get_edge_colors_by_attr(G,attr='length')
    ox.plot_graph(G,edge_color=ec)

#get nearest node, if actual node is not in the graph
def getNearestNode(nodeid,zipN):
    coord_x = zipN.loc[zipN['node']==str(nodeid) ]['X'].item()
    coord_y = zipN.loc[zipN['node']==str(nodeid) ]['Y'].item()
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

def colorEdges(graph):
    edge_color = []
    for edge in graph.edges(data=True):
        if 'primary' in edge[2]['highway']:
            edge_color.append('orange')
        elif 'motorway' in edge[2]['highway']:
            edge_color.append('blue')
        elif 'trunk' in edge[2]['highway']:
            edge_color.append('blue')
        elif 'secondary' in edge[2]['highway']:
            edge_color.append('yellow')
        else:
            edge_color.append('grey')  
    return edge_color
    
def weightGraph(graph):
    keys = {}
    for edge in (graph.edges(data=True)):
        attrs = {}
        if edge[2]['highway']=='motorway':
            attrs['weight'] = edge[2]['length']*1
        elif edge[2]['highway']=='primary':
            attrs['weight'] = edge[2]['length']*1.5
        elif edge[2]['highway']=='secondary':
            attrs['weight'] = edge[2]['length']*1.8
        else:
            attrs['weight'] = edge[2]['length']*3

        keyy=(edge[0],edge[1],0)
        keys[keyy] = attrs
    nx.set_edge_attributes(graph, keys)
    return graph

def showShortest(graph,start_node,start_zip,dest_list,zipN):
    end_node = dest_list[0]
    start_node = 246539610
    end_node = 59768831
    edge_color = colorEdges(graph)
    route = nx.shortest_path(graph, start_node, end_node, weight='weight')
    fig,ax = ox.plot_graph_route(graph,route=route,edge_color=edge_color,node_color='none')
    
def calcRoute(graph,start,zipN):
    start_node = int(start['node'])
    try: #calculate the route from the OSM node to all nodes in the network 
        routes_dij = nx.single_source_dijkstra_path(G=graph,source=start_node,weight='weight')
    except:
        print("Start node " + str(start_node) +" not found")
        nearest_node = getNearestNode(start_node,zipN)
        print("Using nearest node instead, nodeid: " + str(nearest_node))
        routes_dij = nx.single_source_dijkstra_path(G=graph,source=nearest_node,weight='weight')

    routeL = []
    for route in list(routes_dij.items()):
        zipn = zipN[zipN['node'] == route[0]].head(1)
        if zipn.shape[0] < 1:
            continue
        if len(route[1]) < 2: #to make sure it is a route       
            continue
        length,motorway_length,trunk_length,primary_length,secondary_length,first_junction,last_junction = [0 for x in range(7)]
        first_flag = False
        for i in range(len(route[1])-1):
            edgeData = graph.get_edge_data(route[1][i],route[1][i+1])
            segDist = round(edgeData[0].get('length')/1000.)
            length = length + segDist
            edgeType = edgeData[0].get('highway')
            if edgeType == "motorway_link":
                last_junction = "%d-%d" % (route[1][i],route[1][i+1])
                if not first_flag:
                    first_junction = last_junction
                    first_flag = True
            if edgeType in ('motorway','motorway_link') :
                motorway_length = motorway_length + segDist
            elif edgeType  in ('trunk','trunk_link'):
                trunk_length = trunk_length + segDist
            elif edgeType in ('primary','primary_link'):
                primary_length = primary_length + segDist
            elif edgeType in ('secondary','secondary_link'):
                secondary_length = secondary_length + segDist
        routeL.append([int(start['PLZ']),int(zipn['PLZ'].values),length,motorway_length,primary_length,secondary_length,first_junction,last_junction])
    
    routeL = pd.DataFrame(routeL,columns=["start_zip","end_zip","length","motor_len","prim_len","sec_leng","first_junction","last_junction"])
    print('wrote node: ' + str(start_node))
    return routeL

if __name__ == '__main__':
    plog("------------------------load-map--------------------------------")
    start_time = datetime.now()
    graph = ox.load_graphml(filename="network_de.graphml",folder=baseDir+"gis/graph/")
    ## graph = graph.to_undirected()
    graph = weightGraph(graph)
    outF = "zip2zip.csv"
    zipN = pd.read_csv(baseDir + 'gis/graph/zip_node.csv')
    start_list = zipN['node'].tolist()
    start_list = list(map(int,start_list))
    plz_sel = zipN
    if True: ##limit origin zip list
        outF = "zip2zip_sel.csv"
        plz_sel = pd.read_csv(baseDir + 'gis/graph/zip_sel.csv')
        start_list = zipN['node'].tolist()
        start_list = list(map(int,start_list))
        plz_sel = pd.merge(plz_sel,zipN,left_on="PLZ",right_on="PLZ",how="left",suffixes=["","_2"])
        plz_sel.dropna(inplace=True)
    
    dest_list = start_list
    df = pd.DataFrame(columns=["start_zip","end_zip","length","motor_len","prim_len","sec_leng","first_junction","last_junction"])
    df.to_csv(baseDir + "gis/graph/" + outF,index=False)
    plog("------------------------start-routing-------------------------")
    for count,start in plz_sel.iterrows():
        end_time = datetime.now()
        delta_time = int((end_time-start_time).total_seconds())/3600.
        progress(count,plz_sel.shape[0],status='',delta_time=delta_time)
        routeL = calcRoute(graph,start,zipN)
        routeL.to_csv(baseDir+"gis/graph/"+outF,mode="a",index=False,header=False)
        
    print ('origins: ' + str(len(start_list)))
    print ('time:    ' + str(delta_time))
    print ('origins per sec: ' + str(delta_time/len(start_list)))
    print ('rel:     '+ str((len(start_list)*(len(start_list)))))
    print ('rel per sec: ' + str(delta_time/(len(start_list)*(len(start_list)))))
    print('-----------------te-se-qe-te-ve-be-te-ne------------------------')



# jobs = []
# for i in range(5):
#     p = multiprocessing.Process(target=worker)
#     jobs.append(p)
#     p.start()
# pool = Pool(6)
# segments = [(stations_with_coords, cilacs_map, id_cilac_map, start_index) for start_index in range(0, len(stations_with_coords), BUFFER)]
# cell_map_distance = pool.map(get_cilacs_to_segment, segments)

