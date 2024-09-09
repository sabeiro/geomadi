#%pylab inline
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
import os, sys, gzip, random, csv, json, datetime,re, time
sys.path.append(os.environ['LAV_DIR']+'/src/py/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import importlib
from io import StringIO
from pandas import *


def plog(text):
    print(text)


if False:
    G = ox.graph_from_place('Salò, Italy')
    ec = ox.get_edge_colors_by_attr(G, attr='length')
    ox.plot_graph(G, edge_color=ec)

if False:
    nodes_1, edges_1 = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    edges_1.head()
    list(edges_1.columns)
    edge_color = []
    for edge in graph.edges(data=True):
        if 'primary' in edge[2]['highway']:
            edge_color.append('orange')
        elif 'motorway' in edge[2]['highway']:
            edge_color.append('blue')
        elif 'trunk' in edge[2]['highway']:
            edge_color.append('blue')
        elif 'secondary' in edge[2]['highway']:
            edge_color.append('grey')
        else:
            edge_color.append('grey')  
    fig, ax = ox.plot_graph(graph,  edge_color=edge_color, node_color='none')


    if False: ##single path
    origin_node = list(graph.nodes())[0]
    destination_node = list(graph.nodes())[-10000]
    route = nx.shortest_path(graph, origin_node, destination_node, weight='weight')
    length = 0
    motorway_length = 0
    trunk_length = 0
    primary_length = 0
    secondary_length = 0
    #iterate over every single edge along the path
    for i in range(len(route)-1):
        edgeData= graph.get_edge_data(route[i],route[i+1])
        length = length + edgeData[0].get('length')
        if edgeData[0].get('highway') in ('motorway','motorway_link') :
            motorway_length = motorway_length + edgeData[0].get('length')
        elif edgeData[0].get('highway') in ('trunk', 'trunk_link'):
            trunk_length = trunk_length + edgeData[0].get('length')
        elif edgeData[0].get('highway') in ('primary', 'primary_link'):
            primary_length = primary_length + edgeData[0].get('length')
        elif edgeData[0].get('highway') in ('secondary', 'secondary_link'):
            secondary_length = secondary_length + edgeData[0].get('length') 

    #print (length_raw)
    print ("Highway part length: ")
    print ("motorway: " + str(int(round(motorway_length/1000))) + " km")
    print ("trunk:    " + str(int(round(trunk_length/1000))) + " km")
    print ("primary:  " + str(int(round(primary_length/1000))) + " km")
    print ("secondary:  " + str(int(round(secondary_length/1000))) + " km")
    print (" ")
    print ("Route length: " + str(int(round(length/1000))) + " km")
    fig, ax = ox.plot_graph_route(graph, route=route,  edge_color=edge_color, node_color='none')


with open(baseDir + 'gis/graph/route_length_germany-sec_real_linkdata.csv','a', newline='') as newFile:
    newFileWriter = csv.writer(newFile)
    newFileWriter.writerow(['origin_id','destination_id','distance','motorway_distance','trunk_distance','primary_distance','secondary_distance'])
    
for i in range(100):
    origin_node = list(graph1.nodes())[i]
    routes_dij = nx.single_source_dijkstra_path(graph1_proj, origin_node, weight='weight')
    for route in list(routes_dij.items()):
        #to make sure it is a route       
        if len(route[1]) < 1 :
            continue
        length = 0
        motorway_length = 0
        trunk_length = 0
        primary_length = 0
        secondary_length = 0
        #iterate over every single edge along the path
        for i in range(len(route[1])-1):
            edgeData= graph1_proj.get_edge_data(route[1][i],route[1][i+1])
            length = length + edgeData[0].get('length')
            if edgeData[0].get('highway') in ('motorway','motorway_link') :
                motorway_length = motorway_length + edgeData[0].get('length')
            elif edgeData[0].get('highway') in ('trunk', 'trunk_link'):
                trunk_length = trunk_length + edgeData[0].get('length')
            elif edgeData[0].get('highway') in ('primary', 'primary_link'):
                primary_length = primary_length + edgeData[0].get('length')
            elif edgeData[0].get('highway') in ('secondary', 'secondary_link'):
                secondary_length = secondary_length + edgeData[0].get('length') 
        newFileWriter.writerow([origin_node, route[0], int(round(length/1000)), int(round(motorway_length/1000)), int(round(trunk_length/1000)), int(round(primary_length/1000)), int(round(secondary_length/1000))])
            #print ('route_length is '+ str( length)+ ' from ' +str(origin_node)+ ' to ' + str(route[0])+  ' and motorway: ' +str(motorway_length) )
                
if False:
    print ("nodes: " + str(len(graph.nodes(data=True))))
    print ("edges: " + str(len(graph.edges(data=True))))
    fig, ax = ox.plot_graph(graph)

