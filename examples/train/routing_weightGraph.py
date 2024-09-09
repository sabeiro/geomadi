#%pylab inline
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
import os, sys, gzip, random, csv, json, re, time, csv
sys.path.append(os.environ['LAV_DIR']+'/src/py/')
baseDir = os.environ['LAV_DIR'] 
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import importlib
from pandas import *
from networkx.readwrite import json_graph
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
import collections
from collections import OrderedDict as odict


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

graph = ox.load_graphml(filename="network_de.graphml",folder=baseDir+"gis/graph/")
#graph = graph.to_undirected()
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
    keyy=(edge[0], edge[1], 0)
    keys[keyy] = attrs
    
nodes = graph.nodes(data=True)
edges = graph.edges()
G = nx.MultiGraph()
for i,p in pointL.iterrows():
    G.add_node(p['id'],pos=(p['x'],p['y']),x=p['x'],y=p['y']) 
for i,p in lineL.iterrows():
    G.add_edge(p['src'],p['trg'],color=p['color'],weight=p['weight'])

g = nx.MultiDiGraph()
for node, data in graph.nodes(data=True):
    g.add_node(node, json=json.dumps(data))
    
for u, v, key, data in graph.edges(data=True, keys=True):
    data1 = flatten_dict(data)
    data1.pop("geometry",None)
    g.add_edge(u, v, key=key, attr_dict=data1)


featL = []
for u, v, key, data in graph.edges(data=True, keys=True):
    uS = json.loads(g.nodes[u]['json'])
    x1, y1 = float(uS['lon']), float(uS['lat'])
    uS = json.loads(g.nodes[v]['json'])
    x2, y2 = float(uS['lon']), float(uS['lat'])
    edgeData = graph.get_edge_data(u,v)[0]
    edgeData['id1'] = u
    edgeData['id2'] = v
    try:
        edgeData.pop('geometry')
    except:
        i = 0
    location = odict([("type","LineString"),("coordinates",[[x1,y1],[x2,y2]])])
    feature = odict([("type","Feature"),("properties",edgeData),("geometry",location)])
    featL.append(feature)
    
features = {"type": "FeatureCollection", "features": featL}
with open(baseDir + "gis/graph/route_lines.geojson", "w") as f:
    json.dump(features, f)

featL = []
for node, data in graph.nodes(data=True):
    featL.append(data)

featL = pd.DataFrame(featL)
featL.loc[:,"lat"] = featL['lat'].apply(lambda x:float(x))
featL.loc[:,"lon"] = featL['lon'].apply(lambda x:float(x))
featL.to_csv(baseDir + "gis/graph/route_nodes.csv")

lineG = gpd.read_file(baseDir + "gis/graph/route_line.shp")
nodeG1 = pd.DataFrame({"id":lineG['id1']},index=lineG.index)
nodeG2 = pd.DataFrame({"id":lineG['id2']},index=lineG.index)
xy = lineG.geometry.apply(lambda x: x.xy)
nodeG1.loc[:,"x"] = [x[0][0] for x in xy]
nodeG1.loc[:,"y"] = [x[1][0] for x in xy]
nodeG2.loc[:,"x"] = [x[0][1] for x in xy]
nodeG2.loc[:,"y"] = [x[1][1] for x in xy]
nodeG = pd.concat([nodeG1,nodeG2],axis=0)
nodeG = nodeG.groupby("id").head(1)
graph = nx.MultiGraph()
for i,p in nodeG.iterrows():
    graph.add_node(p['id'],pos=(p['x'],p['y']),x=p['x'],y=p['y']) 
for i,p in lineG.iterrows():
    graph.add_edge(p['id1'],p['id2'],weight=float(p['weight']),highway=p['highway'],length=float(p['length']))

ox.save_graphml(graph,filename="route_line.graphml",folder=baseDir+"gis/graph/")



nx.write_shp(g,baseDir + "gis/graph/network_de")
pg = nx.to_pandas_edgelist(g)
pg = gpd.GeoDataFrame(pg)
pg.to_file(baseDir + "gis/graph/network_de.geojson")


nx.set_edge_attributes(graph, keys)

nx.write_shp(graph,baseDir + "gis/graph/")
nx.write_gml(graph,baseDir + "gis/graph/network_de")
nx.write_gexf(g,baseDir + "gis/graph/network_de.gexf")


ox.save_graphml(graph,filename="routes_germany.graphml",folder=baseDir+"gis/graph/")
ox.save_gdf_shapefile(graph,filename="routes_germany.gdf",folder=baseDir+"gis/graph/")
pg = nx.to_pandas_edgelist(graph)
pg = gpd.GeoDataFrame(pg)
pg.to_file(baseDir + "gis/graph/network_de.geojson")
pg.to_file(baseDir + "gis/graph/network_de.shp",driver='ESRI Shapefile')
import fiona
print(fiona.supported_drivers)

zipN = pd.read_csv(baseDir + 'gis/graph/zip_node.csv')

nodeG = list(graph.nodes(data=True))
nodeD = pd.DataFrame(nodeG)
nodeL = []
for g in nodeG:
    kDict = g[1]
    kDict['id'] = g[0]
    nodeL.append(kDict)

nodeL = pd.DataFrame(nodeL)
nodeL = nodeL[nodeL['highway'] == "motorway_junction"]
nodeL.to_csv(baseDir + "gis/graph/nodeList.csv",index=False)


pos = nx.spring_layout(graph)
for e in G.edges():
    i = 1
for d in graph.nodes():
    
    print(d)
    g = graph.node(d)
    i = 1
G = nx.path_graph(4)
pos = nx.spring_layout(G)
for node,(x,y) in pos.items():
    G.node[node]['x'] = float(x)
    G.node[node]['y'] = float(y)

    
zipN = pd.read_csv(baseDir + "gis/geo/zip5_centroid.csv")
for i, row in zipN.iterrows():
    coords = (row['X'], row['Y'])
    point = Point(coords)
    nearest_node = ox.get_nearest_node(graph,coords,method='euclidean')
    zipN.loc[i,'node'] = nearest_node

zipN.to_csv(baseDir + "raw/nissan/zip_node.csv",index=False)

if False:
    graphD = json_graph.node_link_data(graph)
    json.dump(graphD,open(baseDir+"gis/graph/routes_germany.json","w") )
    

    
if False:
    ox.plot_graph(graph)

# Set configuration constants that will be used in this script

BUFFER_SIZE = 50000


def create_edges(nodes, segments, way):
    refs = way['refs']
    for i in range(0, len(refs) - 1):
        src, trg = int(refs[i]), int(refs[i + 1])
        edge = create_edge(way, src, trg, nodes)
        if edge == -1:
            get_logger().info('Same coordinates src: {} and trg: {}, {}', src, trg,
                              nodes[refs[i]]['loc']['coordinates'])
            continue
        segments[(src, trg)] = edge
        if way['oneway'] == "no":
            reverse_edge = create_edge(way, trg, src, nodes)
            segments[(trg, src)] = reverse_edge

def create_edge(way, src, trg, nodes):
    if nodes[src]['loc']['coordinates'] == nodes[trg]['loc']['coordinates']:
        return -1

    edge = {}
    edge['src'] = src
    edge['trg'] = trg
    edge['maxspeed'] = way['maxspeed']
    edge['highway'] = way['highway']
    edge['oneway'] = way['oneway']
    multi_line_dict = {}
    multi_line_dict['type'] = 'LineString'
    multi_line_dict['coordinates'] = []
    multi_line_dict['coordinates'].append(nodes[src]['loc']['coordinates'])
    multi_line_dict['coordinates'].append(nodes[trg]['loc']['coordinates'])
    edge['loc'] = multi_line_dict
    return edge

def save_segments(infra_db, segments_col, host_name, port, segments):
    segments_client = utils.get_mongo_collection_with_host(infra_db, segments_col, host_name, port)
    segments_client.drop()
    save_dict_to_mongo(segments, segments_client)
    segments_client.create_index([('loc', pymongo.GEOSPHERE)])


def save_nodes(infra_db, nodes_col, host_name, port, nodes):
    get_logger().info(infra_db)
    get_logger().info(nodes_col)
    get_logger().info(host_name)
    get_logger().info(str(port))
    get_logger().info("Number of Nodes: {}", str(len(nodes)))
    nodes_client = utils.get_mongo_collection_with_host(infra_db, nodes_col, host_name, port)
    nodes_client.drop()
    get_logger().info("D1: {}", str(datetime.now()))
    save_dict_to_mongo(nodes, nodes_client)
    get_logger().debug('building index')
    nodes_client.create_index([('node_id', pymongo.ASCENDING)])
    nodes_client.create_index([('loc', pymongo.GEOSPHERE)])


def save_dict_to_mongo(elem_dict, client):
    """
    Receives a dictionary of elements and a client connection and stores the elements in the database using bulk write operation.
    :param elem_dict: list of elements to be stored in mongo
    :param client: client (database, collection) where the elements should be stored
    """
    elem_buffer = []
    for elem in elem_dict:
        elem_buffer.append(elem_dict[elem])
        if len(elem_buffer) > BUFFER_SIZE:
            client.insert_many(elem_buffer)
            elem_buffer = []

    flush_to_db(elem_buffer, client)
