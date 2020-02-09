import gzip
import json
import sys

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests
import scipy as sp

import geomadi.geo_enrich as g_e
import geomadi.geo_octree as g_o

gO = g_o.h3tree()


def simplifyGraph(G, connection="strong"):
    """simplify a graph"""
    # G = ox.load_graphml(filename=fName)
    print("G nodes: " + str(len(G.nodes())))
    print("G edges: " + str(len(G.edges())))
    G_simp = ox.simplify_graph(G)
    print('simp_G:')
    print("G simp nodes: " + str(len(G_simp.nodes())))
    print("G simp edges: " + str(len(G_simp.edges())))
    if connection == "strong":
        G_connect = max(nx.weakly_connected_component_subgraphs(G_simp), key=len)
    elif connection == "weak":
        G_connect = max(nx.strongly_connected_component_subgraphs(G_simp), key=len)
    elif connection == "undirect":
        G_connect = max(nx.connected_component_subgraphs(G_simp.to_undirected()), key=len)
    else:
        G_connect = G_simp
    print('connected_G:')
    print("G connected nodes: " + str(len(G_connect.nodes())))
    print("G connected edges: " + str(len(G_connect.edges())))
    G_proj = ox.project_graph(G_connect, to_crs={'init': 'epsg:4326'})
    print('proj done')
    return G_proj


def removeType(G, move_type="driver"):
    """remove fclass types"""
    print("set move type %s" % (move_type))
    l = list(G.edges(data=True))
    tL, n = np.unique([x[2]['highway'] for x in l], return_counts=True)
    tL = ['cycleway', 'footway', 'living_street', 'motorway', 'motorway_link', 'path', 'pedestrian', 'primary',
          'residential', 'secondary', 'secondary_link', 'service', 'steps', 'tertiary', 'track', 'track_grade1',
          'track_grade3', 'trunk', 'trunk_link', 'unclassified']
    if move_type == "pedestrian":
        tL = ['cycleway', 'footway', 'living_street', 'path', 'pedestrian', 'residential', 'primary', 'secondary',
              'secondary_link', 'tertiary', 'trunk', 'trunk_link']
    if move_type == "driver":
        tL = ['living_street', 'motorway', 'motorway_link', 'primary', 'residential', 'secondary', 'secondary_link',
              'tertiary', 'trunk', 'trunk_link']
    G1 = G.copy()
    G1.graph['crs'] = {'init': 'epsg:4326'}
    G1.graph['name'] = move_type + " network"
    for i, j, x in G.edges(data=True):
        if x['highway'] in tL: continue
        G1.remove_edge(i, j)
    idL = []
    for i, j in G1.edges():
        idL.append(i)
        idL.append(j)
    idL = np.unique(idL)
    for i in G.nodes():
        if not i in idL:
            G1.remove_node(i)
    return G1


def composeGraph(graphL):
    """compose a list of graphs into a single one"""
    composed_G = nx.compose_all(graphL)
    return composed_G


def composeFiles(fileL):
    """compose a list of graphs from files"""
    graphL = []
    for f in fileL:
        graph1 = ox.load_graphml(filename=f)
    composed_G = nx.compose_all(graphL)
    simp_G = ox.simplify_graph(composed_G)
    return composed_G


def segment2graph(geo, isPlot=False):
    """create a graph from a collection of lines"""
    print("building the graph")
    gO = g_o.h3tree()
    G = nx.MultiDiGraph()
    # G = nx.Graph()
    G.graph['crs'] = {'init': 'epsg:4326'}
    G.graph['name'] = "local network"
    for i, g in geo.iterrows():
        seg = list(g['geometry'].coords)
        for start, end, j in zip(seg[:-1], seg[1:], range(len(seg))):
            # line = sh.geometry.LineString([start,end])
            dist = g_e.haversine(start[0], start[1], end[0], end[1])
            # j1, j2 = g['osm_id']+"_"+str(j), g['osm_id']+"_"+str(j+1)
            j1, j2 = gO.encode(start[0], start[1], precision=13), gO.encode(end[0], end[1], precision=13)
            G.add_edge(j1, j2, key=0, src=j1, trg=j2, speed=g['maxspeed'], highway=g['fclass'], name=g['name'],
                       length=dist)  # ,geometry=line,osm_id=g['osm_id'])
            G.add_node(j1, pos=start, x=start[0], y=start[1], key=0)  # ,osmid=j1)
            G.add_node(j2, pos=end, x=end[0], y=end[1], key=0)  # ,osmid=j2)
    if isPlot:
        ox.plot_graph(G)
    return G


def line2graph(geo, isPlot=False):
    """create a graph from a collection of lines"""
    gO = g_o.h3tree()
    G = nx.MultiDiGraph()
    # G = nx.Graph()
    G.graph['crs'] = {'init': 'epsg:4326'}
    G.graph['name'] = "local network"
    for i, g in geo.iterrows():
        seg = list(g['geometry'].coords)
        start, end = seg[0], seg[-1]
        dist = g_e.haversine(start[0], start[1], end[0], end[1])
        j1, j2 = gO.encode(start[0], start[1], precision=13), gO.encode(end[0], end[1], precision=13)
        G.add_edge(j1, j2, key=0, speed=g['maxspeed'], highway=g['fclass'], name=g['name'], length=dist)
        G.add_node(j1, pos=start, x=start[0], y=start[1], key=0)
        G.add_node(j2, pos=end, x=end[0], y=end[1], key=0)
    if isPlot:
        ox.plot_graph(G)
    print("built the graph size %d" % (G.size()))
    return G


def downloadGraph(city="Salò, Italy"):
    """download graph from city"""
    G = ox.graph_from_place(city)
    G = ox.project_graph(G)
    ox.plot_graph(G)
    return G


def lineLenght(l):
    """length of a line in kilometers"""
    l1 = [[x, y] for x, y in zip(l.xy[0], l.xy[1])]
    length = sum([g_e.haversine(x[0], x[1], y[0], y[1]) for x, y in zip(l1[:-1], l1[1:])])
    return length


def getNearestNode(G, coord_x, coord_y):
    """get nearest node, if actual node is not in the graph"""
    coords = (coord_x, coord_y)
    node = ox.get_nearest_node(G, coords, method='euclidean')
    return node


def getNeighbors(G, posL):
    """get nearest node, if actual node is not in the graph"""
    pos = [(x['x'], x['y']) for i, x in G.nodes(data=True)]
    tree = sp.spatial.KDTree(pos)
    nearest_dist, nearest_ind = tree.query(posL)
    posI = {}
    for i, j in enumerate(G.nodes()): posI[i] = j
    posI = [posI[x] for x in nearest_ind]
    neiL = [G.node[x] for x in posI]
    return posI


def weightGraph(graph):
    keys = {}
    for edge in (graph.edges(data=True)):
        attrs = {}
        if edge[2]['highway'] == 'motorway':
            attrs['weight'] = edge[2]['length'] * 1
        elif edge[2]['highway'] == 'primary':
            attrs['weight'] = edge[2]['length'] * 1.5
        elif edge[2]['highway'] == 'secondary':
            attrs['weight'] = edge[2]['length'] * 1.8
        else:
            attrs['weight'] = edge[2]['length'] * 3
        keyy = (edge[0], edge[1], 0)
        keys[keyy] = attrs
    nx.set_edge_attributes(graph, keys)
    return graph


def weightSpeed(G):
    """weight the graph according to street speed"""
    wL = {'motorway': 3, 'motorway_link': 3, 'primary': 2, 'residential': 0.5, 'secondary': 2, 'secondary_link': 2,
          'tertiary': 1.5, 'trunk': 2, 'trunk_link': 2}
    keys = {}
    for u, v, k, edge in (G.edges(keys=True, data=True)):
        s = edge['speed']
        l = edge['length']
        t = edge['highway']
        if isinstance(s, list): s = s[0]
        if isinstance(l, list): l = l[0]
        if isinstance(t, list): t = t[0]
        t = wL[t]
        s = max(s, 30)  # kmh
        l = max(l, 0.1)  # km
        w = s / l * 0.01 * t
        G[u][v][k].update(weight=w)
    return G


def calcPair(G, route):
    """pair distance between points"""
    if len(route) < 2: return False, [0, 0, 0, 0]
    length, weight, duration = 0, 0, 0
    for i, j in zip(route[:-1], route[1:]):
        edgeData = G.get_edge_data(i, j)[0]
        length = length + edgeData['length']
        weight = weight + edgeData['weight']
        if isinstance(edgeData['speed'], list):
            time = edgeData['length'] / max(min(edgeData['speed']), 1)  # *(1.-edgeData['weight']))
        else:
            time = edgeData['length'] / max(edgeData['speed'], 1)  # *(1.-edgeData['weight']))
        duration = duration + time
    weight = weight / (len(route) - 1)
    # if length > 3: return False, [0,0,0,0] #long distances are useless
    return True, [route[0], route[-1], length, weight, duration]


def calcRoute(G, start, destL, colL):
    node = start['node']
    try:  # calculate the route from the OSM node to all nodes in the network
        route_dij = nx.single_source_dijkstra_path(G=G, source=node, weight='weight')
    except:
        print("Start node " + str(node) + " not found")
        nearest_node = getNearestNode(node, start['y'], start['x'])
        print("Using nearest node instead, nodeid: " + str(nearest_node))
        route_dij = nx.single_source_dijkstra_path(G=G, source=nearest_node, weight='weight')

    routeL = [x for i, x in route_dij.items() if i in destL]
    pairL = []
    for route in routeL:
        status, link = calcPair(G, route)
        if status: pairL.append(link)

    pairL = pd.DataFrame(pairL, columns=colL)
    return pairL, len(routeL) / len(destL)


def routeOsrm(start, end):
    """routing-with-local-osrm"""
    # 'docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/berlin-latest.osrm')
    x1, y1 = gO.decode(start)
    x2, y2 = gO.decode(end)
    fromP = "%f,%f" % (x1, y1)
    toP = "%f,%f" % (x2, y2)
    resq = requests.get("http://127.0.0.1:5000/route/v1/driving/" + fromP + ";" + toP + "?steps=true")
    route = resq.json()
    route = route['routes'][0]
    return [start, end, route['distance'], route['weight'], route['duration']]


def progress(count, total, status='', delta_time=0, completion=1.):
    """progressbar"""
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%% %s %.2fh comp %.2f\r' % (bar, percents, status, delta_time, completion))
    sys.stdout.flush()


def save(G, fName):
    """suggest different file formats"""
    nx.write_gml(G, fName + ".gml")
    # G = nx.read_gml(fName+".gml")
    if False:
        ox.save_graphml(G, filename=fName, folder=baseDir + "gis/graph/")
        ox.save_gdf_shapefile(graph, filename=fName + ".gdf", folder=baseDir + "gis/graph/")
        ox.save_graphml(G, filename=gName, folder=baseDir + "gis/graph/")
        G = ox.load_graphml(filename=gName, folder=baseDir + "gis/graph/")
        ox.save_gdf_shapefile(G, filename=fName + ".gdf", folder=baseDir + "gis/graph/")
        G = ox.load_graphml(filename=fName + ".gdf", folder=baseDir + "gis/graph/")
        nx.readwrite.write_shp(G, baseDir + "gis/graph/berlin_street")
        nx.write_graphml_lxml(G, baseDir + "fName" + ".graphml")


def saveJson(G, fName, indent=0):
    """save a simple json"""
    l = [(u, v, k, x) for u, v, k, x in G.edges(keys=True, data="geometry") if x != None]
    print("geometry elements %d" % len(l))
    for i in l:
        del G[i[0]][i[1]][i[2]]['geometry']
    data = dict(nodes=list(G.nodes(data=True)), edges=list(G.edges(data=True)))
    json_str = json.dumps(data, indent=indent) + "\n"
    json_bytes = json_str.encode('utf-8')
    with gzip.GzipFile(fName, 'w') as fout:
        fout.write(json_bytes)


def loadJson(fName):
    """load a graph from a json"""
    G = nx.MultiGraph()
    G.graph['crs'] = {'init': 'epsg:4326'}
    G.graph['name'] = "local network"
    with gzip.GzipFile(fName, 'r') as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    data = json.loads(json_str)
    G.add_nodes_from(data['nodes'])
    G.add_edges_from(data['edges'])
    return G
