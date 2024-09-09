import sys
import networkx as nx
import osmnx as ox
import pandas as pd
import multiprocessing
from datetime import datetime
import geomadi.graph_ops as g_p

def plog(labS):
    print(labS)

def worker():
    """worker function"""
    print('Worker')
    return

# get nearest node, if actual node is not in the graph
def getNearestNode(nodeid, nodeL):
    coord_x = nodeL.loc[nodeL['osmid'] == nodeid]['x'].item()
    coord_y = nodeL.loc[nodeL['osmid'] == nodeid]['y'].item()
    coords = (coord_x, coord_y)
    nearest_node = ox.get_nearest_node(graph, coords, method='euclidean')
    return nearest_node

def progress(count, total, status='', delta_time=0):  # progressbar
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s %.2fh\r' % (bar, percents, '%', status, delta_time))
    sys.stdout.flush()


def calcRoute(graph, start_node, dest_list):
    """calculate the route from the OSM node to all nodes in the network"""
    try:  
        routes_dij = nx.single_source_dijkstra_path(G=graph, source=start_node, weight='weight')
        ##route = nx.shortest_path(graph, origin_node, destination_node, weight='weight')
    except:
        print("Start node " + str(start_node) + " not found")
        nearest_node = getNearestNode(start_node, nodeL)
        print("Using nearest node instead, nodeid: " + str(nearest_node))
        routes_dij = nx.single_source_dijkstra_path(G=graph, source=nearest_node, weight='weight')

    routeL = []
    for route in list(routes_dij.items()):
        if route[0] not in dest_list:
            continue
        if len(route[1]) < 2:  # to make sure it is a route
            continue
        length = 0
        motorway_length = 0
        trunk_length = 0
        primary_length = 0
        secondary_length = 0
        for i in range(len(route[1]) - 1):
            edgeData = graph.get_edge_data(route[1][i], route[1][i + 1])
            try:
                length = length + edgeData[0].get('length') / 1000.
            except:
                continue
            edgeType = edgeData[0].get('highway')
            segDist = edgeData[0].get('length') / 1000.
            if edgeType in ('trunk', 'trunk_link'):
                trunk_length = trunk_length + segDist
            elif edgeType in ('primary', 'primary_link'):
                primary_length = primary_length + segDist
            elif edgeType in ('secondary', 'secondary_link'):
                secondary_length = secondary_length + segDist
            elif edgeType in ('motorway', 'motorway_link'):
                motorway_length = motorway_length + segDist
        n_edge = len(route[1]) - 1
        routeL.append([route[0], length, motorway_length, primary_length, secondary_length, n_edge])
    routeL = pd.DataFrame(routeL, columns=["end","length","motorway_length","primary_length","secondary_length","n_edge"]) 
    return routeL


def routeList(graph, nodeL):
    start_time = datetime.now()
    plog("-----------------------undirect-graph---------------------------")
    graph = graph.to_undirected()
    for edge in graph.edges():
        edgeD = graph.get_edge_data(edge[0], edge[1])
        for i, x in edgeD.items():
            weight = 1000
            if x.get('highway') == 'motorway':
                weight = 1
            graph[edge[0]][edge[1]][i]["weight"] = weight
    start_list = list(nodeL['osmid'])
    dest_list = list(nodeL['osmid'])
    plog("------------------------start-routing-------------------------")
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker)
        jobs.append(p)
        p.start()
    # pool = Pool(6)
    routeD = []
    for count, start in nodeL.iterrows():
        end_time = datetime.now()
        delta_time = int((end_time - start_time).total_seconds()) / 3600.
        progress(count, len(start_list), status='', delta_time=delta_time)
        start_node = start['osmid']
        route = calcRoute(graph, start_node, dest_list)
        route = pd.merge(route, nodeL, left_on="end", right_on='osmid', how="left")
        if len(route) == 0:
            continue
        route.loc[:,"octree8_start"] = start['octree8']
        routeL = route[['octree8_start','octree8','length','motorway_length','primary_length','secondary_length','n_edge']]
        routeD.append(routeL)

    print('origins: ' + str(len(start_list)))
    print('time:    ' + str(delta_time))
    print('origins per sec: ' + str(delta_time / len(start_list)))
    print('rel:     ' + str((len(start_list) * (len(start_list)))))
    print('rel per sec: ' + str(delta_time / (len(start_list) * (len(start_list)))))
    return pd.concat(routeD)
