import math

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import scipy as sp


def plot(G):
    """plot a graph with the simplest configuration"""
    # ox.plot_graph(G)
    palette = plt.get_cmap('YlGnBu')
    ew = [.5 * G[u][v][k]['weight'] for u, v, k, o in G.edges(keys=True, data=True)]
    pos, lab = {}, {}
    for i, x in G.nodes(data='pos'): pos[i] = x
    nc = 'blue'  # ['red' if u in nodeL else 'blue' for u in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=nc, cmap=plt.get_cmap('jet'), node_size=5)
    nx.draw_networkx_edges(G, pos, width=ew, edge_color='black', arrows=False)
    # nx.draw_networkx_labels(G, pos)
    plt.show()


def graphRoute(route, ax=None):
    """check route displaying a graph"""
    G = nx.MultiDiGraph()
    G.graph['crs'] = {'init': 'epsg:4326'}
    G.graph['name'] = "local network"
    colorL = ['black', 'blue', 'red', 'green', 'brown', 'orange', 'purple', 'magenta', 'olive', 'maroon', 'steelblue',
              'midnightblue', 'darkslategrey', 'crimson', 'teal', 'darkolivegreen']
    colorL = colorL + colorL
    for i, g in route.iterrows():
        G.add_node(i, pos=(g[['x', 'y']]), x=g['x'], y=g['y'], key=0, potential=g['potential'], active=g['active'],
                   n=g['occupancy'])
    for k, g in route.groupby("agent"):
        if k == 0: continue
        idx = list(g.index)
        if len(idx) < 2: continue
        pair_idx = [(i, j) for i, j in zip(idx[:-1], idx[1:])]
        for i, j in pair_idx:
            g1, g2 = route.loc[i], route.loc[j]
            i1, i2 = g1['geohash'], g2['geohash']
            dist = math.hypot(g1['x'] - g2['x'], g1['y'] - g2['y'])
            G.add_edge(i, j, key=k, origin=i1, destination=i2, length=dist, width=1, speed=1, color=colorL[k])
    G = ox.project_graph(G)
    colorL = [cm.get_cmap('plasma')(x) for x in np.linspace(0, 1, 6)]
    nc = ['blue' if G.nodes()[x]['active'] else 'red' for x in G.nodes()]
    nbins = min(20, len(set(route['potential'])))
    cmap = matplotlib.cm.get_cmap('plasma')
    max_pot = np.percentile(route['potential'], 95)
    nc = [cmap(x['potential'] / max_pot) for i, x in G.nodes(data=True)]
    # nc = ox.get_node_colors_by_attr(G,'potential',cmap='plasma',num_bins=nbins-2)
    ns = [20. * x['n'] for i, x in G.nodes(data=True)]
    # ec = ox.get_edge_colors_by_attr(G, attr='length')
    ec = [o['color'] for u, v, k, o in G.edges(data=True, keys=True)]
    # if len(ec) == 0: return plt.subplots(1,1)
    ew = [200. * o['length'] for u, v, k, o in G.edges(data=True, keys=True)]
    pos, lab = {}, {}
    for i, x in G.nodes(data=True): pos[i] = [x['lon'], x['lat']]
    for i, x in G.nodes(data=True): lab[i] = i
    # nx.draw_networkx_edge_labels(G,pos,edge_colors=ec)
    if ax == None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(10, 6)
    # nx.draw(G,pos,node_color=nc,node_size=ns,edge_color=ec,edge_cmap=plt.cm.Reds)
    nx.draw_networkx_nodes(G, pos, node_color=nc, node_size=ns, alpha=0.1, ax=ax)
    # nx.draw_networkx_labels(G,pos,lab,font_size=8,ax=ax,alpha=0.3)
    nx.draw_networkx_edges(G, pos, width=ew, alpha=0.5, edge_color=ec, ax=ax, edge_cmap=plt.cm.Reds)
    ax.set_axis_off()
    ax.set_xlim(np.percentile(route['x'], [1, 99]))
    ax.set_ylim(np.percentile(route['y'], [1, 99]))
    return ax
    if False:
        gdf_edges = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        graph_map = ox.plot_graph_folium(G, tiles='stamenterrain', edge_color=ec, edge_width=2)
        origin_node = list(G.nodes())[0]
        destination_node = list(G.nodes())[-1]
        route = nx.shortest_path(G, origin_node, destination_node)
        route_map = ox.plot_route_folium(G, route)
        graph_map.save(baseDir + "www/gis/folium.html")

        ox.make_folium_polyline(gdf_edges, ew, ew, popup_attribute=None)
        plot_graph_folium(gdf_edges, graph_map=None, popup_attribute=None, tiles='cartodbpositron', zoom=1,
                          fit_bounds=True, edge_width=5, edge_opacity=1)
    return fig, ax


def graphAdjacency(markovC, posL=[None]):
    """check route displaying a graph"""
    trajS = sp.sparse.coo_matrix(markovC)
    G = nx.Graph(trajS)
    print(len(G.edges) / len(markovC))
    # G = ox.simplify_graph(G)
    if any(posL) == None:
        pos = nx.spring_layout(G)
    else:
        pos, lab = {}, {}
        for i, x in posL.iterrows(): pos[i] = [x['x'], x['y']]
    ew = [200. * o['weight'] for u, v, o in G.edges(data=True)]
    fig, ax = plt.subplots(1, 1)
    nx.draw_networkx_nodes(G, pos, alpha=0.1, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.05, ax=ax)  # ,width=ew)
    ax.set_axis_off()
    # ax.set_xlim(np.percentile(posL['x'],[0.5,99.9]))
    # ax.set_ylim(np.percentile(posL['y'],[1,99]))
    plt.show()


def exampleGraph():
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('C', 'D'), ('G', 'D')], weight=1)
    G.add_edges_from([('D', 'A'), ('D', 'E'), ('B', 'D'), ('D', 'E')], weight=2)
    G.add_edges_from([('B', 'C'), ('E', 'F')], weight=3)
    G.add_edges_from([('C', 'F')], weight=4)
    val_map = {'A': 1.0, 'D': 0.5714285714285714, 'H': 0.0}
    values = [val_map.get(node, 0.45) for node in G.nodes()]
    edge_labels = dict([((u, v,), d['weight']) for u, v, d in G.edges(data=True)])
    red_edges = [('C', 'D'), ('D', 'A')]
    edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
    pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos, node_color=values, node_size=1500, edge_color=edge_colors, edge_cmap=plt.cm.Reds)
    plt.show()


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


def showShortest(G, start_node=246539610, end_node=59768831):
    edge_color = "black"  # colorEdges(G)
    route = nx.shortest_path(G, start_node, end_node, weight='weight')
    fig, ax = ox.plot_graph_route(G, route=route, edge_color=edge_color, node_color='none')


if False:
    print("-------------graph-style-examples-----------------")
    print(trajB.shape)
    trajM = trajB.pivot_table(index="origin", columns="destination", values="n", aggfunc=np.sum)
    setR = trajM.index.isin(trajM.columns)
    setC = trajM.columns.isin(trajM.index)
    trajM = trajM.loc[setR, setC]
    plt.imshow(trajM.values)
    plt.show()

    trajS = sp.sparse.coo_matrix(trajM, dtype=np.int8)
    G = nx.Graph(trajS)
    simp_G = ox.simplify_graph(G)

    val_map = {'A': 1.0, 'D': 0.5714285714285714, 'H': 0.0}
    values = [val_map.get(node, 0.45) for node in G.nodes()]
    edge_labels = dict([((u, v,), d['weight']) for u, v, d in G.edges(data=True)])
    red_edges = [('C', 'D'), ('D', 'A')]
    edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
    pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos, node_color=values, node_size=1500, edge_color=edge_colors, edge_cmap=plt.cm.Reds)
    plt.show()

    G = nx.florentine_families_graph()
    adjacency_matrix = nx.adjacency_matrix(G)
    G = nx.fast_gnp_random_graph(100, 0.04)
    adj_matrix = nx.adjacency_matrix(G)
    # The actual work
    # You may prefer `nx.from_numpy_matrix`.
    G2 = nx.from_scipy_sparse_matrix(trajS)
    G2 = nx.from_scipy_sparse_matrix(adjacency_matrix)
    nx.draw_circular(G2)
    plt.axis('equal')
    plt.plot()

    # import igraph
    conn_indices = np.where(a_numpy)
    weights = a_numpy[conn_indices]
    edges = zip(*conn_indices)
    G = igraph.Graph(edges=edges, directed=True)
    G.vs['label'] = node_names
    G.es['weight'] = weights
    G.es['width'] = weights
    igraph.plot(G, layout="rt", labels=True, margin=80)
    plt.show()
