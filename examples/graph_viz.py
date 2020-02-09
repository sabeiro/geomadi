import math

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import scipy as sp


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
