import numpy as np
import pandas as pd

import importlib
import datetime
import geomadi.graph_ops as gro
import geomadi.graph_viz as g_v
import geomadi.geo_octree as g_o

importlib.reload(gro)
importlib.reload(g_v)

gO = g_o.h3tree()

gName = "berlin_street.graphml"
pName = "raw/opt/route.csv"

if False:
    print("-------------------------create-the-graph-if-missing-----------------------------")
    netw = gpd.read_file(baseDir + "gis/geo_berlin/berlin_street.shp")
    tL = ['motorway', 'motorway_link', 'primary', 'residential', 'secondary', 'secondary_link', 'tertiary', 'trunk',
          'trunk_link']
    netd = netw[netw['fclass'].isin(tL)]
    importlib.reload(gro)
    G = gro.line2graph(netd)
    G_proj = gro.simplifyGraph(G, connection="strong")
    G_wei = gro.weightSpeed(G_proj)
    ox.plot_graph(G_wei)
    # ox.plot_graph(G)
    gro.saveJson(G_wei, baseDir + "gis/graph/berlin_street_driver.json.gz", indent=0)

if False:
    print("-------------------------find-the-first-node-if-missing-----------------------------")
    route = pd.read_csv(baseDir + pName)
    G = gro.loadJson(baseDir + "gis/graph/berlin_street_driver.json.gz")
    neiL = gro.getNeighbors(G, route[['x', 'y']])
    route.loc[:, "node"] = neiL
    route.to_csv(baseDir + pName, index=False)


def worker(start):
    end_time = datetime.now()
    delta_time = int((end_time - start_time).total_seconds()) / 3600.
    gro.progress(count, plz_sel.shape[0], status='', delta_time=delta_time)
    routeL = gro.calcRoute(graph, start, zipN, colL)
    routeL.to_csv(baseDir + "gis/graph/" + outF, mode="a", index=False, header=False)


def routeOsm():
    importlib.reload(gro)
    start_time = datetime.datetime.now()
    route = pd.read_csv(baseDir + pName)
    ware = pd.read_csv(baseDir + "raw/opt/warehouse.csv")
    route1 = pd.read_csv(baseDir + "raw/opt/benchmark/sol_job_s592_v9_routific.csv")
    geoL = list(route['geohash']) + list(ware['geohash']) + list(route1['geohash'])
    geoL = np.unique(geoL)
    spotL = pd.DataFrame({"geohash": geoL})
    colL = ['origin', 'destination', 'length', 'weight', 'duration']
    outF = "spot2spot.csv.gz"
    routeL = pd.DataFrame(columns=colL)
    routeL.to_csv(baseDir + "gis/graph/" + outF, compression="gzip", index=False)
    for count, start in spotL.iterrows():
        delta_time = int((datetime.datetime.now() - start_time).total_seconds()) / 3600.
        routeL = []
        for end in spotL['geohash']:
            routeL.append(gro.routeOsrm(start['geohash'], end))
        gro.progress(count, spotL.shape[0], status='', delta_time=delta_time)
        routeL = pd.DataFrame(routeL, columns=colL)
        routeL.to_csv(baseDir + "gis/graph/osm_" + outF, compression="gzip", mode="a", index=False, header=False)


def routeGraph():
    print("------------------------load-graph--------------------------------")
    importlib.reload(gro)
    start_time = datetime.datetime.now()
    G = gro.loadJson(baseDir + "gis/graph/berlin_street_driver.json.gz")
    route = pd.read_csv(baseDir + pName)
    ware = pd.read_csv(baseDir + "raw/opt/warehouse.csv")
    route1 = pd.read_csv(baseDir + "raw/opt/benchmark/sol_job_s592_v9_routific.csv")
    geoL = list(route['geohash']) + list(ware['geohash']) + list(route1['geohash'])
    geoL = np.unique(geoL)
    spotL = pd.DataFrame({"geohash": geoL})
    posX = spotL['geohash'].apply(lambda x: gO.decode(x))
    spotL.loc[:, "x"] = [x[0] for x in posX]
    spotL.loc[:, "y"] = [x[1] for x in posX]
    # startL = np.unique(spotL['node'].values)
    startL = gro.getNeighbors(G, spotL[['x', 'y']])
    spotL.loc[:, "node"] = startL
    destL = startL
    print("------------------------start-routing-------------------------")
    colL = ['origin', 'destination', 'length', 'weight', 'duration']
    outF = "spot2spot.csv.gz"
    routeL = pd.DataFrame(columns=colL)
    routeL.to_csv(baseDir + "gis/graph/" + outF, compression="gzip", index=False)
    compL = []
    importlib.reload(gro)
    for count, start in spotL.iterrows():
        delta_time = int((datetime.datetime.now() - start_time).total_seconds()) / 3600.
        node = start['node']
        routeL, completion = gro.calcRoute(G, start, destL, colL)
        routeL = routeL.merge(spotL[['node', 'geohash']], left_on="origin", right_on="node")
        routeL = routeL.merge(spotL[['node', 'geohash']], left_on="destination", right_on="node", suffixes=["_o", "_d"])
        routeL.drop(columns=['origin', 'destination'], inplace=True)
        routeL.rename(columns={"geohash_o": "origin", "geohash_d": "destination"}, inplace=True)
        routeL = routeL[colL]
        compL.append(completion)
        gro.progress(count, spotL.shape[0], status='', delta_time=delta_time, completion=np.mean(compL))
        routeL.to_csv(baseDir + "gis/graph/" + outF, compression="gzip", mode="a", index=False, header=False)

    print("done: completion rate %.2f in %.2f min" % (np.mean(compL), delta_time * 60))


if __name__ == '__main__':
    routeGraph()
    # routeOsm()

if False:
    print("-----------------------route-test--------------------")
    G = G_wei.to_undirected()
    nodeL = list(G.nodes())
    i, j = np.random.choice(nodeL), np.random.choice(nodeL)
    g_v.showShortest(G_wei, i, j)

if False:
    print("-------------------check-graph-difference----------------")
    pair1 = pd.read_csv(baseDir + "gis/graph/osm_" + outF, compression="gzip")
    pair2 = pd.read_csv(baseDir + "gis/graph/" + outF, compression="gzip")
    pair2['length'] = pair2['length'] * 1000.
    pair1 = pair1.groupby(['origin', 'destination']).first().reset_index()
    pair2 = pair2.groupby(['origin', 'destination']).first().reset_index()
    pair = pair1.merge(pair2, on=["origin", "destination"], how="inner")
    oc = pair['origin'].apply(lambda x: gO.decode(x))
    dc = pair['destination'].apply(lambda x: gO.decode(x))
    pair['dist'] = [1000. * g_e.haversine(x[0], x[1], y[0], y[1]) for x, y in zip(oc, dc)]
    print(sp.stats.pearsonr(pair['length_x'], pair['length_y'])[0])
    print(sp.stats.pearsonr(pair['length_x'], pair['dist'])[0])
    print(sp.stats.pearsonr(pair['dist'], pair['length_y'])[0])
    pair['diff'] = 2. * (pair['length_x'] - pair['length_y']) / (pair['length_x'] + pair['length_y'])
    pair['diff2'] = 2. * (pair['length_x'] - pair['dist']) / (pair['length_x'] + pair['dist'])
    pairD = pair.pivot_table(index="origin", columns="destination", values="diff", aggfunc=np.sum)
    plt.imshow(pairD)
    plt.show()
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(pair['diff'], bins=20, label="graph difference")
    ax[1].hist(pair['diff2'], bins=20, label="air distance")
    ax[0].legend()
    ax[1].legend()
    plt.show()

    t_v.plotCorr(pair[['length_x', 'length_y', 'dist']].values, labV=["osrm", "simplified", "air_dist"])
    plt.show()

if False:
    pass

if False:
    print('---------------------anaylze-matrix---------------------')

    gO = gtree.h3tree()
    pairL = pd.read_csv(baseDir + "gis/graph/" + outF, compression="gzip")
    odm = pairL.pivot_table(index="origin", columns="destination", values="length", aggfunc=np.sum)
    odw = pairL.pivot_table(index="origin", columns="destination", values="weight", aggfunc=np.sum)
    odm = odm[odm.index.isin(odm.columns)]
    odm = odm.loc[:, odm.columns.isin(odm.index)]
    odm.replace(float('nan'), 0., inplace=True)
    odw = odw[odw.index.isin(odw.columns)]
    odw = odw.loc[:, odw.columns.isin(odw.index)]
    odw.replace(float('nan'), 0., inplace=True)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("distance matrix")
    ax[0].imshow(odm)
    ax[1].set_title("weight matrix")
    ax[1].imshow(odw)
    plt.show()

    posL = [gO.decode(x) for x in odw.columns]
    posL = [{"x": x, "y": y} for x, y in posL]
    posL = pd.DataFrame(posL)

    distM = pd.DataFrame(spatial.distance_matrix(posL.values, posL.values), index=posL.index, columns=posL.index)
    markovC = 1. / distM
    markovC.replace(float('inf'), 0, inplace=True)
    markovC = markovC / markovC.sum(axis=0)
    markovC = markovC ** 5
    markovC = markovC / markovC.sum(axis=0)
    m = 1. / (len(markovC))
    markovC[markovC < m] = 0.
    markovC1 = markovC / markovC.sum(axis=0)

    markovC = odw
    markovC.replace(float('inf'), 0, inplace=True)
    markovC.replace(float('nan'), 0, inplace=True)
    per = [np.percentile(x.values, 90) for i, x in markovC.iterrows()]
    m = np.mean(per)
    markovC[markovC < m] = 0.
    print((markovC > 0).sum().sum() / len(markovC))
    markovC = markovC / markovC.sum(axis=0)
    markovC = markovC ** 5
    markovC.replace(float('nan'), 0, inplace=True)
    markovC = markovC / markovC.sum(axis=0)
    markovC.replace(float('inf'), 0, inplace=True)
    m = 10. / (1. * len(markovC))
    markovC[markovC < m] = 0.
    print((markovC > 0).sum().sum() / len(markovC))
    markovC = markovC / markovC.sum(axis=0)
    markovC.replace(float('nan'), 0, inplace=True)
    trajS = sp.sparse.coo_matrix(markovC)
    G = nx.Graph(trajS)
    trajS = sp.sparse.coo_matrix(markovC1)
    G1 = nx.Graph(trajS)
    print("%.2f - %.2f" % (len(G.edges) / len(markovC), len(G1.edges) / len(markovC1)))
    pos, lab = {}, {}
    for i, x in posL.iterrows(): pos[i] = [x['x'], x['y']]
    ew = [10. * o['weight'] for u, v, o in G.edges(data=True)]
    ew1 = [10. * o['weight'] for u, v, o in G1.edges(data=True)]
    fig, ax = plt.subplots(1, 1)
    nx.draw_networkx_nodes(G, pos, alpha=0.1, ax=ax)
    nx.draw_networkx_edges(G, pos, width=ew, edge_color="red", alpha=0.2, ax=ax)  # ,width=ew)
    nx.draw_networkx_edges(G1, pos, width=ew1, edge_color="green", alpha=0.2, ax=ax)  # ,width=ew)
    ax.set_axis_off()
    # ax.set_xlim(np.percentile(posL['x'],[0.5,99.9]))
    # ax.set_ylim(np.percentile(posL['y'],[1,99]))
    plt.show()

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
