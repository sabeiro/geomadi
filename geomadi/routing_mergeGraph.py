import geopandas as gpd
import osmnx as ox
import pandas as pd

import importlib
import geomadi.graph_ops as g_o
import geomadi.graph_viz as g_v

importlib.reload(g_o)
importlib.reload(g_v)
gName = "berlin_street.graphml"
pName = "raw/opt/route.csv"

if True:
    print("-------------------------create-the-graph-if-missing-----------------------------")
    netw = gpd.read_file(baseDir + "gis/geo_berlin/berlin_street.shp")
    G = g_o.line2graph(netw)
    G_type = g_o.removeType(G, move_type="driver")
    G_simp = ox.simplify_graph(G_type)
    G_proj = ox.project_graph(G_simp, to_crs={'init': 'epsg:4326'})
    # ox.plot_graph(G_proj)
    g_o.saveJson(G_proj, baseDir + "gis/graph/berlin_street_driver" + ".json.gz")

    print("-------------------------create-the-move-type-----------------------------")
    route = pd.read_csv(baseDir + pName)
    G = g_o.loadJson(baseDir + "gis/graph/berlin_street_driver.json.gz")
    route.loc[:, "node"] = ''
    for i, g in route.iterrows():
        node = g_o.getNearestNode(G, g['y'], g['x'])
        print(i, node)
        route.loc[i, "node"] = node
    route.to_csv(baseDir + pName, index=False)
