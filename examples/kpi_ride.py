import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely as sh
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import geomadi.geo_octree as g_o
import lernia.train_viz as t_v
import networkx as nx
import geomadi.geo_enrich as g_e

gO = g_o.h3tree(BoundBox=[5.866,47.2704,15.0377,55.0574])
precDigit = 8

if False:
    print('----------------------group-all-single-days----------------')
    projDir = baseDir + "raw/traj/"
    fL = os.listdir(projDir)
    for precDigit in [8,9,10]:
        trajL = []
        for f in fL:
            print(f)
            traj = pd.read_csv(projDir+f,compression="gzip",converters={"tx":eval,"bound":eval,"state":eval})
            traj.loc[:,"origin"] = traj['tx'].apply(lambda x: gO.encode(x[0][1],x[0][2],precision=precDigit))
            traj.loc[:,"destination"] = traj['tx'].apply(lambda x: gO.encode(x[-1][1],x[-1][2],precision=precDigit))
            traj.loc[:,"n"] = 1
            traj = traj.groupby(['usertype','zoneid','origin','destination']).agg(sum).reset_index()
            trajL.append(traj)

        trajL = pd.concat(trajL)
        trajL = trajL.groupby(['usertype','zoneid','origin','destination']).agg(sum).reset_index()
        trajL.to_csv(baseDir + "raw/ride/odm_"+str(precDigit)+".csv.gz",index=False)
        line = trajL.apply(lambda x: sh.geometry.LineString([gO.decode(x['origin']),gO.decode(x['destination'])]),axis=1 )
        trajG = gpd.GeoDataFrame(trajL.copy(),geometry=line)
        trajG[trajG['n']>1].to_file(baseDir + "gis/ride/traj_"+str(precDigit)+"_geohash.shp")

if False:
    print('---------------------start-end-length-number-----------------------')
    precDigit = 8
    for precDigit in [8,9,10]:
        print(precDigit)
        trajL = pd.read_csv(baseDir + "raw/ride/odm_"+str(precDigit)+".csv.gz")
        trajL = trajL[['usertype','zoneid','origin','destination','n','dt']]
        trajL = trajL[trajL['n'] > 2]
        startL = trajL.groupby(['usertype','zoneid','origin']).agg(sum).reset_index()
        endL = trajL.groupby(['usertype','zoneid','destination']).agg(sum).reset_index()
        pointL = startL.merge(endL,left_on=['usertype','zoneid','origin'],right_on=['usertype','zoneid','destination'],how='inner',suffixes=["_s","_e"])
        pointL.drop(columns=['destination'],inplace=True)
        pointL.loc[:,"diff"] = 2.*(pointL['n_s'] - pointL['n_e'])/(pointL['n_s']+pointL['n_e'])
        pointL = pointL[pointL['usertype'] == 'CUSTOMER']
        hexL = pointL['origin'].apply(lambda x: sh.geometry.Polygon(gO.decodePoly(x)))
        pointG = gpd.GeoDataFrame(pointL,geometry=hexL) 
        pointL.to_csv(baseDir + "raw/ride/start_end_"+str(precDigit)+".csv.gz",compression="gzip",index=False)
        pointG.to_file(baseDir + "gis/ride/start_end_"+str(precDigit)+".shp")
    
if False:
    print('-----------------------merge-layers---------------------')
    precDigit = 8
    for precDigit in [8,9,10]:
        print(precDigit)
        pointL = pd.read_csv(baseDir + "raw/ride/start_end_"+str(precDigit)+".csv.gz",compression="gzip")
        pointL = pointL[pointL['zoneid'] == 'berlin']
        openL = pd.read_csv(baseDir + "raw/app/open_dens_"+str(precDigit)+".csv.gz",compression="gzip")
        mixL = pd.merge(openL[['geohash','n']],pointL[['origin','n_s','n_e']],left_on="geohash",right_on="origin",how="outer")
        mixL.to_csv(baseDir + "raw/app/offer_demand_"+str(precDigit)+".csv.gz",compression="gzip",index=False)
        
        
if False:
    print('-------------------graph-from-rides--------------------')
    trajL = pd.read_csv(baseDir + "raw/ride/odm_"+str(precDigit)+".csv.gz")
    trajB = trajL[trajL['zoneid'] == 'berlin']
    trajB = trajB[trajB['usertype'] == 'CUSTOMER']
    trajB = trajB[trajB['n'] > 5]
    trajB = trajB[trajB['origin'] != trajB['destination']]
    trajB.loc[:,"pos_origin"] = trajB['origin'].apply(lambda x: gO.decode(x))
    trajB.loc[:,"pos_destination"] = trajB['destination'].apply(lambda x: gO.decode(x))
    trajB.loc[:,"distance"] = trajB.apply(lambda x: np.sqrt((x['pos_origin'][0]-x['pos_destination'][0])**2) + np.sqrt((x['pos_origin'][1]-x['pos_destination'][1])**2),axis=1)
    G = g_e.odm2graph(trajB)
    G = ox.simplify_graph(G)
    
    ox.plot_graph(G,edge_linewidth=0.5,edge_color='black')

    
if False:
    print('-------------------graph-per-user-----------------------')

    mapiL = np.unique(mapi['state'])
    custL = np.unique(mapi['usertype'])
    
    custM = []
    for c in custL:
        print(c)
        user = traj[traj['usertype'] == c]
        mapiM = pd.DataFrame(np.zeros((6,6)),columns=mapiL,index=mapiL)
        for i,g in user.iterrows():
            l = list(g['state'])
            for j1,j2 in zip(l[:-1],l[1:]):
                mapiM.loc[j1,j2] += 1.
        custM.append({"name":c,"matrix":mapiM})

    graphL = []
    for c in custM:
        M = c['matrix']
        m = M.sum(axis=1)
        M = (M.T/m).T
        M = M[m>0]
        G = nx.Graph()
        for i in M.index: G.add_node(i,size=1)
        for i,g in M.iterrows():
            g = g[g>0]
            for j,k in zip(g.index,g):
                G.add_edge(g.name,j,weight=k,color='b')
        c['graph'] = G

    fig, ax = plt.subplots(1,3)
    for i,c in enumerate(custM):
        G = c['graph']
        labels = {}
        for n in G.nodes: labels[n] = n.split("_")[0]
        val_map = {'A': 1.0,'D': 0.5714285714285714,'H': 0.0}
        values = [val_map.get(node, 0.45) for node in G.nodes()]
        edge_labels = dict([((u,v,),"%.2f" % d['weight']) for u,v,d in G.edges(data=True)])
        w = [20.*d['weight'] for u,v,d in G.edges(data=True)]
        colors = [G[u][v]['color'] for u,v in G.edges()]
        #colors = range(7)
        pos = nx.circular_layout(G)
        ax[i].set_title(c['name'])
        #nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,ax=ax[i])
        # nx.draw(G,pos,node_color=values,node_size=1500,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
        nx.draw_networkx_edges(G,pos,width=w,alpha=0.3,edge_color=colors,ax=ax[i])
        nx.draw_networkx_nodes(G,pos,node_color='g',node_size=nodS,alpha=0.3,ax=ax[i])
        nx.draw_networkx_labels(G,pos,labels,font_size=12,ax=ax[i])
        ax[i].set_axis_off()
        plt.show()
    

# from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
# A = to_agraph(G)
# A.layout('dot')
# A.draw('abcd.png')




if False:
    plt.hist(traj['n'])
    plt.show()

    plt.hist(traj['dt'])
    plt.show()

    t = pd.crosstab(traj['state'],traj['locked'])
    




# restsummary_postgres_rest start-end
# datalake.mapi_scooter_status_change
# stg.device_location_event

# ods -  
# stg
# weather


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
