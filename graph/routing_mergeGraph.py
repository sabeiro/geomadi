#%pylab inline
import osmnx as ox
import networkx as nx
import csv

#load input
graph1 = ox.load_graphml( filename="germany_split_motorway_motorwaylink.graphml")
print(["motorway_motorwaylink loaded"])
    
graph2 = ox.load_graphml( filename="germany_split_trunk_trunk_link.graphml")
print(["trunk_trunk_link loaded"])

graph3 = ox.load_graphml( filename="germany_split_primlink.graphml")
print(["primlink loaded"])

graph4 = ox.load_graphml( filename="germany_split_prim.graphml")
print(["prim loaded"])

graph5 = ox.load_graphml( filename="germany_split_seclink.graphml")
print(["seclink loaded"])
    
graph6 = ox.load_graphml( filename="germany_split_sec.graphml")
print(["sec loaded"])    

#compose
c_graphs = [graph1, graph2, graph3, graph4, graph5, graph6]
composed_G = nx.compose_all(c_graphs)
print(["composed_G:"])   
print(["nodes: " + str(len(composed_G.nodes(data=True)))])   
print(["edges: " + str(len(composed_G.edges(data=True)))])
    
#clearing
graph1.clear()
graph2.clear()
graph3.clear()
graph4.clear()
graph5.clear()
graph6.clear()
print(["input cleared from memory"])

#simplify
simp_G = ox.simplify_graph(composed_G)
print(["simp_G:"])   
print(["nodes: " + str(len(simp_G.nodes(data=True)))])   
print(["edges: " + str(len(simp_G.edges(data=True)))])

#connected
connected_G = max(nx.strongly_connected_component_subgraphs(simp_G), key=len)
print(["connected_G:"])   
print(["nodes: " + str(len(connected_G.nodes(data=True)))])   
print(["edges: " + str(len(connected_G.edges(data=True)))])

#projection
graph_proj = ox.project_graph(connected_G)
print(["proj done"])  

#save
ox.save_graphml(graph_proj, filename="allGermany_allstreetsUntilSec_proj.graphml")
print(["saving done"])  

import osmnx as ox
import networkx as nx
graph1 = ox.load_graphml(filename="germany_split_motorway_motorwaylink.graphml")
print ('motorway_motorwaylink loaded')
graph2 = ox.load_graphml( filename="germany_split_trunk_trunk_link.graphml")
print ('trunk_trunk_link loaded')
graph3 = ox.load_graphml( filename="germany_split_primlink.graphml")
print ('primlink loaded')
graph4 = ox.load_graphml( filename="germany_split_prim.graphml")
print ('prim loaded')
graph5 = ox.load_graphml( filename="germany_split_seclink.graphml")
print ('seclink loaded')
graph_01 = ox.load_graphml( filename="graph_bremen.graphml")
graph_02 = ox.load_graphml( filename="graph_hamburg.graphml")
graph_03 = ox.load_graphml( filename="graph_baden-wuerttemberg.graphml")
graph_04 = ox.load_graphml( filename="graph_brandenburg.graphml")
print ('sec1 loaded')
graph_05 = ox.load_graphml( filename="graph_hessen.graphml")
graph_06 = ox.load_graphml( filename="graph_niedersachsen.graphml")
graph_07 = ox.load_graphml( filename="graph_nordrhein-westfalen.graphml")
graph_08 = ox.load_graphml( filename="graph_sachsen.graphml")
print ('sec2 loaded')
graph_09 = ox.load_graphml( filename="graph_thueringen.graphml")
graph_10 = ox.load_graphml( filename="graph_schleswig-holstein.graphml")
graph_11 = ox.load_graphml( filename="graph_rheinland-pfalz.graphml")
graph_12 = ox.load_graphml( filename="graph_saarland.graphml")
print ('sec3 loaded')
graph_13 = ox.load_graphml( filename="graph_sachsen-anhalt.graphml")
graph_14 = ox.load_graphml( filename="graph_berlin.graphml")
graph_15 = ox.load_graphml( filename="graph_mecklenburg-vorpommern.graphml")
graph_16 = ox.load_graphml( filename="graph_bayern.graphml")
print ('sec4 loaded')
c_graphs = [graph1, graph2, graph3, graph4, graph5, graph_01,graph_02,graph_03,graph_04,graph_05,graph_06,graph_07,graph_08,graph_09,graph_10,graph_11,graph_12,graph_13,graph_14,graph_15,graph_16]
composed_G = nx.compose_all(c_graphs)
print ('composed_G:')
print ("nodes: " + str(len(composed_G.nodes(data=True))))
print ("edges: " + str(len(composed_G.edges(data=True))))
graph1.clear()
graph2.clear()
graph3.clear()
graph4.clear()
graph5.clear()
graph_01.clear()
graph_02.clear()
graph_03.clear()
graph_04.clear()
graph_05.clear()
graph_06.clear()
graph_07.clear()
graph_08.clear()
graph_09.clear()
graph_10.clear()
graph_11.clear()
graph_12.clear()
graph_13.clear()
graph_14.clear()
graph_15.clear()
graph_16.clear()
print ('input cleared')
simp_G = ox.simplify_graph(composed_G)
print ('simp_G:')
print ("nodes: " + str(len(simp_G.nodes(data=True))))
print ("edges: " + str(len(simp_G.edges(data=True))))
connected_G = max(nx.strongly_connected_component_subgraphs(simp_G), key=len)
print ('connected_G:')
print ("nodes: " + str(len(connected_G.nodes(data=True))))
print ("edges: " + str(len(connected_G.edges(data=True))))
graph_proj = ox.project_graph(connected_G)
print ('proj done')
ox.save_graphml(graph_proj, filename="allGermany_allstreetsUntilSec_proj.graphml")
