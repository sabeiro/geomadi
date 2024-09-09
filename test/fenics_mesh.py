#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from dolfin import *
from fenics import *
import numpy as np
import seaborn as sns
import scipy.stats as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import shapely as sh
import geo_geohash
from shapely.ops import cascaded_union
from scipy import signal as sg
import cv2


def plog(text):
    print(text)

with open(baseDir + '/credenza/geomadi.json') as f:
    cred = json.load(f)

import pymongo
client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])
cent = pd.read_csv(baseDir + "raw/roda/fem_centroid.csv")
cent = cent.sort_values('AreaInters')
act = pd.read_csv(baseDir + "raw/roda/fem_act.csv")
BBox = [min(act['x']),max(act['x']),min(act['y']),max(act['y'])]

coll = client["tdg_infra"]["segments_col"]
neiN = coll.find({'loc':{'$geoWithin':{'$box':[ [BBox[0],BBox[2]],[BBox[1],BBox[3]] ]}}})
#neiN = coll.find({"$or":queryL})
neiN = coll.find({'loc':{'$geoIntersects':{'$geometry':{
    "type":"Polygon"
    ,"coordinates":[
    [ [BBox[0],BBox[3]],[BBox[1],BBox[3]],[BBox[1],BBox[2]],[BBox[0],BBox[2]],[BBox[0],BBox[3]] ]
    ]}}}})
pointL, lineL, lineS = [],[],[]
for neii in neiN:
    pointL.append({"id":neii['src'],"x":neii['loc']['coordinates'][0][0],"y":neii['loc']['coordinates'][0][1],"speed":neii['maxspeed']})
    pointL.append({"id":neii['trg'],"x":neii['loc']['coordinates'][1][0],"y":neii['loc']['coordinates'][1][1],"speed":neii['maxspeed']})
    lineL.append({"src":neii['src'],"trg":neii['trg'],"speed":neii['maxspeed'],"highway":neii['highway']})
    lineS.append(sh.geometry.LineString([(neii['loc']['coordinates'][0][0],neii['loc']['coordinates'][0][1]),(neii['loc']['coordinates'][1][0],neii['loc']['coordinates'][1][1])]))

pointL = pd.DataFrame(pointL)
pointL = pointL.groupby("id").head(1)
pointL.loc[:,"z"] = pointL['speed'].apply(lambda x: 0.1*x/(max(pointL['speed'])) )
lineL = pd.DataFrame(lineL)
colorL = ["firebrick","sienna","olivedrab","crimson","steelblue","tomato","palegoldenrod","darkgreen","limegreen","navy","darkcyan","darkorange","brown","lightcoral","blue","red","green","yellow","purple","black"]
colorI, _ = pd.factorize(lineL['highway'])
lineL = gpd.GeoDataFrame(lineL)
lineL.loc[:,"color"] = [colorL[int(i)] for i in colorI]
lineL.loc[:,"weight"] = lineL['speed']/max(lineL['speed'])*(BBox[3]-BBox[2])/60.
lineL.geometry = lineS
lineL.loc[:,"geometry"] = [x.buffer(y) for x,y in zip(lineL['geometry'],lineL['weight'])]
routeN = cascaded_union(lineL.geometry)
if False:
    lineL.plot(color=lineL['color'])
    plt.show()


G = nx.MultiGraph()
for i,p in pointL.iterrows():
    G.add_node(int(p['id']),pos=(p['x'],p['y']),x=p['x'],y=p['y']) 
for i,p in lineL.iterrows():
    G.add_edge(int(p['src']),int(p['trg']),color=p['color'],weight=p['weight'])
        
G.graph['crs'] = {'init': 'epsg:4326'}
G.graph['name'] = 'local streets'
edges = G.edges()
colors = [G[u][v][0]['color'] for u,v in edges]
weights = [5.*G[u][v][0]['weight'] for u,v in edges]
nx.draw(G,nx.get_node_attributes(G,'pos'),with_labels=False,node_size=1,edge_color=colors,width=weights)
plt.show()
##ox.plot_graph(G)
simp_G = ox.simplify_graph(G)
connected_G = max(nx.strongly_connected_component_subgraphs(simp_G), key=len)
graph_proj = ox.project_graph(connected_G)
G = nx.MultiDiGraph(name="local",crs=netP.crs)

geoS = "lc = 1e-1;\n"
sBox = [0.,0.,1.,1.]
for i,j,k in zip(range(4),[0,1,1,0],[1,1,0,0]):
    geoS += "Point(%d) = {%f,%f,%f,lc};\n" % (i,BBox[j%2],BBox[k%2+2],0)
for i,p in enumerate(BBox):
    geoS += "Line(%d) = {%d,%d};\n" % (i,i,(i+1)%4)
geoS += "Line Loop(1) = {0,1,2,3} ;\n"
geoS += "Plane Surface(1) = {1} ;\n"
##for i,p in pointL.iterrows():
for i,p in G.nodes(data=True):
    geoS += "Point(%d) = {%f,%f,%f,lc};\n" % (int(i),p['x'],p['y'],0.)#p['z'])
##for i,p in lineL.iterrows():
idL = 4
for u, v, key, p in G.edges(data=True,keys=True):
    geoS += "Line(%d) = {%d,%d};\n" % (idL,u,v)
    idL += 1
    
geoF = open(baseDir + "gis/roda/network.geo","w")
geoF.write(geoS)
geoF.close()

for neii in neiN:
    if not neii['highway'] == "motorway_link":
        continue
    nodeD.append({"highway":neii["highway"],"src":neii["src"],"trg":neii["trg"],"x":neii["loc"]['coordinates'][0][0],"y":neii["loc"]['coordinates'][0][1],"grp":poii['ref'],"id":poii['@id'].split("/")[1]})


confF = open(baseDir + "src/allink/dynAntenna.conf","r").read()
with open(baseDir + "src/allink/dynAntenna1.conf","w") as f:
    f.write(confF)
    for i,x in cent.iterrows():
        f.write("Rigid x(%.2f %.2f 0.0) a(0.00 0.00 1.00) c(%.2f 15.00 %.2f 2.2000) s{cyl}\n" %
              ((x['x']-BBox[0])/(BBox[1]-BBox[0])
               ,(x['y']-BBox[2])/(BBox[3]-BBox[2])
               ,x['area']/cent['area'].sum()*.5
               ,x['AreaInters']/cent['AreaInters'].sum()
              ) )

    

space = pd.read_csv(baseDir + "raw/roda/fem_space.dat",sep=" ",header=None)
Nl = int(np.sqrt(space.shape[0]))
x = space[0].values.reshape((Nl,Nl))*(BBox[1]-BBox[0]) + BBox[0]
y = space[1].values.reshape((Nl,Nl))*(BBox[3]-BBox[2]) + BBox[2]
z = space[2].values.reshape((Nl,Nl))
dz = np.gradient(z)
vx = -dz[0]
vy = -dz[1]
vz = np.zeros((Nl,Nl))
if False:
    plt.quiver(x,y,vx,vy,z)
    plt.plot(cent['x'],cent['y'],'o',color="r")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

vact = act
for i,a in act.iterrows():
    ix = int( (a['x']-BBox[0])/(BBox[1]-BBox[0])*Nl - 0.00000001)
    iy = int( (a['y']-BBox[2])/(BBox[3]-BBox[2])*Nl - 0.00000001)
    vact.loc[i,'x'] = a['x'] + vx[ix][iy]*.1
    vact.loc[i,'y'] = a['y'] + vy[ix][iy]*.1

plotDens(vact['x'].values,vact['y'].values)
x,y = vact['x'].values, vact['y'].values
vact.to_csv(baseDir + "gis/roda/gradient.csv",index=False)

#http://fenics.readthedocs.io/projects/dolfin/en/2017.2.0/demos/poisson/python/demo_poisson.py.html
# Create mesh and define function space
mesh = UnitSquareMesh(64, 64, "right/left")#,"left","crossed"
if False:
    plt.figure()
    plot(mesh, title="Unit interval")
    plt.show()

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["ghost_mode"] = "shared_facet"

V = FunctionSpace(mesh, "CG", 2)
V = FunctionSpace(mesh, "Lagrange", 1)
# Define Dirichlet boundary (x = 0 or x = 1)
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class Source(UserExpression):
    def eval(self, values, x):
        values[0] = 4.0*pi**4*sin(pi*x[0])*sin(pi*x[1])

def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)
bc = DirichletBC(V, u0, DirichletBoundary())
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

plot(u)
plt.show()

# Save solution in VTK format
fileS = File("poisson.pvd")
fileS << u



# Define normal component, mesh size and right-hand side
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2.0
n = FacetNormal(mesh)
f = Source(degree=2)

# Penalty parameter
alpha = Constant(8.0)


# Define bilinear form
a = inner(div(grad(u)), div(grad(v)))*dx \
  - inner(avg(div(grad(u))), jump(grad(v), n))*dS \
  - inner(jump(grad(u), n), avg(div(grad(v))))*dS \
  + alpha/h_avg*inner(jump(grad(u),n), jump(grad(v),n))*dS

# Define linear form
L = f*v*dx

# Solve variational problem
u = Function(V)
solve(a == L, u, bc)

# Plot solution
plot(u)
plt.show()







# Define function space
P2 = VectorElement('P', tetrahedron, 2)
P1 = FiniteElement('P', tetrahedron, 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)
 
# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
a = inner(grad(u), grad(v))*dx - p*div(v)*dx + div(u)*q*dx
L = dot(f, v)*dx
 
# Compute solution
w = Function(W)
solve(a == L, w, [bc1, bc0])








print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
