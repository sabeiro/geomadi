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
import geomadi.calc_finiteDiff as c_f
import importlib

def plog(text):
    print(text)


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
