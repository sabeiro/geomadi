"""
geo_ops:
geometrical operations
"""
import json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.geo_octree as g_o
from shapely import geometry

def bbox(BBox):
    """return a polygon from a bounding box"""
    P = geometry.Polygon([[BBox[0][0],BBox[0][1]]
                          ,[BBox[0][0],BBox[1][1]]
                          ,[BBox[1][0],BBox[1][1]]
                          ,[BBox[1][0],BBox[0][1]]
                          ,[BBox[0][0],BBox[0][1]]
    ])
    return P
