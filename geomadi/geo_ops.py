"""
geo_ops:
geometrical operations
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import shapely as sh
import shapely.speedups
from shapely import geometry
from shapely.ops import cascaded_union
import shapely.speedups
from rtree import index
import geomadi.geo_octree as otree

shapely.speedups.enable()


def bbox(BBox):
    """return a polygon from a bounding box"""
    P = geometry.Polygon([[BBox[0][0], BBox[0][1]]
                             , [BBox[0][0], BBox[1][1]]
                             , [BBox[1][0], BBox[1][1]]
                             , [BBox[1][0], BBox[0][1]]
                             , [BBox[0][0], BBox[0][1]]
                          ])
    return P


def boxAround(xl, yl, BBox=[0.05, 0.05]):
    """a list of boxes around the coordinates"""
    P = [sh.geometry.Polygon([[x - BBox[0], y - BBox[1]], [x + BBox[0], y - BBox[1]], [x + BBox[0], y + BBox[1]],
                              [x - BBox[0], y + BBox[1]]]) for x, y in zip(xl, yl)]
    return P


def intersectionList(poly1, poly2):
    """return the intersection area between two lists of polygons"""
    idx = index.Index()
    for i, p in enumerate(poly2):
        idx.insert(i, p.bounds)
    sectL = []
    for p in poly1:
        merged_cells = cascaded_union([poly2[i] for i in idx.intersection(p.bounds)])
        sectL.append(p.intersection(merged_cells).area)
    return sectL


def intersectGeom(poly1, poly2, id1, id2, precDigit=10):
    """return the intersection ids between two lists of polygons"""
    gO = otree.h3tree()
    idx = index.Index()
    for i, p in enumerate(poly2):
        idx.insert(i, p.bounds)
    sectL = {}
    numL = []
    for i, p in zip(id1, poly1):
        merged_cells = cascaded_union([poly2[i] for i in idx.intersection(p.bounds)])
        pint = p.intersection(merged_cells)
        if isinstance(pint, sh.geometry.Point):
            l = [gO.encode(pint.x, pint.y, precision=precDigit)]
        else:
            l = [gO.encode(x.x, x.y, precision=precDigit) for x in pint]
        sectL[i] = l
        numL.append(len(l))
    print("%.2f points per polygon" % (np.mean(numL)))
    return sectL


def minDist(point1, poly2):
    """return the minimum distances between points and polygons"""
    p2 = cascaded_union(poly2)
    distL = [p1.distance(p2) for p1 in point1]
    return distL


def densPoint(pos, radius=0.03, isPlot=False):
    """densities around each point"""
    tree = sp.spatial.KDTree(np.array(pos))
    neighbors = tree.query_ball_tree(tree, radius, p=2.0)
    frequency = np.array(map(len, neighbors))
    frequency = np.array([len(x) for x in neighbors])
    density = frequency / radius ** 2
    return frequency


def densArea(x, y, density=2, isPlot=False):
    """calculate the area depending on a density threshold"""
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = sp.stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    cset = plt.contour(xx, yy, f, colors='k')
    plt.clf()
    line = sh.geometry.LineString(cset.allsegs[1][0])
    area = max(0.00000000000000000001, line.convex_hull.area)
    area = np.sqrt(area)
    count = cset.levels[1]
    if isPlot:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
        ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, f, colors='k')
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.title('2D Gaussian Kernel density estimation')
        plt.show()
        plt.figure(figsize=(8, 8))
        for j in range(len(cset.allsegs)):
            for ii, seg in enumerate(cset.allsegs[j]):
                plt.plot(seg[:, 0], seg[:, 1], '.-', label=f'Cluster{j}, level{ii}')
                plt.legend()
        plt.show()
        x1, y1 = line.xy
        plt.plot(x1, y1)
        plt.show()

    return count / area
