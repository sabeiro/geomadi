"""
multi_thread:
prepare parallel queries and start a multi thread process
"""
import json, datetime, re
import time
import numpy as np
import pandas as pd
import scipy as sp
import pymongo
import shapely as sh
from shapely import geometry
from multiprocessing.dummy import Pool as ThreadPool

class multiTh:
    """class for multithread parallel query requests
    Args:
        poi: list of points of interest
        idField: single or multiple index to pivot

    """
    def __init__(self,poi,idField="id_poi",nList=400,nPool=40):
        """environment variables and iterators"""
        self.nChunk = max(1,int(poi.shape[0]/nList))
        self.idField = idField
        self.step = 0
        self.nPool = 40
        self.start_time = time.time()
        self.poi = poi.copy()
        print("prepared %d requests" % self.nChunk)    

    def setCollection(self,coll):
        """save the mongo collection"""
        self.coll = coll

    def chunk_poi(self):
        """prepare a data frame where each row has nChunk number of ids"""
        sortL = sorted([x % self.nChunk for x in list(range(self.poi.shape[0]))])
        self.poi.loc[:,"fold"] = sortL
        def clampF(x):
            return pd.Series({"id_list":[int(y) for y in x[self.idField]]})
        self.chunk = self.poi.groupby("fold").apply(clampF).reset_index()
        self.id_list = list(enumerate(self.chunk['id_list']))

    def getChunk(self):
        """return the chunked id list"""
        return self.chunk

    def printStatus(self,i):
        """print current status"""
        j = self.step/self.nChunk
        meanT = (time.time()-self.start_time)/float(self.step+1)
        eta = meanT*self.nChunk/3600./24. * (1.-j)
        print("complete %.2f %%, average time %.2f s, time left %.2f d" % (j*100.,meanT,eta),end='\r',flush=True)

    def query(self,id_list):
        """find the centroid from poi id"""
        neiN = self.coll.find({'tile_id':{"$in":id_list}})
        neiL = []
        for n in neiN:
            loc = geometry.Polygon(n['geom']['coordinates'][0]).centroid.xy
            neiL.append({'id_poi':n['tile_id'],"x":loc[0][0],"y":loc[1][0]})
        return pd.DataFrame(neiL)
        
    def routine(self,i,id_list):
        """run a query on the list of ids"""
        self.step = self.step + 1
        self.current = i
        self.printStatus(i)
        return self.query(id_list)

    def run_wrapper(self,args):
        """expand arguments from list"""
        return self.routine(*args)

    def check(self):
        """control the definition of essential objects"""
        if not hasattr(self,"id_list"):
            self.chunk_poi()
        if not hasattr(self,"coll"):
            print("please define a mongo collection/geometry first")
            return False
        return True

    def test(self):
        """test an iteration"""
        if not self.check(): return
        id_test = self.id_list[0]
        id_test = (0,id_test[1][:10])
        print(self.run_wrapper(id_test).head())
        print('|||routine check done|||')

    def run(self):
        """execute the pooling"""
        if not self.check(): return        
        pool = ThreadPool(self.nPool)
        results = pool.map(self.run_wrapper,self.id_list)
        pool.close()
        pool.join()
        tileL = pd.concat(results)
        return tileL

class find_coord(multiTh):
    """find the centroid coordinate for the given polygon id"""
    def query(self,id_list):
        """find the centroid from poi id"""
        neiN = self.coll.find({'tile_id':{"$in":id_list}})
        neiL = []
        for n in neiN:
            loc = geometry.Polygon(n['geom']['coordinates'][0]).centroid.xy
            neiL.append({'id_poi':n['tile_id'],"x":loc[0][0],"y":loc[1][0]})
        return pd.DataFrame(neiL)

class node_tile(multiTh):
    """match the node with the tile"""
    def query(self,id_list):
        """find the centroid from poi id"""
        neiN = self.coll.find({'tile_id':{"$in":id_list}})
        neiL = []
        for n in neiN:
            neiL.append({'id_poi':n['tile_id'],"id_node":n['node_id']})
        return pd.DataFrame(neiL)

class node_class(multiTh):
    """match the node and the class"""
    def query(self,id_list):
        """from node id to class and position"""
        neiN = self.coll.find({'src':{"$in":id_list}})
        neiL = []
        for n in neiN:
            neiL.append({'id_node':n['src'],"x":n['loc']['coordinates'][0][0],"y":n['loc']['coordinates'][0][1],'highway':n['highway']})
        return pd.DataFrame(neiL)

class motorway_distance(multiTh):
    """calculate distance from motorway"""
    def setCollection(self,line):
        """load motorway line string"""
        self.coll = line
    
    def query(self,id_list):
        """return positions of the closest point on the motorway"""
        neiL = []
        for i in id_list:
            g = self.poi[self.poi[self.idField] == i]
            p = sh.geometry.Point(g[['x','y']].values[0])
            neip = self.coll.interpolate(self.coll.project(p))
            #neiL.append({self.idField:i,"x_mot":neip.x,"y_mot":neip.y,"dist":p.distance(neip)})
            neiL.append({self.idField:i,"dist":p.distance(neip)})
        return pd.DataFrame(neiL)

