"""
geo_octree:
octree algebra for two dimensional geographic positioning
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import geomadi.geo_geohash as g_g
import shapely as sh
import albio.series_stat as s_s
import re

class quadtree:
    """algebra on a spatial octree index"""
    def __init__(self,BoundBox=[5.866,47.2704,15.0377,55.0574],max_precision=20):
        """define specific geometrical coordinates"""
        self.BoundBox = BoundBox
        self.BCenter = [(BoundBox[2]-BoundBox[0])*.5,(BoundBox[3]-BoundBox[1])*.5]
        self.dispD = {self.basis(i,j):(i,j) for i in [-1,0,1] for j in [-1,0,1]}
        self.max_precision = max_precision
        self.dH = (BoundBox[3] - BoundBox[1])
        self.dW = (BoundBox[2] - BoundBox[0])
        self.locMat = np.array([[1,3],[2,4]])
        boxMat = np.array([[0.,0.,0.,0.]
            ,[0,0,2,2],[2,0,4,2]
            ,[0,2,2,4],[2,2,4,4]])/4.
        self.boxMat = boxMat
        self.nx = 2
        self.ny = 2
        # self.gCenter = self.encode(self.BCenter[0],self.BCenter[1],precision=15)

    def getPrecision(self,x,y):
        """number of digits for a given lat lon"""
        lenx = len(str(x).split(".")[1])
        leny = len(str(y).split(".")[1])
        lenx = max(lenx,leny)
        BBox = self.BoundBox.copy()
        dH = (BBox[3] - BBox[1])
        dW = (BBox[2] - BBox[0])
        prec = 0
        for i in range(25):
            px = int(np.log(dH/2**i))
            py = int(np.log(dW/2**i))
            p = max(px,py)
            if -p == lenx:
                prec = i
        return prec

    def getDeviation(self,sr,default=15):
        """number of digits for a given standard deviation"""
        powr = np.log10(sr**2*.25)
        if abs(powr) == float('inf'): return default
        BBox = self.BoundBox.copy()
        dH = (BBox[3] - BBox[1])
        dW = (BBox[2] - BBox[0])
        prec = default
        for i in range(default):
            px = int(np.log(dH/2**i))
            py = int(np.log(dW/2**i))
            p = max(px,py)
            if p == int(powr)-3:
                prec = i
        return prec
    
    def getEdge(self,sr,default=15):
        """number of digits for a given standard deviation"""
        powr = np.log10(sr)
        if abs(powr) == float('inf'): return default
        BBox = self.BoundBox.copy()
        dH = (BBox[3] - BBox[1])
        dW = (BBox[2] - BBox[0])
        prec = default
        for i in range(default):
            px = int(np.log(dH/2**i))
            py = int(np.log(dW/2**i))
            p = max(px,py)
            if p == int(powr)-3:
                prec = i
        return prec

    def encode(self,x,y,precision=8):
        """encode two coordinates into a octree for a given precision"""
        BBox = self.BoundBox.copy()
        spaceId = ''
        dx, dy = 1./self.nx, 1./self.ny
        for i in range(precision):
            dW, dH = BBox[2] - BBox[0], BBox[3] - BBox[1]
            nx, ny = int((x-BBox[0])/dW*self.nx), int((y-BBox[1])/dH*self.ny)
            try: digit  = self.locMat[nx,ny]
            except:
                print("coordinate outside borders (%.2f,%.2f) (%d,%d)" % (x,y,nx,ny))
                return ''
            box = self.boxMat[digit]*[dW,dH,dW,dH]
            BBox = [box[0]+BBox[0],box[1]+BBox[1],box[2]+BBox[0],box[3]+BBox[1]]
            spaceId = spaceId + str(digit)
        return spaceId

    def decode(self,sId):
        """decode a octree into lon and lat and box sizes"""
        if sId == None: return (None,None,None,None)
        BBox = self.BoundBox.copy()
        spaceId = str(sId).split(".")[0]
        dx, dy = 1./self.nx, 1./self.ny
        for digit in spaceId:
            dW, dH = BBox[2] - BBox[0], BBox[3] - BBox[1]
            box = self.boxMat[int(digit)]*[dW,dH,dW,dH]
            BBox = [box[0]+BBox[0],box[1]+BBox[1],box[2]+BBox[0],box[3]+BBox[1]]
        x = BBox[0] + dW*dx
        y = BBox[1] + dH*dy
        return (x,y,dW*dx,dH*dy)

    def decodePoly(self,sId):
        """decode an octree and return it as a box edges"""
        x,y,dx,dy = self.decode(sId)
        return [[x-dx,y-dy],[x+dx,y-dy],[x+dx,y+dy],[x-dx,y+dy]]

    def decodeVolume(self,sId,z):
        """decode an octree and return it as a box edges"""
        x,y,dx,dy = self.decode(sId)
        return [[x-dx,y-dy],[x+dx,y-dy],[x+dx,y+dy],[x-dx,y+dy]]

    def basis(self,ix,iy):
        """project differences on an orthogonal basis"""
        return ix*0b001 + iy*0b011 + 0b100

    def disp(self,d):
        """project differences on an orthogonal basis"""
        return self.dispD[int(d)]

    def diff(self,g1,g2):
        """difference between octrees"""
        g1, g2 = str(g1), str(g2)
        n = min(len(g1),len(g2))
        g1, g2 = g1[:n], g2[:n]
        gd = ''
        for i,j in zip(g1,g2):
            i, j = int(i)-1, int(j)-1
            ix = (j  & 0b01) - (i  & 0b01)
            iy = (j >> 0b01) - (i >> 0b01)
            gd += str(basis(ix,iy))
        return gd

    def speed(self,tx1,tx2,firstOrder=False):
        """speed modulus from spacetime, first order"""
        dt = abs(tx2[0]-tx1[0])
        if dt <= 0: return 0.
        if firstOrder: dx = self.dispFirstOrder(tx2[1],tx1[1])
        else: dx = self.calcDisp2(tx2[1],tx1[1])
        return dx/dt

    def speedThreshold(self,x,prec=14):
        """compute the speed threshold"""
        digit = np.log10(x)
        return abs(digit-prec)

    def logSpeed(self,tx1,tx2):
        """compute log speed from space-time"""
        dt = abs(tx2[0] - tx1[0])
        if dt == 0: return 0.
        dx = abs(tx2[1] - tx1[1])
        if dx == 0: return 0.
        speed = np.log10(dx)/np.log10(dt)
        return speed

    def motion(self,XV,precision=15):
        """compute the motion for a timespace array"""
        XV.loc[:,"octree"] = [self.encode(x,y,precision=precision) for x,y in zip(XV['x'],XV['y'])]
        XV.loc[:,"n"] = 1.
        dens = XV[['octree','n','speed','angle','chirality']]
        dens = dens.groupby('octree').agg(np.nansum).reset_index()
        tL = [x for x in dens if not x in ['octree','n']]
        for i in tL: dens.loc[:,i] = dens[i]/dens['n']
        return dens

    def splitTraj(self,tx,threshold=0.37):
        """split a time-space sequence into dwelling and moving"""
        movL = []
        dweL = []
        for tx1, tx2 in zip(tx[:-1],tx[1:]):
            moving, speed = self.moving(tx1,tx2,threshod)
            if moving: movL.append((tx1[0],tx2[0],speed))
            else: dweL.append((tx1[0],tx2[0],speed))
        return movL, dweL
    
    def calcDisp(self,g1,g2):
        """return displacement between two octrees in lat lon"""
        BBox = self.BoundBox.copy()
        g1, g2 = str(g1), str(g2)
        pre1, pre2 = len(g1), len(g2)
        gForm = "%0" + str(max(pre1,pre2)) + "d"
        g1, g2 = gForm % int(g1), gForm % int(g2)
        dx, dy = 0, 0
        dH = (BBox[3] - BBox[1])*.5
        dW = (BBox[2] - BBox[0])*.5
        for i,j in zip(g1,g2):
            dH = dH*.5
            dW = dW*.5
            i, j = int(i)-1, int(j)-1
            ix = (j  & 0b01) - (i  & 0b01)
            iy = (j >> 0b01) - (i >> 0b01)
            dx = dx + dW*ix
            dy = dy + dH*iy
        return dx, dy, dW, dH

    def calcDisp2(self,g1,g2):
        """return displacement between two octrees in lat lon"""
        dx, dy, dW, dH = self.calcDisp(g1,g2)
        return np.sqrt((dx)**2 + (dy)**2)

    def dispFirstOrder(self,g1,g2):
        """displacement first relevant digit"""
        dg = self.diff(g1,g2)
        db = [(i,x) for i,x in enumerate(dg) if x != '4']
        if len(db) == 0: return 0.
        dx = self.disp(db[0][1])
        res = 1./2**db[0][0]
        dx = dx[0]*self.dW*res + dx[1]*self.dH*res
        return dx
    
    def chiralityTest(self,g1,g2):
        """calculate the spin of the trajectory"""
        x1, y1, dx1, dy1 = decode(g1)
        x2, y2, dx2, dy2 = decode(g2)
        vp = [x1 - self.BCenter[0],y1 - self.BCenter[1]]
        vc = [x2 - self.BCenter[0],y2 - self.BCenter[1]]
        crossP = vp[0]*vc[1] - vc[0]*vp[1]
        return 1*(crossP > 0.)

    def chirality(self,g1,g2):
        """calculate the spin of the trajectory"""
        pre1, pre2 = len(g1), len(g2)
        gCenter = gCenter[:pre1]
        gCenter = self.encode(self.BCenter,precision=pre1)
        vp = calcDisp(g1,gCenter)
        vc = calcDisp(g2,gCenter)
        crossP = vp[0]*vc[1] - vc[0]*vp[1]
        return 1*(crossP > 0.)
            
    def calcVector(self,g1,g2):
        """displacement vector between two octrees"""
        if (g1 == "") | (g2 == ""):
            return (None,None,None)
        chi = self.chirality(g1,g2)
        dx, dy, dW, dH = self.calcDisp(g1,g2)
        modulus = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy,dx)*180./np.pi
        return (modulus, angle, chi)

    def boundingBox(self,bound,precision=15):
        """return the octrees including the polygon"""
        g1 = str(self.encode(bound[0],bound[1],precision=precision))
        g2 = str(self.encode(bound[2],bound[3],precision=precision))
        j = 0
        for i in range(len(g1)):
            if g1[:i] != g2[:i]:
                j = i
                break
        if j <= 1: return 0
        return g1[:j-1]
            
    def neighbor(self,octree):
        """return the neighbors of a given octree"""
        

    def trajBBox(self,traj):
        """bounding box for the trajectory series"""
        if len(traj) == 0: return 0
        elif len(traj) == 1: return traj[0]
        t0 = traj[0]
        gmax, bbox = 0, 0
        bbox = max([abs(x-y) for x,y in zip(traj[1:],traj[:-1])])
        if bbox == 0: return traj[0]            
        prec = len(bbox)
        bbox = t0//10**prec
        return bbox

    def dwellingBox(self,dweL,max_iter=5,threshold=30,isGroup=False):
        """create a octree for every dwelling center and standard deviation"""
        dwe = []
        for i,g in dweL.iterrows():
            prec = self.getDeviation(g['sr'])
            x = self.encode(g['x'],g['y'],precision=prec)
            dwe.append(x)
        dwe = pd.DataFrame({"octree":dwe})
        dwe.loc[:,"n"] = 1
        if isGroup:
            dwe = densGroup(dwe,max_iter=max_iter,threshold=threshold)
        return dwe

    def motionPair(self,motL,precision=15,max_iter=5,threshold=30,isGroup=False):
        """create a octree for every dwelling center and standard deviation"""
        mot = []
        for i,g in motL.iterrows():
            g1 = self.encode(g['x1'],g['y1'],precision=precision)
            g2 = self.encode(g['x2'],g['y2'],precision=precision)
            mot.append({'origin':g1,'destination':g2})
        mot = pd.DataFrame(mot)
        mot.loc[:,"n"] = 1
        if isGroup:
            mot = densOriginDest(mot,max_iter=max_iter,threshold=threshold)
        return mot

    def intersectGeometry(self,c,isGeometry=False):
        """return the intersecting octress"""
        oct1 = []
        g = c.copy()
        p = g.geometry
        bbox = str(self.boundingBox(p.bounds))
        l = [bbox + i + j + k for i in ['1','2','3','4'] for j in ['1','2','3','4'] for k in ['1','2','3','4']]
        if False:
            rad = np.sqrt(p.area)
            prec = self.getEdge(rad)
            xc, yc = p.centroid.xy
            xp, yp = p.exterior.coords.xy
            l = p.exterior.coords
        for i,xy in enumerate(l):
            p1 = sh.geometry.Polygon(self.decodePoly(xy))
            if not p1.intersects(p): continue
            subarea = p1.intersection(p).area
            ratio = subarea/p.area
            g['id_sub'] = i
            g['ratio'] = ratio
            g['octree'] = xy
            if isGeometry: g['geometry'] = p1
            oct1.append(g.copy())
        oct1 = gpd.GeoDataFrame(oct1)
        if isGeometry:
            c1 = c.copy()
            c1['geometry'] = p
            oct1.append(c1)
        return oct1
    
    def geoDataframe(self,dens):
        """create a geodataframe from density dataframe"""
        polyL = dens.apply(lambda x: sh.geometry.Polygon(self.decodePoly(x['octree'])),axis=1)
        #poly = gpd.GeoDataFrame({"geometry":polyL,'n':dens['n'],'octree':dens['octree']})
        poly = gpd.GeoDataFrame(dens,geometry=polyL)
        poly.loc[:,"digit"] = poly['octree'].apply(lambda x: len(str(x)))
        poly.loc[:,"area"] = poly['geometry'].apply(lambda x: x.area)
        poly.loc[:,"dens"] = poly.apply(lambda x: x['n']/x['area'],axis=1)
        poly = poly.sort_values("digit")
        return poly

    def geoVolume(self,dens):
        """create a geodataframe from density dataframe"""
        polyL = dens.apply(lambda x: sh.geometry.Polygon(self.decodePoly(x['octree'])),axis=1)
        #poly = gpd.GeoDataFrame({"geometry":polyL,'n':dens['n'],'octree':dens['octree']})
        poly = gpd.GeoDataFrame(dens,geometry=polyL)
        poly.loc[:,"digit"] = poly['octree'].apply(lambda x: len(str(x)))
        poly.loc[:,"area"] = poly['geometry'].apply(lambda x: x.area)
        poly.loc[:,"dens"] = poly.apply(lambda x: x['n']/x['area'],axis=1)
        poly = poly.sort_values("digit")
        return poly

    def geoLines(self,mot,isCoord=False):
        """create a geodataframe from density dataframe"""
        if not isCoord:
            p1 = dens.apply(lambda x: self.decode(x['origin']),axis=1)
            p2 = dens.apply(lambda x: self.decode(x['destination']),axis=1)
        else:
            p1 = dens.apply(lambda x: [x['x1'],x['y1']],axis=1)
            p2 = dens.apply(lambda x: [x['x2'],x['y2']],axis=1)
        p1 = p1.apply(lambda x: sh.geometry.Point(x[0],x[1]))
        p2 = p2.apply(lambda x: sh.geometry.Point(x[0],x[1]))
        l = [sh.geometry.LineString([x,y]) for x,y in zip(p1,p2)]
        poly = gpd.GeoDataFrame(dens,geometry=l)
        poly.loc[:,"digit"] = poly.apply(lambda x: len(str(x['origin'])) + len(str(x['destination'])),axis=1)
        poly.loc[:,"length"] = poly['geometry'].apply(lambda x: max(1e-10,x.length) )
        poly.loc[:,"dens"] = poly.apply(lambda x: x['n']/x['length'],axis=1)
        poly = poly.sort_values("digit")
        return poly

    def geoTraj(self,traj,isCoord=False):
        """create a geo json from density dataframe"""
        typD = [("int",int),("float",float),("string",str),("string",object)]
        dtyp = []
        for d in dens.dtypes: dtyp.append([x[0] for x in typD if x[1] == d][0])
        tL = [(x,y) for x,y in zip(traj.columns,dtyp) if x not in ['tx']]
        geoD = {}
        field = [{"name":x[0],"format":"","tableFieldIndex":y,"type":x[1]} for y,x in enumerate(tL)]
        field.append({"name":'start_datetime',"format":'YYYY-M-D H:m:s',"tableFieldIndex":len(tL)+1,"type":'timestamp'})
        field.append({"name":'end_datetime',"format":'YYYY-M-D H:m:s',"tableFieldIndex":len(tL)+1,"type":'timestamp'})
    
    def geoVector(self,dens):
        """create a geodataframe from density dataframe"""
        polyL = dens.apply(lambda x: sh.geometry.Polygon(self.decodePoly(x['octree'])),axis=1)
        #poly = gpd.GeoDataFrame({"geometry":polyL,'n':dens['n'],'octree':dens['octree']})
        poly = gpd.GeoDataFrame(dens,geometry=polyL)
        poly.loc[:,"digit"] = poly['octree'].apply(lambda x: len(str(x)))
        poly.loc[:,"area"] = poly['geometry'].apply(lambda x: x.area)
        poly.loc[:,"dens"] = poly.apply(lambda x: x['n']/x['area'],axis=1)
        poly = poly.sort_values("digit")
        return poly
    
class octree(quadtree):
    """algebra on a spatial octree index"""
    def __init__(self,BoundBox=[5.866,47.2704,15.0377,55.0574],max_precision=20,padding=0.0):
        """define specific geometrical coordinates"""
        self.BoundBox = BoundBox
        self.BCenter = [(BoundBox[2]-BoundBox[0])*.5,(BoundBox[3]-BoundBox[1])*.5]
        self.dispD = {self.basis(i,j):(i,j) for i in [-1,0,1] for j in [-1,0,1]}
        self.max_precision = max_precision
        self.dH = (BoundBox[3] - BoundBox[1])
        self.dW = (BoundBox[2] - BoundBox[0])
        self.locMat = np.array([[1,5],[2,6],[3,7],[4,8]])
        self.padding = padding
        boxMat = np.array([[0.,0.,0.,0.]
                           ,[0,0,2,4],[2,0,4,4],[4,0,6,4],[6,0,8,4]
                           ,[0,4,2,8],[2,4,4,8],[4,4,6,8],[6,4,8,8]])/8.
        self.boxMat = boxMat
        self.nx = 4
        self.ny = 2
        # self.gCenter = self.encode(self.BCenter[0],self.BCenter[1],precision=15)

class novtree(quadtree):
    """algebra on a spatial novtree index"""
    def __init__(self,BoundBox=[5.866,47.2704,15.0377,55.0574],max_precision=20,padding=0.1):
        """define specific geometrical coordinates"""
        self.BoundBox = BoundBox
        self.BCenter = [(BoundBox[2]-BoundBox[0])*.5,(BoundBox[3]-BoundBox[1])*.5]
        self.dispD = {self.basis(i,j):(i,j) for i in [-1,0,1] for j in [-1,0,1]}
        self.max_precision = max_precision
        self.dH = (BoundBox[3] - BoundBox[1])
        self.dW = (BoundBox[2] - BoundBox[0])
        self.locMat = np.array([[1,6,3],[5,9,8],[2,7,4]])
        boxMat = np.array([[0.,0.,0.,0.]
            ,[0,0,2,2],[4,0,6,2],[0,4,4,6],[4,4,6,6]
            ,[2,0,4,2],[0,2,2,4],[4,2,6,4],[2,4,4,6]
            ,[2,2,4,4]])/6.
        self.boxMat = boxMat
        self.padding = padding
        self.nx = 3
        self.ny = 3
        #self.gCenter = self.encode(self.BCenter[0],self.BCenter[1],precision=15)

    def boundingBox(self,bound,precision=15):
        """return the octrees including the polygon"""
        g1 = str(self.encode(bound[0],bound[1],precision=precision))
        g2 = str(self.encode(bound[2],bound[3],precision=precision))
        j = 0
        for i in range(len(g1)):
            if g1[:i] != g2[:i]:
                j = i
                break
        if j <= 1: return 0
        print("%s %s %s" % (g1,g2,g1[:j-1]))
        return int(g1[:j-1])

class hashtree(quadtree):
    """index based on geohash"""

    def encode(self,x,y,precision=8):
        """encode a geohash calling the original library"""
        return g_g.encode(x,y,precision=precision)

    def decode(self,sId):
        """decode a geohash calling the original library"""
        return g_g.decode(sId)

    def decodePoly(self,sId):
        """decode a geohash calling the original library"""
        box = g_g.bbox(sId)
        return [[box['s'],box['w']],[box['s'],box['e']],[box['n'],box['e']],[box['n'],box['w']]]
        # return [[x-dx,y-dy],[x+dx,y-dy],[x+dx,y+dy],[x-dx,y+dy]]

def densGroup(dens,max_iter=5,threshold=30):
    """remove a digit from octree and group the coarse octree together until the threshold density is met"""
    dens.loc[:,"octree"] = dens['octree'].astype(str)
    dens = dens.groupby('octree').agg(np.sum).reset_index()
    for i in range(max_iter):
        print("grouping iter %d entries %d" % (i,dens.shape[0]))
        setL = dens['n'] < threshold
        if sum(setL) == 0: break
        dens.loc[setL,"octree"] = dens.loc[setL,'octree'].apply(lambda x: x[:-1])
        dens = dens.groupby('octree').agg(np.sum).reset_index()
    dens = dens[dens['octree'] != '']
    dens.loc[:,"octree"] = dens['octree'].astype(int)
    return dens

def densGroupAv(dens,max_iter=5,threshold=30):
    """remove a digit from octree and group the coarse octree together until the threshold density is met"""
    dens.loc[:,"octree"] = dens['octree'].astype(str)
    dens = dens.groupby('octree').agg(np.sum).reset_index()
    tL = [x for x in dens if not x in ['octree','n']]
    for i in range(max_iter):
        print("grouping iter %d entries %d" % (i,dens.shape[0]))
        setL = dens['n'] < threshold
        if sum(setL) == 0: break
        dens.loc[setL,"octree"] = dens.loc[setL,'octree'].apply(lambda x: x[:-1])
        X = dens.loc[:,tL].values
        n = dens.loc[:,"n"].values
        tX = np.multiply(X,n[:,np.newaxis])
        dens.loc[:,tL] = tX
        dens = dens.groupby('octree').agg(np.sum).reset_index()
        for i in tL: dens.loc[:,i] = dens[i]/dens['n']
        dens = dens[dens['octree'] != '']
        dens.loc[:,"octree"] = dens['octree'].astype(int)
    if False:
        wm = lambda x: (x * dens.loc[x.index,"n"]).sum() / dens.loc[x.index,"n"].sum()
        wm.__name__ = 'wa'
        g = dens.groupby('octree').agg({'speed':wm,'n':'sum'})
        g.columns = g.columns.map('_'.join)
    return dens

def densOriginDest(dens,max_iter=5,threshold=30,isAverage=False):
    """remove a digit from octree and group the coarse octree together until the threshold density is met"""
    dens.loc[:,"origin"] = dens['origin'].astype(str)
    dens.loc[:,"destination"] = dens['destination'].astype(str)
    dens = dens.groupby(['origin','destination']).agg(np.sum).reset_index()
    tL = [x for x in dens if not x in ['origin','destination','n']]
    for i in range(max_iter):
        print("grouping iter %d entries %d" % (i,dens.shape[0]))
        setL = dens['n'] < threshold
        if sum(setL) == 0: break
        dens.loc[setL,"destination"] = dens.loc[setL,'destination'].apply(lambda x: x[:-1])
        X = dens.loc[:,tL].values
        n = dens.loc[:,"n"].values
        tX = np.multiply(X,n[:,np.newaxis])
        dens.loc[:,tL] = tX
        dens = dens.groupby(['origin','destination']).agg(np.sum).reset_index()
        for i in tL: dens.loc[:,i] = dens[i]/dens['n']
        setL = dens['n'] < threshold
        if sum(setL) == 0: break
        X = dens.loc[:,tL].values
        n = dens.loc[:,"n"].values
        tX = np.multiply(X,n[:,np.newaxis])
        dens.loc[:,tL] = tX
        dens.loc[setL,"origin"] = dens.loc[setL,'origin'].apply(lambda x: x[:-1])
        dens = dens.groupby(['origin','destination']).agg(np.sum).reset_index()
        for i in tL: dens.loc[:,i] = dens[i]/dens['n']
    dens = dens[dens['origin'] != '']
    dens = dens[dens['destination'] != '']
    dens.loc[:,"origin"] = dens['origin'].astype(int)
    dens.loc[:,"destination"] = dens['destination'].astype(int)
    if isAverage:
        tL = [x for x in dens if not x in ['origin','destination','n']]
        for i in tL: dens.loc[:,i] = dens[i]/dens['n']
    return dens

def mergeSum(df1,df2,cL=["octree"]):
    """merge and weighted sum two dataframes"""
    df = df1.merge(df2,on=cL,how="outer")
    df = df.replace(float('nan'),0.)
    df.loc[:,"n"] = df.loc[:,"n_x"].values + df.loc[:,"n_y"].values
    tLx = [x for x in df.columns if bool(re.search("_x",x))]
    tLy = [x for x in df.columns if bool(re.search("_y",x))]
    tLx = [x for x in tLx if not bool(re.search("n_",x))]
    tLy = [x for x in tLy if not bool(re.search("n_",x))]
    tX = np.multiply(df[tLx].values,df['n_x'].values[:,np.newaxis])
    tY = np.multiply(df[tLy].values,df['n_y'].values[:,np.newaxis])
    norm = 1./(df['n'].values)
    df.loc[:,tLx] = np.multiply(tX+tY,norm[:,np.newaxis])
    for i in tLy + ['n_x','n_y']: del df[i]
    for i in tLx: df.rename(columns={i:i.split("_")[0]},inplace=True)
    return df
    
if False:
    #library testing
    print("x | y | dx | dy")
    import geomadi.geo_octree as g_o
    import importlib
    importlib.reload(g_o)
    gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574],padding=0.1)
    g2 = gO.encode(14.989551,48.218262,10)
    print(gO.decode(gO.encode(14.989551,48.218262,10)))
    print("de center  %s" % (gO.encode(10.28826401,51.13341344,15)) )

    
if False:
    #choose algebra base
    l = np.array([1,2,4,8])
    comb = [i-j for i in l for j in l]
    base, count = np.unique(comb,return_counts=True)
    print(base,count)
    print(len(count)/sum(count))
    print(sum(l))
    plt.bar(range(len(comb)),sorted(comb))
    plt.show()
    l2 = [-1,0,1]
    comb2 = [basis(i,j) for i in l2 for j in l2]
    plt.bar(range(len(comb2)),sorted(comb2))
    plt.show()


