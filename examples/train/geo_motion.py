"""
geo_motion:
equation of motion 
"""

import numpy as np
import pandas as pd
import albio.series_stat as s_s

class motion:
    """calculation on equation of motion"""
    def __init__(self,BoundBox=[5.866,47.2704,15.0377,55.0574]):
        """define specific geometrical coordinates"""
        self.BoundBox = BoundBox
        self.BCenter = [(BoundBox[2]-BoundBox[0])*.5,(BoundBox[3]-BoundBox[1])*.5]
        self.dH = (BoundBox[3] - BoundBox[1])
        self.dW = (BoundBox[2] - BoundBox[0])

    def motion(self,X,isSmooth=False,steps=5,isPlot=False):
        """compute the motion for a timespace array"""
        BCenter = self.BCenter
        def smoothFun(x,steps=10):
            return s_s.serSmooth(x,steps=steps)
            #return s_s.serRunAv(x,steps=steps)
        x1, x2 = X[:-1,1], X[1:,1]
        y1, y2 = X[:-1,2], X[1:,2]
        dt = [max(x,1) for x in X[1:,0] - X[:-1,0]]
        if isSmooth:
            dt = smoothFun(dt,steps=steps)
            x1 = smoothFun(x1,steps=steps)
            x2 = smoothFun(x2,steps=steps)
            y1 = smoothFun(y1,steps=steps)
            y2 = smoothFun(y2,steps=steps)
        dx = x2 - x1
        dy = y2 - y1
        sp = [np.sqrt(x**2 + y**2) for x,y in zip(dx,dy)]
        an = [np.arccos((x+y)/(abs(x)+abs(y))) for x,y in zip(dx,dy)]
        sp = [x/y for x,y in zip(sp,dt)]
        vp = [x2 - BCenter[0],y2 - BCenter[1]]
        vc = [x1 - BCenter[0],y1 - BCenter[1]]
        ch = 1*(vp[0]*vc[1] - vc[0]*vp[1]>0.)
        if isSmooth:
            sp = smoothFun(sp,steps=steps)
            an = smoothFun(an,steps=steps)
            ch = smoothFun(ch,steps=steps)
        XV = np.c_[dt,x1,y1,sp,an,ch]
        if isPlot:
            fig, ax = plt.subplots(2,1)
            ax[0].set_title("movement profile")
            ax[0].plot((dx-min(dx))/(max(dx)-min(dx)),label="dx")
            # plt.plot((x2-min(x2))/(max(x2)-min(x2)),label="x2")
            ax[0].plot((dy-min(dy))/(max(dy)-min(dy)),label="dy")
            # plt.plot((y2-min(y2))/(max(y2)-min(y2)),label="y2")
            ax[0].plot((dt-min(dt))/(max(dt)-min(dt)),label="dt")
            ax[0].plot((sp-min(sp))/(max(sp)-min(sp)),label="speed",linewidth=2)
            ax[1].plot((an-min(an))/(max(an)-min(an)),label="angle")
            ax[1].plot((ch-min(ch))/(max(ch)-min(ch)),label="chirality")
            ch1 = (vp[0]*vc[1] - vc[0]*vp[1])/(np.linalg.norm(vp)+np.linalg.norm(vc))
            ax[1].plot((ch1-min(ch1))/(max(ch1)-min(ch1)),label="cross prod")
            ax[0].legend()
            ax[1].legend()
            plt.show()
            plt.boxplot(sp)
            plt.show()
        return pd.DataFrame(XV,columns=['dt','x','y','speed','angle','chirality'])

    def chirality(self,X):
        """chirality calculation"""
        return ch

    def cluster(self,XV,threshold=0.05):
        """build clusters based on speed profile"""
        setL = XV['speed'] > threshold
        j1, c = XV['speed'].iloc[0], 0
        mot = XV.copy()
        mot.loc[:,"m"] = j1
        mot.loc[:,"c"] = 0
        for i,g in mot.iterrows():
            mot.loc[i,"c"] = c
            j = setL[i]
            if j != j1:
                j1 = j
                c += 1
        #np.split(a[:, 1], np.cumsum(np.unique(a[:, 0], return_counts=True)[1])[:-1])
        motM = mot.groupby("c").agg(np.mean)
        motS = mot.groupby("c").agg(np.std)
        motF = mot.groupby("c").first()
        motE = mot.groupby("c").last()
        motM.columns = ["m_"+x for x in motM.columns]
        motS.columns = ["s_"+x for x in motS.columns]
        motF = motF[['x','y','dt']]
        motF.columns = ["x1","y1","t1"]
        motE = motE[['x','y','dt']]
        motE.columns = ["x2","y2","t2"]
        motL = pd.concat([motM,motS,motF,motE],axis=1)
        motL.loc[:,'sr'] = (motL['s_x'] + motL['s_y'])*.5
        return motL, mot['c']

    def clustering(self,XV,clustD=1.0):
        """clustering with linkage"""
        XV = np.nan_to_num(XV,0.)
        XS = (XV - XV.min(axis=0))/(XV.max(axis=0)-XV.min(axis=0))
        Z = linkage(XS,'ward')
        clu = fcluster(Z,clustD,criterion='distance')
        c = len(set(clu))
        print("n cluster %d on %d points, ratio %.2f" % (c,XS.shape[0],c/XS.shape[0]))
        mot = pd.DataFrame(XV,columns=['t','x','y','s','a'])
        mot.loc[:,"c"] = clu
        motM = mot.groupby("c").agg(np.mean)
        motS = mot.groupby("c").agg(np.std)
        motF = mot.groupby("c").first()
        motE = mot.groupby("c").last()
        motL = pd.concat([motM,motS,motF[['x','y']],motE[['x','y']]],axis=1)
        motL.columns = ['t','x','y','s','a','m','st','sx','sy','ss','sa','sm','x1','y1','x2','y2']
        motL.loc[:,'sr'] = (motL['sx'] + motL['sy'])*.5
        return motL, mot['c']

    def geoJson(self,mot,tL):
        """create a geo json from density dataframe"""
        mot = mot.replace(float('nan'),1e-10)
        typD = [("int",int),("float",float),("string",str),("string",object)]
        dtyp = []
        for d in mot[tL].dtypes: dtyp.append([x[0] for x in typD if x[1] == d][0])
        cL = [(x,y) for x,y in zip(tL,dtyp) if x not in ['tx']]
        cL = [(re.sub("^x","lon_",x[0]),x[1]) for x in cL]
        cL = [(re.sub("^y","lat_",x[0]),x[1]) for x in cL]
        field = [{"name":x[0],"format":"","tableFieldIndex":y,"type":x[1]} for y,x in enumerate(cL)]
        rows = [list(x[tL]) for i,x in mot.iterrows()]
        geoD = {"fields":field,"rows":rows}
        return geoD

