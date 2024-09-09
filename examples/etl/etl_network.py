#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph

def plog(text):
    print(text)

if False:
    plog('----------------------parse-all-files-----------------')
    projDir = baseDir + "src/"
    patterN = "\\.py$"
    fL = []
    for path,dirs,files in os.walk(projDir):
        for f in files:
            if re.search(patterN,f):
                if f == "etl_network.py":
                    continue
                if any([path.split("/")[-1] == x for x in ['py','old','img','junk','nodejs']]):
                    continue
                for line in open(path+"/"+f, 'r'):
                    if re.search("def ", line):
                        fL.append({"file":path+"/"+f,"func":line})

    fL = pd.DataFrame(fL)
    fL.loc[:,"func"] = fL['func'].apply(lambda x: x.split("def ")[1])
    #fL.loc[:,"arg"] = fL['func'].apply(lambda x: x.split("(")[1])
    fL.loc[:,"func"] = fL['func'].apply(lambda x: x.split("(")[0])
    fL.loc[:,"def"] = fL['file'].apply(lambda x: x.split("/")[-2])
    fL.loc[:,"name"] = fL['file'].apply(lambda x: x.split("/")[-1])
    fL.loc[:,"name"] = fL['name'].apply(lambda x: re.sub("\\.py","",x))
    fL = fL.groupby(["func","def"]).first().reset_index()
    fL = fL[~fL['func'].isin(["apply","write","close","__init__","main","f","name","load","fit","clampF","find","save"])]
    fL = fL[[not bool(re.search("^_",x)) for x in fL['func']]]
    
    linkS = []
    for i,g in fL.groupby('func'):
        for f in fL['file']:
            for line in open(f, 'r'):
                if re.search("def ", line):
                    continue
                if re.search("[\\. ]" + i + "\\(", line):
                    l = f.split("/")
                    linkS.append({"use":l[-1],"dir_use":l[-2]
                                  ,"func":i,"line":line
                                  ,"def":g['name'].values[0],"dir_def":g['def'].values[0]})
    linkS = pd.DataFrame(linkS)
    linkS.loc[:,"use"] = linkS['use'].apply(lambda x: re.sub("\\.py","",x))
    linkS.to_csv(baseDir + "raw/basics/pdoc_link.csv",index=False)
    
linkS = pd.read_csv(baseDir + "raw/basics/pdoc_link.csv")
del linkS['line']

nodeD = linkS.groupby(["use","dir_use"]).agg(len).reset_index()
nodeD = nodeD[['use','dir_use','func']]
nodeD.columns = ['name','group','size']
nodeD.loc[:,"idx"] = nodeD.index

linkS.loc[:,"source"] = pd.merge(linkS,nodeD,how="left",left_on="def",right_on="name")['idx']
linkS.loc[:,"target"] = pd.merge(linkS,nodeD,how="left",left_on="use",right_on="name")['idx']
linkS = linkS[linkS['source'] == linkS['source']]
linkS = linkS[linkS['target'] == linkS['target']]
linkS.loc[:,"source"] = linkS['source'].astype(int)
linkS.loc[:,"target"] = linkS['target'].astype(int)
linkS = linkS[linkS['source'] != linkS['target']]

linkG = linkS.groupby("func").agg(len).reset_index()
linkS.loc[:,"value"] = pd.merge(linkS,linkG,on="func",how="left")['source_y'].values
linkS.loc[linkS['value'] != linkS['value'],'value'] = 1.
linkS = linkS.sort_values("value")
linkS.loc[:,"name"] = linkS['func']

nodeL = nodeD[nodeD['size'] > 50]['name']
nodeD.loc[:,"size"] = np.log(nodeD['size']+1.) + 2.
# linkS = linkS[linkS['def'].isin(nodeL)]
# linkS = linkS[linkS['use'].isin(nodeL)]
linkS = linkS[linkS['value'] > 100.]
linkS.loc[:,"value"] = np.log(linkS['value']+1.) + 2.

print(linkS.shape)
linkJ = {}
linkJ['links'] = json.loads(linkS.to_json(orient='table',index=False))['data']
linkJ['nodes'] = json.loads(nodeD.to_json(orient='table',index=False))['data']
#json.dump(nodeJ,open(baseDir + "www/d3/data/func_node.json","w"))
json.dump(linkJ,open(baseDir + "www/d3/data/func_network.json","w"))
linkS.to_csv(baseDir + "raw/basics/pdoc_sourceCode.csv",index=False)
nodeD.to_csv(baseDir + "raw/basics/pdoc_sourceDef.csv",index=False)



print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
