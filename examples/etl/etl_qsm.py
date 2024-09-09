#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

cred = json.load(open(baseDir + "credenza/geomadi.json"))

if False:
    print('-----------------------tank-actRep------------------------')
    dayL = pd.read_csv(baseDir + "raw/tank/dateList.csv")
    dayL = [re.sub("-","/",x) for x in dayL['day']]
    distTh = 20.0
    fName = baseDir + "src/job/activity/" + "qsm.activity_report.tank_chirality_y20.json"
    qsm = json.load(open(fName,"r"))
    qsm['priority'] = 120
    qsm['version'] = "10.5.0"
    qsm['zone_mapping']['dominant_zone_mapping_fname'] = 'tank_'+str(int(distTh))+'_reweighted'
    qsm['zone_mapping']['zone_mapping_fname'] = 'tank_30_reweighted'
    qsm['date_list'] = dayL
    qsm['aggregation']['target_attr'] = ["dominant_zone","metrics.previous_trip_chirality"]
    qsm['attribute_extractions']['international']['active'] = False
    qsm['anonymization']['k_value'] = 0
    qsm['filtering']["neighbouring_trips_dist_sum"]['active'] = True
    qsm['filtering']["neighbouring_trips_dist_sum"]['min_in_km'] = float(distTh)
    qsm['filtering']["neighbouring_trips_dist_sum"]['max_in_km'] = 9999.0
    qsm['filtering']["neighbouring_trips_durs_sum"]['active'] = False
    qsm["multidates_aggregation"]['active'] = True
    qsm['time_mapping']["time_mapping_fname"] = "hours_de"
    #print(json.dumps(qsm,indent=4, sort_keys=True))
    fName = baseDir + "src/job/activity/" + "qsm.activity_report.tank_chirality_y" + str(int(distTh)) + ".json"
    json.dump(qsm,open(fName,"w"),indent=4, sort_keys=True)

if False:
    print('---------------------prepare-fhWestkuste----------------------')
    fL = os.listdir(baseDir + "src/job/fhWestkueste")
    fL = [x for x in fL if bool(re.search("json",x))]
    for f in fL:
        fName = baseDir + "src/job/fhWestkueste/"+ f
        qsm = json.load(open(fName,"r"))
        qsm['version'] = "10.0.0"
        qsm['priority'] = 40
        #qsm['zone_mapping'] = qsm.pop('dominant_zone_mapping')
        qsm['zone_mapping']['home_zone_mapping_fname'] = 'westkuste'
        qsm['zone_mapping']['dominant_zone_mapping_fname'] = 'westkuste'
        json.dump(qsm,open(fName,"w"),indent=4,sort_keys=True)

if False:
    print('-------------------prepare-pluto--------------------')
    dateL = {"kw10-18":[ "2018/03/06","2018/03/07","2018/03/08"],
             "kw11-18":[ "2018/03/13","2018/03/14","2018/03/15"],
             "kw12-18":[ "2018/03/20","2018/03/21","2018/03/22"],
             "kw13-18":[ "2018/03/27","2018/03/28","2018/03/29"],
             "kw14-18":[ "2018/04/03","2018/04/04","2018/04/05"],
             "kw15-18":[ "2018/04/10","2018/04/11","2018/04/12"],
             "kw16-18":[ "2018/04/17","2018/04/18","2018/04/19"],
             "kw17-18":[ "2018/04/24","2018/04/25","2018/04/26"],
             "kw02-19":[ "2019/01/08","2019/01/09","2019/01/10"],
             "kw03-19":[ "2019/01/15","2019/01/16","2019/01/17"],
             "kw04-19":[ "2019/01/22","2019/01/23","2019/01/24"],
             "kw05-19":[ "2019/01/29","2019/01/30","2019/01/31"],
             "kw06-19":[ "2019/02/05","2019/02/06","2019/02/07"],
             "kw07-19":[ "2019/02/12","2019/02/13","2019/02/14"],
             "kw08-19":[ "2019/02/19","2019/02/20","2019/02/21"],
             "kw09-19":[ "2019/02/26","2019/02/27","2019/02/28"]}

    projDir = baseDir + "src/job/tmp/pluto/"
    jobF = json.load(open(projDir + "template/qsm.activity_report.pluto_home_kw10-18.json"))
    jobF['version'] = "11.6.0"
    jobF['priority'] = 86
    for d in ["day","hour"]:
        if d == "day":
            jobF["time_mapping"]["time_mapping_fname"] = "day"
            jobF["aggregation"]["target_attr"] = ["dominant_zone","home_zone"]
        elif d == "hour":
            jobF["time_mapping"]["time_mapping_fname"] = "hours_de"
            jobF["aggregation"]["target_attr"] = ["dominant_zone"]
        for w in dateL.keys():
            jobF['date_list'] = dateL[w] 
            nameF = "qsm.activity_report.pluto_"+d+"_"+w+".json"
            json.dump(jobF,open(projDir+"/"+nameF,"w"),sort_keys=True,indent=4)

if False:
    print('-------------prepare-odm-via----------------')
    custD = "tank"
    job = json.load(open(baseDir + "src/job/odm_via/qsm.odm_extraction.odm_via.json"))
    month = "03"
    job['date_list'] = ["2019/02/%02d" % x for x in range(1,29)]
    job['date_list'] = ["2019/03/%02d" % x for x in range(1,32)]
    job['date_list'] = ["2019/04/%02d" % x for x in range(1,31)]
    job['odm_result_configs']['mode'] = 'daily'
    job['priority'] = 50
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    inLoc = []
    for i,g in poi.groupby(idField):
        c = g.iloc[0]
        name = str(c[idField])
        id_node = int(c['id_node_manual'])
        inLoc.append({"location_id":name,"node_list":[id_node]})

    job["odm_via_conf"]["input_locations"] = inLoc
    json.dump(job,open(baseDir + "src/job/odm_via/qsm.odm_extraction.odm_via_"+custD+"_19_"+month+".json","w"),sort_keys=True,indent=4)

if False:
    print('-------------odm-via-nodes-position---------------')
    custD = "tank"
    job = json.load(open(baseDir + "src/job/odm_via/qsm.odm_extraction.odm_via_tank_19_04.json"))
    inLoc = job["odm_via_conf"]["input_locations"]
    inLoc = pd.DataFrame({idField:[x['location_id'] for x in inLoc],"id_node":[x['node_list'][0] for x in inLoc]})
    client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'],username=cred['mongo']['user'],password=cred['mongo']['pass'])
    coll = client["tdg_infra_internal"]["nodes"]
    nodeL = coll.find({"node_id":{"$in":[int(x) for x in inLoc['id_node'].values]}})
    coord = [] 
    for n in nodeL:
        coord.append({"x":n["loc"]["coordinates"][0],
                      "y":n["loc"]["coordinates"][1],
                      "id_node":n["node_id"]})
    coorD = pd.DataFrame(coord)
    inLoc = inLoc.merge(coorD,on="id_node",how="left")
    inLoc.to_csv(baseDir + "gis/"+custD+"/node_job.csv",index=False)

if False:
    print('-------------prepare-odm-via-h----------------')
    custD = "tank"
    custD = "bast"
    job = json.load(open(baseDir + "src/job/odm_via/qsm.odm_extraction.odm_via.json"))
    job['date_list'] = ["2019/02/%02d" % x for x in range(1,29)]
    job['odm_result_configs']['mode'] = 'daily'
    job['odm_result_configs']['time_mapping'] = "hours_UTC_to_UTC_tdg"
    job["aggregation"]["target"] = ["location","time_origin"]
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    nodeL = pd.read_csv(baseDir + "gis/"+custD+"/nei_node.csv")    
    inLoc = []
    for i,g in nodeL.groupby(idField):
        name = str(i)
        id_node = [int(x) for x in g['src']]
        inLoc.append({"location_id":name,"node_list":id_node})

    job["odm_via_conf"]["input_locations"] = inLoc
    json.dump(job,open(baseDir + "src/job/odm_via/qsm.odm_extraction.odm_via_"+custD+"_h.json","w"),sort_keys=True,indent=4)
    
if False:
    print('-------------preparing-activity-statWeek------------------')
    projDir = baseDir + "src/job/activity/statWeek/"
    dateL = json.load(open(projDir + "template/datelist.json"))
    jobF = json.load(open(projDir + "template/qsm.activity_report.stat_week_template.json"))
    jobF['version'] = "11.6.0"
    jobF['priority'] = 86
    for d in ["day","hour"]:
        if d == "day":
            jobF["time_mapping"]["time_mapping_fname"] = "single"
        elif d == "hour":
            jobF["time_mapping"]["time_mapping_fname"] = "hours_de"
        for w in dateL.keys():
            jobF['date_list'] = dateL[w] 
            for k in ["mcc","age","gender","home_zone"]:
                jobF["aggregation"]["target_attr"] = ["dominant_zone",k]
                nameF = "qsm.activity_report.stat_week_"+k.split("_")[0]+"_"+w+".json"
                json.dump(jobF,open(projDir+"/"+d+"/"+nameF,"w"),sort_keys=True,indent=4)
                
if False:
    print('-------------preparing-odm-statWeek------------------')
    projDir = baseDir + "src/job/odm/"
    dateL = json.load(open(projDir + "template/datelist.json"))
    jobF = json.load(open(projDir + "template/qsm.odm_extraction.statWeek.json"))
    jobF['version'] = "11.6.0"
    jobF['priority'] = 86
    for d in ["day","hour"]:
        if d == "day":
            jobF["odm_result_configs"]["time_mapping"] = "single"
        elif d == "hour":
            jobF["odm_result_configs"]["time_mapping"] = "hours_UTC_to_UTC_tdg"
        for w in dateL.keys():
            jobF['date_list'] = dateL[w] 
            for k in ["mtc","zip5","ags8"]:
                jobF["odm_result_configs"]["origin_zone_mapping"] = k
                jobF["odm_result_configs"]["destination_zone_mapping"] = k
                nameF = "qsm.odm_extraction.statWeek_"+k.split("_")[0]+"_"+w+"_"+d[0]+".json"
                json.dump(jobF,open(projDir+"/"+"winter"+"/"+nameF,"w"),sort_keys=True,indent=4)

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
