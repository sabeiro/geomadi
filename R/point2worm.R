#!/usr/bin/env Rscript
setwd('~/lav/motion/')
source('src/R/graphEnv.R')
## require(stats)
## library(grid)
## library('rjson')
## library('jsonlite')
## library('RJSONIO')
## library(RCurl)

spotL = read.csv("raw/mc/mc_whitespot2.csv",stringsAsFactor=F)
spotM <- as.matrix(spotL[,c('XCenter','YCenter')])

library(geosphere)
distM <- distm(spotM,fun=distHaversine)/1000.
head(distM)

cutDist = 3
chainId = 1
spotL[,'chainId'] = NA
totIdx =  1:nrow(spotM)
for(i in 1:nrow(spotM)){
    pNei = distM[i,] < cutDist
    if(is.na(spotL[i,'chainId']) ) {
        spotL[pNei,'chainId'] = chainId        
        chainId = chainId + 1
    } else {spotL[pNei,'chainId'] = spotL[i,'chainId']}
}
cutDist = 2
for(i in 1:nrow(spotM)){
    pNei = distM[i,]<cutDist
    spotL[pNei,'chainId'] = spotL[i,'chainId']
}
cutDist = 3
for(i in 1:nrow(spotM)){
    pNei = distM[i,]<cutDist
    spotL[pNei,'chainId'] = spotL[i,'chainId']
}
print(length(table(spotL$chainId)))
write.csv(spotL,'gis/mc/mc_whitespotCentroid.csv',sep="\t")

spotC = ddply(spotL,.(chainId),summarise,x=mean(XCenter),y=mean(YCenter),count=length(XCenter),guest_pot=median(gp_v2),cr=median(cr))#,Rail_Distance=median(Rail_Distance))
neiL = NULL
for(i in 1:nrow(spotC)){
    distMin = 1e12
    partId = NA
    for(j in 1:nrow(spotL)){
        dist = (spotL[j,'XCenter']-spotC[i,'x'])**2 + (spotL[j,'YCenter']-spotC[i,'y'])**2
        if(dist < distMin){
            distMin = dist
            partId = j
        }
    }
    print(partId)
    neiL <- rbind(neiL,data.frame(xNei=spotL[partId,'XCenter'],yNei=spotL[partId,'YCenter']))
}
spotC <- cbind(spotC,neiL)

write.csv(spotC,'gis/mc/mc_whitespotClust.csv')


## library(tripack)
## plot(voronoi.mosaic(runif(100), runif(100), duplicate="remove"))

library(raster)
x <- shapefile('file.shp')
crs(x)
x$area_sqkm <- area(x) / 1000000



nba <- read.csv("http://datasets.flowingdata.com/ppg2008.csv")
dist_m <- as.matrix(dist(nba[1:20, -1]))
dist_mi <- 1/dist_m # one over, as qgraph takes similarity matrices as input
library(qgraph)
jpeg('example_forcedraw.jpg', width=1000, height=1000, unit='px')
qgraph(dist_mi, layout='spring', vsize=3)
dev.off()



"coldiss" <- function(D, nc = 4, byrank = TRUE, diag = FALSE)
{
	require(gclus)

	if (max(D)>1) D <- D/max(D)

	if (byrank) {
		spe.color = dmat.color(1-D, cm.colors(nc))
	}
	else {
		spe.color = dmat.color(1-D, byrank=FALSE, cm.colors(nc))
	}

	spe.o = order.single(1-D)
	speo.color = spe.color[spe.o,spe.o]
	
	op = par(mfrow=c(1,2), pty="s")

	if (diag) {
		plotcolors(spe.color, rlabels=attributes(D)$Labels, 
			main="Dissimilarity Matrix", 
			dlabels=attributes(D)$Labels)
		plotcolors(speo.color, rlabels=attributes(D)$Labels[spe.o], 
			main="Ordered Dissimilarity Matrix", 
			dlabels=attributes(D)$Labels[spe.o])
	}
	else {
		plotcolors(spe.color, rlabels=attributes(D)$Labels, 
			main="Dissimilarity Matrix")
		plotcolors(speo.color, rlabels=attributes(D)$Labels[spe.o], 
			main="Ordered Dissimilarity Matrix")
	}

	par(op)
}

