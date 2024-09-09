#!/usr/bin/env Rscript
setwd('~/lav/motion/')
source('src/R/graphEnv.R')


fs <- read.csv("raw/tac.csv")
ds <- ddply(fs,.(manufacturer),summarise,count=length(manufacturer))
ds <- ds[order(ds$count,decreasing = TRUE),]
nDisp = 10
nOther = sum(ds[nDisp:nrow(ds),"count"])
ds = ds[1:nDisp,]
ds[nDisp+1,] = c("other",nOther)
ds$count = as.integer(ds$count)
ds <- ds[order(ds$count,decreasing = FALSE),]
ds$manufacturer = factor(ds$manufacturer,levels=ds$manufacturer)
gLabel = c("manufacturer","count",paste("models counts"),"manufacturer","convexity")
p <- ggplot(ds) +
    geom_bar(aes(x=manufacturer,y=count,fill=manufacturer,group=1),stat="identity") + 
    coord_polar(theta="y") + 
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])

p





