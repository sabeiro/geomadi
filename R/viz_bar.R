#!/usr/bin/env Rscript
setwd('~/lav/motion/')
source('src/R/graphEnv.R')
require(stats)
library(grid)

mist <- read.csv("raw/tank/single_poi.csv")
mist$sani = mist$sani*sum(bist$bon,na.rm=TRUE)/sum(mist$sani,na.rm=TRUE)
mist$avg = mist$bon#apply(mist[,c("sani","bon")],1,function(x) max(x,na.rm=TRUE))
directions <- mist$day
directions.factor <- factor(directions)
mist['day'] = as.numeric(directions.factor)
mist['hour'] = lapply(mist['time'], function(x) substring(x,12,13))
mist['hour'] = as.numeric(mist$hour)
mist = mist[mist['hour'] > 3,]
cist = ddply(mist,.(id_poi),summarize,avg=sum(avg,na.rm=TRUE),dist_raw=sum(dist_raw,na.rm=TRUE))
cist['diff'] = cist['dist_raw']/cist['avg']
cist = cist[cist['diff'] > 0.5,]
cist = cist[cist['diff'] < 2,]
idL <- apply(mist['id_poi'], 1, function(r) any(r %in% cist['id_poi']))
mist = mist[mist$id_poi %in% as.vector(cist$id_poi),]
bist = ddply(mist,.(day,hour),summarize,avg=sum(avg,na.rm=TRUE),dist_raw=sum(dist_raw,na.rm=TRUE))

bist %>% head
mist %>% head

for(i in unique(mist$id_poi)){
    bist = mist[mist['id_poi'] == i,]
    melted = melt(bist[,c("day","hour","avg","dist_raw")],id.vars=c("day","hour"))
    melted$day = as.factor(melted$day)
    levels(melted$variable) <- c("visits","activities")
    gLabel = c("hour","count",paste("vistors vs activities"),"thursday #","source")
    p <- ggplot(melted,aes(x=hour,y=value,color=day,group=day)) +
        geom_line(alpha=0.7,size=2) +
        scale_color_manual(values=gCol1) +
        facet_grid(~variable) + 
        labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[5],color=gLabel[4],linetype=gLabel[5])
    print(p)
    print(i)
    line <- readline()
}
     


bist %>% head
melted %>% head
melted %>% str
bist %>% str


barW = 0.10

gLabel = c("thursday","count",paste("vistors vs activities"),"hour","source")
p <- ggplot(bist,aes(x=day,color=hour)) +
    geom_bar(aes(x=day-0.25,y=avg,group=1,fill="visitor"),width=barW,stat="identity",position="stack",alpha=0.5) +
    geom_bar(aes(x=day-0.10,y=ref,group=1,fill="visitors smooth"),width=barW,stat="identity",position="stack",alpha=0.5) +
    geom_bar(aes(x=day+0.10,y=dist_20,group=1,fill="activities smooth"),width=barW,stat="identity",position="stack",alpha=0.5) +
    geom_bar(aes(x=day+0.25,y=dist_raw,group=1,fill="activities"),width=barW,stat="identity",position="stack",alpha=0.5) +
    scale_color_gradient(low="darkblue",high="darkgreen") +
    scale_fill_manual(values=gCol1[c(1,3,5,7)]) +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[5],color=gLabel[4],linetype=gLabel[5])
p


