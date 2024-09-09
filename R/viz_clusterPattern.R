#!/usr/bin/env Rscript
setwd('~/lav/motion/')
source('src/R/graphEnv.R')
library(gridExtra)

##------------------activities----------------------
fs <- read.csv("out/activity_class_corr.csv")
fs$X = seq(nrow(fs))
fs = fs[fs$z_corr >1,]
fs[is.na(fs)] = 0
classL = unique(fs$c_dist)
nBin = length(classL)
rs <- read.csv("out/activity_range.csv")
rs[,"c_source"] = seq(1,nrow(rs))
rs[,"c_type"] = seq(1,nrow(rs))
##------------------ranges-----------------------------
es <- data.frame(variable=seq(6,23))
es = as.data.frame(sapply(es,rep.int,times=nBin))
es[,"col1"] = as.factor(rep(seq(nBin)-2,each=(24-6)))
es[,"c_inter"] = rs[as.integer(es$col1),"c_inter"]
es[,"c_slope"] = rs[as.integer(es$col1),"c_slope"]*seq(7,24)
es[,"c_convex"] = rs[as.integer(es$col1),"c_convex"]*seq(7,24)*seq(7,24)
esM <- ddply(es,.(col1),summarise,c_slope=mean(c_slope),c_convex=mean(c_convex))
es[,"c_slope"] = es[,"c_slope"] - esM[as.integer(es$col1),"c_slope"] + 20
es[,"c_convex"] = es[,"c_convex"] - esM[as.integer(es$col1),"c_convex"] + 20
es[,"c_std"] = rs[as.integer(es$col1),"c_std"]*seq(7,24)*.27
es[,"c_median"] = rs[as.integer(es$col1),"c_median"]
es[es$variable > es$c_median,"c_median"] = NA
es[!is.na(es$c_median),"c_median"] = 20
es[c(rep(FALSE,18),!is.na(es[1:(nrow(es)-18),"c_median"])),"c_median"] = NA
es[,"c_dist"] = NA
es[,"c_sum"] = NA
es[,"c_tech"] = NA
es[,"c_type"] = NA
es[,"c_max"] = NA
es[,"c_trend"] = NA
es[,"c_trend2"] = NA
es[,"c_source"] = NA
##----------------------------benchmark---------------------------------
fs <- read.csv("out/tank_visit_max.csv")
gs <- read.csv("out/activity_shape.csv")
hL = colnames(fs)[colnames(fs) %>% grepl("^X.",.)]#[seq(6,23)]
labL = c("distance","intercept","slope","convexity","trend","trend2","maximum","std","sum","median","#source","type")
gs[is.na(gs)] = 0
gsM <- ddply(melt(gs[,hL]),.(variable),summarise,value=mean(value))
gsM$variable = gsM$variable %>% gsub("^X","",.) %>% substring(.,12,13) %>% as.numeric(.)
gsM <- ddply(gsM,.(variable),summarise,value=sum(value,na.rm=T))
#gsM$variable = as.POSIXct(gsM$variable,format="%Y.%m.%dT%H.%M",tz="CET")
gsM = sapply(gsM,rep.int,times=nBin)
gsM = as.data.frame(gsM)
gsM$col1 = as.factor(rep(classL,each=(24-6)))
gsM1 <- ddply(melt(gs[,c("c_type",hL)],id.vars="c_type"),.(variable,c_type),summarise,value=mean(value))
gsM1$variable = gsM1$variable %>% gsub("^X","",.) %>% substring(.,12,13) %>% as.numeric(.)
colnames(gsM1) <- c("variable","col1","value")
gsM1 <- ddply(gsM1,.(variable,col1),summarise,value=sum(value,na.rm=T))
gsM1$col1 = as.factor(gsM1$col1)

##-------------------------plot-single-poi------------------------------
if(FALSE){
    fs <- read.csv("out//activity_shape.csv")
    hL = colnames(fs)[colnames(fs) %>% grepl("^X2017",.)]
    ps = fs[fs$id_poi == 1001,hL]
    melted <- melt(t(ps))#,id.vars=c("clust"))
    melted = merge(melted,fs[,c("index","z_corr","z_reg")],by.x="Var2",by.y="index")
    melted$Var1 = melted$Var1 %>% gsub("^X","",.)
    melt1$Var1 = melt1$Var1 %>% gsub("^X","",.)
    melted$col1 = melted$z_reg
    melt1 = melt(t())
    melt1 = sapply(melt1,rep.int,times=nBin)
    melt1 = as.data.frame(melt1)
    melt1$col1 = rep(seq(nBin)-2,each=(24-6))
    melt1$Var1 = melt1$Var1 + 5
    gLabel = c("hour","activities",paste("score by regression"),"cluster","sum")
    p <- ggplot() +
        geom_line(data=melted,aes(group=Var2,x=Var1,y=value,color=col1),alpha=.3,size=1.1) +
        geom_line(data=melt1,aes(x=Var1,y=value,group=col1),size=4,alpha=0.7) +
        theme(legend.position="bottom",text = element_text(size = 14)) +
        facet_wrap(~col1,ncol=2) + 
        guides(fill=guide_legend(keywidth=rel(1.3),keyheight=rel(1.3))) +
        ##    scale_color_manual(values=gCol1) +
        labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
    p
}
##------------------------plot-clustering---------------------------
gs <- read.csv('raw/tank/out/tank_activity_20.csv')
lL = colnames(gs)[grepl("^c_.",colnames(gs))]
melted <- melt(t(gs[,hL]))#,id.vars=c("clust"))
melted$Var1 = melted$Var1 %>% gsub("^X","",.) %>% substring(.,12,13) %>% as.numeric(.)
melted = ddply(melted,.(Var1,Var2),summarise,value=sum(value))
melted = merge(melted,gs[,c("index",lL,"y_corr")],by.x="Var2",by.y="index")
melted %>% head
polyP = list()
for(i in seq(length(lL))){
    melted$col1 = as.factor(melted[,lL[i]])
    melted$col2 = as.factor(melted$c_sum)
    meltD = ddply(melted,.(Var1,col1),summarise,value=sum(value,na.rm=T)/length(value))
    meltD$Var2 = meltD$col1
    meltD$value = meltD$value + 30
    classL = as.character(unique(melted$col1))
    meltN = ddply(melted,.(col1),summarise,value=round(length(value)/length(hL)),corr=mean(y_corr))
    meltN$corr = round(meltN$corr,2)
    gLabel = c("hour","activities",paste("shape by",labL[i]),"shape","sum")
    esT <- es[,c("variable","col1",lL[i])]
    colnames(esT) <- c("Var1","col1","value")
    esT$col1 <- as.factor(esT$col1)
    esT$value <- as.numeric(esT$value)
    gsT = gsM
    if(lL[i] == "c_type"){gsT = gsM1}
    gsT = gsT[gsT$col1 %in% classL,]
    esT = esT[esT$col1 %in% classL,]
    labV = unique(rs[,lL[i]])
    labV = labV[1:length(classL)]
    if(is.numeric(labV)){labV = round(labV,1)}
    ## faceN = setNames(labV,classL)
    flabeller <- function(variable,value){return(labV[value])}
    p <- ggplot(melted,aes(x=Var1,y=value,color=col1)) +
        geom_line(aes(linetype=col2,group=Var2),alpha=.2,size=0.6) +
        geom_line(data=gsT,aes(x=variable),color="black",linetype=1,size=3,alpha=0.5) +
        geom_line(data=meltD,aes(group=col1),size=4,alpha=0.7) +
        geom_line(data=esT,linetype=1,size=2,alpha=0.3,color="red") +
        geom_text(data=meltN,aes(x=15,y=05,group=1,label=paste("N",value)),size=6,color="black") + 
        geom_text(data=meltN,aes(x=15,y=25,group=1,label=paste("r",corr)),size=6,color="black") + 
        theme(legend.position="bottom",text=element_text(size = 14)) +
        facet_grid(~col1) + #,labeller=labeller(col1=flabeller) ) +
        coord_cartesian(ylim = c(0.,200.)) + 
        guides(fill=guide_legend(keywidth=rel(1.3),keyheight=rel(1.3))) +
                                        #    scale_color_manual(values=gCol1) +
        labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
    p
    polyP[[i]] = p
    ##ggsave(file=paste("fig/shapeCluster_",lL[i],".jpg"), plot=p, width=2*gWidth, height=2*gHeight)
}

polyL = list(2,3,4,5)
lay <- rbind(c(1,2),c(3,4))
gl = lapply(polyL,function(p,i){polyP[[p]]})
ga <- grid.arrange(grobs=gl,ncol=2,top=textGrob("shape clustering",gp=gpar(fontsize=20,font=3)))
ggsave(file=paste("fig/shapeCluster_","1",".jpg",sep=""), plot=ga, width=2*gWidth, height=2*gHeight)
ga

polyL = list(1,7,10,11)
lay <- rbind(c(1,2),c(3,4))
gl = lapply(polyL,function(p,i){polyP[[p]]})
ga <- grid.arrange(grobs=gl,ncol=2,top=textGrob("shape clustering",gp=gpar(fontsize=20,font=3)))
ggsave(file=paste("fig/shapeCluster_","2",".jpg",sep=""), plot=ga, width=2*gWidth, height=2*gHeight)
ga

polyL = list(9,12)
lay <- rbind(c(1,2))
gl = lapply(polyL,function(p,i){polyP[[p]]})
ga <- grid.arrange(grobs=gl,ncol=2,top=textGrob("shape clustering",gp=gpar(fontsize=20,font=3)))
ggsave(file=paste("fig/shapeCluster_","3",".jpg",sep=""), plot=ga, width=2*gWidth, height=2*gHeight)
ga

