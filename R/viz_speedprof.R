#!/usr/bin/env Rscript
setwd('~/lav/motion/')
source('src/R/graphEnv.R')
library("data.table")


fL <- list.files("out",pattern="telia_dir")
fs <- read.csv("out/telia_dir_loc.csv")
fs = fs[1:(nrow(fs)-1),]
melted <- melt(fs,id.vars="time")
melted$time = as.Date(melted$time)
#melted$time = as.POSIXct(melted$time,format="%Y-%m-%d %H:%M")
melted1 = ddply(melted,.(time,variable),summarise,value=sum(value,na.rm=TRUE))
melted1$week = strftime(melted1$time, format = "%V")

melted = melted1

meltedw = ddply(melted,.(week,variable),summarise,value=mean(value,na.rm=TRUE),time=head(time,1))
derivM = as.matrix(melted %>% xtabs("value ~ time + variable",.))
derivM = derivM[2:nrow(derivM),] - derivM[1:(nrow(derivM)-1),]
#derivM = derivM[2:(nrow(derivM)-1),]
spikeL = colnames(derivM)[rev(order(abs(colSums(derivM))))][1:5]
melted = melted[melted$variable %in% spikeL,]

meltedw = ddply(melted,.(week,variable),summarise,value=mean(value,na.rm=TRUE),time=head(time,1))
meltedw$time = meltedw$time + 3#*60*60*24
#meltedw$time = as.Date(paste(2017,meltedw$week, 1, sep="-"), "%Y-%U-%u")
##dsw = cs[,lapply(.SD,sum(./7.,nan.rm=T)),by="week"]
wD = data.frame(time=unique(melted$time),wday=wday(unique(melted$time)))
wD = wD[wD$wday<3,]
wD$color = gCol1[wD$wday]

gLabel = c("date","count",paste("count density"),"location","convexity")
p <- ggplot(melted,aes(x=time,y=value,group=variable,color=variable)) +
#    geom_vline(data=wD,aes(xintercept=time,color=color),alpha=0.3,linetype="longdash",show.legend=FALSE) + 
    geom_line(aes(color=variable)) +
    geom_line(data=meltedw,aes(y=value),size=2,alpha=0.5,linetype="dashed") +
#    ylim(c(0,1e05)) + 
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p




## fs <- readLines(gzfile("log/telia/spikes.csv.tar.gz",'rt'))
## fs <- read.csv(textConnection(fs),header=FALSE)
## colnames(fs) <- c("id","name","direction","time","weekday","total","col1","col2","col3","light","heavy","vehic","person")
## fs = fs[fs$name %in% unique(fs$name)[1:10],]
## fs$time = as.Date(fs$time,"%Y-%m-%d %H:%M")
## gs = ddply(fs,.(name,time),summarise,person=sum(person))
## gLabel = c("date","count",paste("count density"),"location","convexity")
## p = ggplot(gs,aes(x=time,color=name,y=person,group=name)) +
##     geom_line() +
##     labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
## p

#select * from tile_count_daily_hours_sum_20170924 where tile_id=5883827;
fs1 <- read.csv("raw/speed_prof1.csv",stringsAsFactor=F)
#select tile_id, ts_start, east_out, east_out_speed, north_out, north_out_speed, south_out, south_out_speed, west_out, west_out_speed from tile_direction_daily_hours_sum_20170924 where tile_id=5883827;
fs2 <- read.csv("raw/telia/dir_loc116.csv",stringsAsFactor=F)
fs2$version = "v5"
fs2$version[fs2$fromDateTime > 1506816000000/1000] = "v6"
fs2$tile_id = 213
fs2$hour = format(as.POSIXct(fs2$fromDateTime,origin="1970-01-01",tz="GMT"),"%H")
fs2$day = format(as.POSIXct(fs2$fromDateTime,origin="1970-01-01",tz="GMT"),"%Y-%m-%d")
fs2$ts_start = fs2$fromDateTime
fs2$ts_end = fs2$toDateTime
## as.POSIXct(fs2$fromDateTime,origin="1970-01-01",tz="GMT") 
## new Date("2017-10-01").getTime()

fs2 %>% head
fs2$day <- substring(fs2$ts_start,2,11)
fs2$hour <- substring(fs2$ts_start,13,14)
fs2$grp <- paste(fs2$tile_id,fs2$day)
fs2$tile_id <- as.factor(fs2$tile_id)
fs2[,"in_flux"] = fs2$north_in + fs2$east_in + fs2$south_in + fs2$west_in
fs2$out_flux= fs2$north_out + fs2$east_out + fs2$south_out + fs2$west_out
fs2$diff = fs2[,"in_flux"] - fs2[,"out_flux"]

p <- ggplot(fs2,aes(x=diff)) +
    geom_density()
p

if(FALSE){
    melted <- melt(fs2[,c("tile_id","east_out_speed","west_out_speed","north_out_speed","south_out_speed","east_in_speed","west_in_speed","north_in_speed","south_in_speed","version","day","hour")],id.vars=c("tile_id","version","day","hour"))
}
if(TRUE){
    melted <- melt(fs2[,c("tile_id","east_out","west_out","north_out","south_out","east_in","west_in","north_in","south_in","version","day","hour")],id.vars=c("tile_id","version","day","hour"))
}

melted$dir = "out"
melted[grepl("in",melted$variable),"dir"] = "in"
melted$variable <- melted$variable %>% gsub("_out_speed","",.) %>% gsub("_in_speed","",.) 
melted$variable <- melted$variable %>% gsub("_out","",.) %>% gsub("_in","",.) 
melted$variable <- factor(melted$variable,levels=c("north","east","south","west"))
##melted <- melted[melted$variable!="west",]
#melted = melted[melted$value > 0,]
melted$version <- as.factor(melted$version)
velM <- ddply(melted,.(variable,version,dir),summarise,mean=mean(value,na.rm=T),std=sd(value,na.rm=T))
con <- pipe("xclip -selection clipboard -i", open="w")
write.table(velM,con,row.names=F,col.names=F,sep=",")
close(con)
velM$y = velM$mean
velM$y[velM$dir=="in"] = velM$y[velM$dir=="in"]*1.4

gLabel = c("version","speed",paste("speed density"),"version","convexity")
gLabel = c("flux","count",paste("count density"),"version","convexity")
p = ggplot(melted,aes(x=variable,y=value,group=variable)) + 
    geom_jitter(aes(color=dir,fill=variable),alpha=0.4,size=1) +
#    geom_boxplot(aes(y=value,fill=variable),alpha=0.4,size=1) +
    RadarTheme +
    geom_text(data=velM,aes(x=variable,y=y,label=paste(round(mean),"±",round(std))),color="black",size=5) + 
    theme(axis.text.x=element_text(angle=0, hjust=1)) + 
    coord_radar() +
#    scale_y_continuous(limits=c(0,1.)) +
    guides(fill=guide_legend(keywidth=rel(1.3),keyheight=rel(1.3))) +
    facet_grid(. ~ version) +
    scale_color_manual(values=gCol1[seq(1,10,3)]) +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p

melted$variable = paste(melted$variable,melted$dir,sep="-")
melted = melted[melted$value > 0,]

gLabel = c("dir counts","density",paste("dir counts density"),"direction","convexity")
p <- ggplot(melted,aes(x=value,color=variable)) +
    geom_density(alpha=.5) + 
    theme(legend.position="bottom",
          text = element_text(size = gFontSize)) + 
    scale_color_manual(values=gCol1) +
    xlim(c(0,1000)) + 
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p


gLabel = c("version","speed",paste("speed density"),"version","convexity")
p <- ggplot(melted,aes(x=version,y=value,color=version)) +
    geom_boxplot(alpha=.3,size=1.1) +
    geom_violin(alpha=.5) +
    geom_jitter(alpha=.3) + 
    theme(legend.position="bottom",
          text = element_text(size = gFontSize)) + 
    scale_color_manual(values=gCol1) +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p



fs1$day <- substring(fs1$ts_start,2,11)
fs1$hour <- substring(fs1$ts_start,13,14)
fs1$sig <- fs1$all_events_sum
fs1$grp <- paste(fs1$tile_id,fs1$day)
fs1$tile_id <- as.factor(fs1$tile_id)

gLabel = c("hour","activities",paste("activity count per hour"),"shape","convexity")
p <- ggplot(fs1,aes(x=hour,y=sig,group=grp,color=tile_id)) +
    geom_line(alpha=.3,size=1.1) +
    theme(legend.position="bottom",
          text = element_text(size = gFontSize)) + 
    scale_color_manual(values=gCol1) +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p

gLabel = c("version","activities",paste("activity count per hour"),"version","convexity")
p <- ggplot(fs1,aes(x=version,y=sig,color=version)) +
    geom_boxplot(alpha=.3,size=1.1) +
    geom_violin(alpha=.5) + 
    geom_jitter(alpha=.3) + 
    theme(legend.position="bottom",
          text = element_text(size = gFontSize)) + 
    scale_color_manual(values=gCol1) +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p



