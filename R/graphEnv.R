## Sys.setlocale(category = "LC_ALL", locale = "English_United States.1252")
## Sys.setenv(LANG = "en_US.UTF-8")
##Sys.setlocale('LC_ALL', 'UTF-8');
options(stringsAsFactors=FALSE)

library(ggplot2)
library(RColorBrewer)
library(gtable)
library(grid)
library(gridExtra)
library(reshape2)
library(plyr)
library(dplyr)
library(magrittr)
library(ggplot2)
library(stringi)
library(RColorBrewer)
##library("textcat")
require(grDevices) # for colours
library(gtable)
library(grid)
library(gridExtra)
library(scales)
library(stringi)
require(stats)

##install.packages(c("ggplot2","RColorBrewer","gtable","grid","gridExtra","reshape2","plyr","dplyr","magrittr","stringi","RColorBrewer","textcat","grDevices","gtable","grid","gridExtra","scales","stringi",'svglite',"stats","grid","'rjson'","'jsonlite'","'RJSONIO'","RCurl","stats"))


gCol = c("#000066","#DCDCDC","#FFb915")
gCol = c("#000033","#FFb915","#AAAAAA")
gCol2 <- c("#b6dbe1","#d2daef","#f5e0e9","#f4e3c9","#f4d9d0","#d6cdc8","#d8e0b1","#dee0d3")
gCol2 = c(gCol2,gCol2,gCol2)
rainbow(12, s = 1, v = 1, start = 0, end = max(1, 12 - 1)/12, alpha = 1)
heat.colors(12, alpha = 1)
terrain.colors(12, alpha = 1)
topo.colors(12, alpha = 1)
cm.colors(12, alpha = 1)
##display.brewer.all()
gCol1 <- c("#97003F","#D6604D","#EE9900","#CC9900","#47B757","#4393C3","#2166AC","#053091","#B2182B","#F4A582","#92C54E","#719540")
gCol1 <- c(brewer.pal(11,'Spectral'),brewer.pal(11,'RdBu'),gCol1)
gCol1[5] <- "#9E50DB"
gCol1[6] <- "#4F9F2F"
gCol1[7] <- "#367528"
gCol1[8] <- "#ABDDA4"
gCol1[13] <- "#B2183B"
gCol1[16] <- "#978747"
gCol1[17] <- "#577787"
gCol1[18] <- "#479737"
##pie(rep(1,length(gCol1)), col=gCol1)

##blu grigio senape corallo verde brillante fucsia
skillPal = c("#B17ACC","#A83399","#CC0099","#EE9900","#FFCC00","#008888","#99BB99","#5A5aCC","#007ACC","#003399","#000099","#4713CC","#86F368","#B5A338");
skillPal = c(skillPal,skillPal,skillPal,skillPal)


gAvCol <- 'red'
gWidth <- 12
gHeight <- 6
pngWidth <- 950
pngHeight <- 600
gRes <- 100
gFontSize  <- 18
gLabelSize  <- 5
pointSize <- 3
lineSize <- c(1.5,1.5,1.5,1.5,2,2,2,2)
gUn <- "in"
gWidth <- 9.5
gHeight <- 6
gRes <- 300

theme_new <- theme_set(theme_bw())
theme_new <- theme_update(
    legend.justification=c(0,0),
    ##    legend.position=c(.75,.73),
    ##    legend.background = element_rect(fill=alpha('white',0.3)),
    text = element_text(size = gFontSize),
    legend.position ="right",
    panel.background = element_blank(),
    panel.border = element_blank(),
    axis.text.x = element_text(angle = 30, hjust = 1),
    legend.key = element_blank(),
    legend.background = element_blank()
                                        #    text = element_text(family = "sans", colour = "grey50", size = gFontSize, vjust = 1, lineheight = 0.9,face="plain",hjust=0.5,angle=0,margin=0,debug=0)
)
RadarTheme <- theme(
    panel.background=element_blank(),
    plot.title= element_text(size=13,face=c("bold","italic")),
    plot.margin = unit(c(20,-20,20,-20), "pt"),
    ## plot.margin = margin=margin(t=-10,unit="pt")),
    text=element_text(family="Open Sans"),aspect.ratio=1,
    legend.position="bottom",
    legend.title=element_blank(),legend.direction="horizontal",
    axis.text.x = element_text(size=12,face="bold",angle=0,vjust=1,hjust=1,margin=margin(t=-40,r=-40,unit="pt")),#
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    strip.background = element_rect(fill="orange"),
    ## axis.line.x=element_line(size=0.5),
    panel.grid.major=element_line(size=0.3,linetype = 2,colour="grey"))

coord_radar <- function(theta="x",start=0,direction=1){
    theta <- match.arg(theta, c("x", "y"))
    r <- if(theta == "x") "y" else "x"
    ggproto("CordRadar", CoordPolar,theta=theta,r=r,start=start,direction=sign(direction),is_linear = function(coord) TRUE)
}

PieTheme <- theme(
    panel.border = element_blank(),
    text = element_text(size = gFontSize),
    axis.line=element_blank(),
    axis.text.x=element_blank(),
    axis.ticks=element_blank(),
    legend.position="none",
    plot.background=element_blank(),
    panel.background = element_blank()
)

facetTheme <- theme(strip.text.x = element_text(size=12, angle=0),
                    strip.text.y = element_text(size=12, face="bold"),
                    strip.background = element_rect(colour="#FFFFFF", fill="#FFFFFF"))

overlapTheme <- theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    text = element_text(size = gFontSize),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    legend.justification=c(0,0),
##    legend.position=c(.5,.73),
    legend.background = element_rect(fill=alpha('white',0.3)),
    legend.position ="right"
)

coord_radar <- function(theta="x",start=0,direction=1){
    theta <- match.arg(theta, c("x", "y"))
    r <- if(theta == "x") "y" else "x"
    ggproto("CordRadar", CoordPolar,theta=theta,r=r,start=start,direction=sign(direction),is_linear = function(coord) TRUE)
}

PieTheme <- theme(
    panel.border = element_blank(),
    text = element_text(size = gFontSize),
    axis.line=element_blank(),
    axis.text.x=element_blank(),
    axis.ticks=element_blank(),
    legend.position="none",
    plot.background=element_blank(),
    panel.background = element_blank()
)

facetTheme <- theme(strip.text.x = element_text(size=12, angle=0),
                    strip.text.y = element_text(size=12, face="bold"),
                    strip.background = element_rect(colour="#FFFFFF", fill="#FFFFFF"))

overlapTheme <- theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    text = element_text(size = gFontSize),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    legend.justification=c(0,0),
    ## legend.position=c(.5,.73),
    legend.background = element_rect(fill=alpha('white',0.3)),
    legend.position ="right"
)

blankTheme <- theme(
    panel.border = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    text = element_text(size = gFontSize),
    axis.ticks = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.line = element_blank(), 
    axis.title.x = element_blank(), 
    axis.title.y = element_blank(), 
    ## axis.ticks.margin = unit(c(0,0,0,0), "lines"), 
    legend.position="none"
) 

calWeek <- function(date) {
    d <- as.numeric(format(date, "%d"))
    m <- as.numeric(format(date, "%m"))
    y <- as.numeric(format(date, "%y"))
    monthSeq <- c(31,28,31,30,31,30,31,31,30,31,30,31)
    monthSeq <- c(0,cumsum(monthSeq))
    ySeq <- c(10,11,12,13,14,15,16)
    mondaySeq <- c(4,3,2,7,6,5,4) + c(0,0,1,1,1,1,2)
    dm <- monthSeq[m] + d - mondaySeq[match(y,ySeq)]
    cw <- ((dm) %/% 7) + 1
    cw[cw==0] <- 52
    cw
}
dayCW <- function(cw){
    monthSeq <- c(31,28,31,30,31,30,31,31,30,31,30,31)
    weekSeq <- c(0,5,9,14,18,22,27,31,36,40,44,49,53)
    d <- cw*7
    m <- floor(cw/4) - 1
    for(i in seq(1:length(weekSeq)-1)){
        if( (cw >= weekSeq[i]) && (cw < weekSeq[i+1]) ){
            m <- i
            break
        }
        d <- d - monthSeq[i]
    }
    dateF <- paste("2015",m,d-2,sep="-") #1st Jan Thu
    as.Date(dateF,format="%Y-%m-%d")
}

URL_parts <- function(x) {
    m <- regexec("^(([^:]+)://)?([^:/]+)(:([0-9]+))?(/.*)", x)
    parts <- do.call(rbind,
                     lapply(regmatches(x, m), `[`, c(3L, 4L, 6L, 7L)))
    colnames(parts) <- c("protocol","host","port","path")
    parts
}


stackBarLabel <- function(bstack,gLabel){
    ## months <- levels(bstack$month)[c(3,2,1)]
    ## topic_names <- list("1"="October","2"="November","3"="December","4"="Misc")
    ## top_labeller <- function(variable,value){
    ##     return(topic_names[value])
    ## }
    pos <- rep(0,length(bstack$timeVar))
    for(xx in names(table(bstack$timeVar))){
        set <- bstack$timeVar == xx
        pos[set] <- cumsum(bstack[set,]$y) - 0.5*bstack[set,]$y
    }
    bstack$pos <- pos
    p <- ggplot(bstack) +
        geom_bar(aes(x=x,y=y,fill=f,group=1),stat="identity",alpha=.90) +
        geom_text(aes(x=x,label=label,y=pos,0),size=5,colour='white') +
        theme(
            axis.text.x = element_text(angle = 30, hjust = 1),
            panel.background = element_blank(),
            text = element_text(size = gFontSize)
        ) +
        labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4]) +
        scale_fill_manual(values=gCol)
    p
}
stackBarLabelNoFill <- function(bstack,gLabel){
    pos <- cumsum(bstack$y) - 0.5*bstack$y
    p <- ggplot(bstack) +
        geom_bar(aes(x=timeVar,y=y,group=1),stat="identity",alpha=.90) +
        geom_text(aes(x=timeVar,label=label,y=pos,0),size=3,colour='white') +
        theme(
            axis.text.x = element_text(angle = 30, hjust = 1),
            panel.background = element_blank(),
            text = element_text(size = gFontSize)
        ) +
        labs(x=gLabel[1],y=gLabel[2],title=gLabel[3]) +
        scale_fill_manual(values=gCol)
    p
}

graphImprVolume <- function(btstack,fName,gLabel){
    bstack$y <- bstack$volume
    bstack$f <- bstack$channel
    bstack$label <- sapply(bstack$share,function(x){paste(round(x),"")})
    p <- stackBarLabel(bstack,gLabel)
    ggsave(file=fName, plot=p, width=gWidth, height=gHeight)
}
graphImprShare <- function(btstack,fName,gLabel){
    bstack$y <- bstack$share
    bstack$f <- bstack$channel
    bstack$label <- sapply(bstack$share,function(x){paste(round(x*100)," %")})
    p <- stackBarLabel(bstack,gLabel)
    ggsave(file=fName, plot=p, width=gWidth, height=gHeight)
}

pasqua <- as.Date(c("2016-03-27","2016-03-28","2015-04-05","2015-04-06","2014-04-20","2014-04-21","2013-03-31","2013-04-01","2012-04-08","2012-04-09"))
format(pasqua,"%W")
festeInv <- as.Date(c("2016-01-01","2016-01-06","2016-08-12","2016-12-25","2016-12-26"))
festeInv <- c(festeInv,as.Date(c("2015-01-01","2015-01-06","2015-08-12","2015-12-25","2015-12-26")))
festeInv <- c(festeInv,as.Date(c("2014-01-01","2014-01-06","2014-08-12","2014-12-25","2014-12-26")))
festeInv <- c(festeInv,as.Date(c("2013-01-01","2013-01-06","2013-08-12","2013-12-25","2013-12-26")))
festeInv <- c(festeInv,as.Date(c("2012-01-01","2012-01-06","2012-08-12","2012-12-25","2012-12-26")))
festeEst <- as.Date(c("2016-04-25","2016-05-01","2016-6-02","2016-08-15","2016-01-11"))
festeEst <- c(festeEst,as.Date(c("2015-04-25","2015-05-01","2016-6-02","2016-08-15","2016-01-11")))
festeEst <- c(festeEst,as.Date(c("2014-04-25","2014-05-01","2016-6-02","2016-08-15","2016-01-11")))
festeEst <- c(festeEst,as.Date(c("2013-04-25","2013-05-01","2016-6-02","2016-08-15","2016-01-11")))
festeEst <- c(festeEst,as.Date(c("2012-04-25","2012-05-01","2016-6-02","2016-08-15","2016-01-11")))
feste <- c(festeInv,pasqua,festeEst)
tryTolower = function(x){
                                        # create missing value
    y = NA
                                        # tryCatch error
    try_error = tryCatch(tolower(x), error=function(e) e)
    if (!inherits(try_error, "error"))
        y = tolower(x)
    return(y)
}


deriv <- function(x, y) diff(y) / diff(x)


map2color<-function(x,pal,limits=NULL){
    if(is.null(limits)) limits=range(x)
    pal[findInterval(x,seq(limits[1],limits[2],length.out=length(pal)+1), all.inside=TRUE)]
}

map2color(0:11,rainbow(200),limits=c(1,10))


comprss <- function(tx) {
    div <- findInterval(as.numeric(gsub("\\,", "", tx)),
                        c(1, 1e3, 1e6, 1e9, 1e12) )
    paste(round( as.numeric(gsub("\\,","",tx))/10^(3*(div-1)), 2),
          c("","K","M","B","T")[div] )}


df2l <- function(tD){
    tList <- list()
    for(i in 1:nrow(tD)){
        rList = "list("
        for(j in colnames(tD)){
            chr <-  ifelse(typeof(tD[i,j])=="character",paste(j,"=\"",tD[i,j],"\",",sep=""),paste(j,"=",tD[i,j],",",sep=""))
            rList <- paste(rList,chr)
        }
        rList <- paste(gsub(",$","",rList),")",sep="")
        rVect <- eval(parse(text=rList))
        tList[[i]] = rVect
    }
    return(tList)
}


group2l <- function(df,gVar){
    grpL <- unique(df[,gVar])
    pointL = list()
    for(i in 1:length(grpL)){
        melted1 = df[df[,gVar]==grpL[i],]
        empG = list()
        for(k in 1:nrow(melted1)){
            rList = "list("
            for(j in colnames(melted1)){
                chr <-  ifelse(typeof(melted1[k,j])=="character",paste(j,"=\"",melted1[k,j],"\",",sep=""),paste(j,"=",melted1[k,j],",",sep=""))
                rList <- paste(rList,chr)
            }
            rList <- paste(gsub(",$","",rList),")",sep="")
            rVect <- eval(parse(text=rList))
            empG[[k]] = rVect
        }
        sumV <- lapply(colSums(melted1[sapply(melted1,is.numeric)]),function(x) x)
        if(length(sumV)>0){
            sumV[['name']] = grpL[i]
            sumV[['size']] = nrow(melted1)
            sumV[['children']] = empG
            pointL[[i]] = sumV
        } else {
            pointL[[i]] = list(name=grpL[i],size=nrow(melted1),children=empG)
        }            
    }
    return(pointL)
}


toInt <- function(col,alpha){
    str <- 'rgba('
    str = paste(str,strtoi(paste("0",substring(col,2,3),sep="x")),",",sep="")
    str = paste(str,strtoi(paste("0",substring(col,4,5),sep="x")),",",sep="")
    str = paste(str,strtoi(paste("0",substring(col,6,7),sep="x")),",",alpha,")",sep="")
    return(str)
}



## Sys.setenv("HADOOP_CMD"="/usr/hdp/2.3.2.0-2950/hadoop/bin/yarn")
## Sys.setenv("HADOOP_STREAMING"="/usr/hdp/2.3.2.0-2950/hadoop-mapreduce/hadoop-streaming-2.7.1.2.3.2.0-2950.jar")
## Sys.setenv("HADOOP_HOME"="/usr/hdp/2.3.2.0-2950/hadoop")
## Sys.setenv("HADOOP_PREFIX"="/usr/hdp/2.3.2.0-2950/hadoop")


