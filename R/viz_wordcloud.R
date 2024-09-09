setwd('~/lav/motion')
source('src/R/graphEnv.R')
library('tm')
library(wordcloud)
library(cluster)
library(tripack)
library(deldir)
library(igraph)


corR = data.frame()
for(i in c("raw/mc/attribute_table.csv","raw/tank/poi.csv","raw/mc/poi_mc.csv")){
    print(i)
    gs <- read.csv(i,stringsAsFactor=F)
    nums <- unlist(lapply(gs, is.numeric))  
    gs = gs[,nums]
    corMat = cor(gs)
    corR1 = as.data.frame(rowSums(corMat,na.rm=TRUE))
    colnames(corR1)[1] = "weight"
    corR1['text'] = rownames(corR1)
    corR = rbind(corR,corR1)
}
corR$text = gsub("^t_","",corR$text)
corR$text = gsub("^m_","",corR$text)
corR$weight = abs(corR$weight)

svg("www/f/wordcloud.svg", width=1080/80,height=720/80)
wordcloud(corR$text,corR$weight,scale=c(8,.3),min.freq=0.1,max.words=100,random.order=T,rot.per=.15,colors=gCol1,vfont=c("sans serif","plain"))
dev.off()


fs = fs[order(fs$group,fs$skill),]
textL = paste(fs$group,fs$label,sep=" | ")
textV = strsplit(textL,split=" \\| ")
textI = sort(unique(unlist(textV)))
textDf = matrix(0,nrow=nrow(fs),ncol=length(textI))
colnames(textDf) = textI
rownames(textDf) = fs$skill
i=1
for(i in 1:nrow(fs)){
    for(j in unlist(textV[i])){
        k = match(j,textI)
        textDf[i,k] = textDf[i,k] + 1*fs[i,"level"]
    }
}
textM = textDf %*% t(textDf)

textL = tryTolower(textL)
textS = textL %>% gsub("[[:digit:]]","",.) %>% gsub("[[:punct:]]","",.)
corpusC <- Corpus(VectorSource(textS))
tdm <- TermDocumentMatrix(corpusC)
mTdm = as.matrix(tdm)
wordF = sort(rowSums(mTdm),decreasing=T)
cProb = 0.6
kN = 9
lim = max(quantile(wordF,probs=cProb))
good = mTdm %*% t(mTdm)
good = textM
#good <- good[rowSums(good)>lim,colSums(good)>lim]

adja_matrix <- good
diag(adja_matrix) <- 0
affi_matrix <- good
##    mode=c("directed", "undirected", "max","min", "upper", "lower", "plus"),
gp_graph = graph.adjacency(adja_matrix,weighted=TRUE,mode="max",add.rownames=TRUE)
##posi_matrix = layout.fruchterman.reingold(gp_graph, list(weightsA=E(gp_graph)$weight))
##posi_matrix = layout.drl(gp_graph, list(weightsA=E(gp_graph)$weight))
posi_matrix = layout.spring(gp_graph, list(weightsA=E(gp_graph)$weight))
posi_matrix = cbind(V(gp_graph)$name, posi_matrix)
gp_df = data.frame(posi_matrix, stringsAsFactors=FALSE)
names(gp_df) = c("word", "x", "y")
gp_df$x = as.numeric(gp_df$x)
gp_df$y = as.numeric(gp_df$y)
se = diag(affi_matrix) / max(diag(affi_matrix))
words_km = kmeans(cbind(as.numeric(posi_matrix[,2]), as.numeric(posi_matrix[,3])), kN)
w_size <- diag(affi_matrix)#^(0.1)
gp_df = transform(gp_df, freq=w_size, cluster=as.factor(words_km$cluster))
V <- voronoi.mosaic(words_km$centers[,1],words_km$centers[,2])
P <- voronoi.polygons(V)
voro <- deldir(words_km$centers[,1],words_km$centers[,2])
row.names(gp_df) = 1:nrow(gp_df)
fs = merge(fs,gp_df[,c("word","cluster")],by.x="skill",by.y="word",all.x=T)
gp_df = merge(gp_df,fs[,c("skill","group")],by.y="skill",by.x="word",all.x=T)
#gp_df$group = fs$group


groupL = ddply(fs,.(group),summarise,size=sum(level)/4)
groupL$cluster = ddply(gp_df,.(group),summarise,cluster=floor(mean(as.numeric(cluster))))$cluster
groupL = groupL[order(groupL$cluster),]
groupL$x = words_km$centers[,1]
groupL$y = words_km$centers[,2]
textW = data.frame(word=fs$skill,desc=textL)
textW = merge(textW,gp_df[,c("word","cluster")])
groupL$word = ddply(textW,.(cluster),summarise,word=paste(desc,collapse=" | "))$word
groupL$label = "rest"
i=7
for(i in 1:nrow(groupL)){
    catN = rev(sort(table(unlist(strsplit(groupL[i,"word"]," \\| ")))))
    groupL$label[i] = names(catN[1])
}
groupL[,c("label","group","cluster")]

gLabel <- c("","","","","")
gp_words = ggplot(gp_df, aes(x=x, y=y)) +
    geom_text(aes(size=freq, label=gp_df$word, alpha=.90, color=as.factor(cluster))) +
    geom_text(data=groupL,aes(size=size,label=group,color=as.factor(cluster)),alpha=.3) +
#    geom_segment(aes(x = x1, y = y1, xend = x2, yend = y2),size = 1,data = voro$dirsgs,linetype = "dotted",color= "#FFB958",alpha=.5) +
scale_size_continuous(breaks = c(10,20,30,40,50,60,70,80,90), range = c(1,8)) +
    scale_colour_manual(values=gCol1) +
    scale_x_continuous(breaks=c(min(gp_df$x), max(gp_df$x)), labels=c("","")) +
    scale_y_continuous(breaks=c(min(gp_df$y), max(gp_df$y)), labels=c("","")) +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3]) +
    theme(panel.grid.major=element_blank(),
          legend.position="none",
          panel.background = element_rect(fill="transparent",colour=NA),
          panel.grid.minor=element_blank(),
          axis.ticks=element_blank(),
          title = element_text("Skills clustering"),
          plot.title = element_text(size=12))
plot(gp_words)
ggsave(file="intertino/fig/skillCloud.svg",plot=gp_words,width=gWidth,height=gHeight)
##ggsave(file="intertino/fig/skillCloudSmall.svg",plot=gp_words,height=gWidth,width=gHeight)
urlPrefix = "https://github.com/sabeiro/"

empL <- list()
for(i in 1:nrow(groupL)){
    tmp = gp_df[gp_df$group==groupL[i,"group"],]
    emlL1 <- list()
    for(j in 1:nrow(tmp)){
        empL1[[j]] = list(name=tmp[j,"word"],size=tmp[j,"freq"],,x=tmp[j,"x"],y=tmp[j,"y"],desc="")
    }
    empL[[i]] = list(name=groupL[i,"group"],size=groupL[i,"size"],x=groupL[i,"x"],y=groupL[i,"y"])
}
write(toJSON(list(name="skills",color="rgba(255,255,255,0.9)",title="...",children=empL)),"intertino/data/networkSkill.json")

melted = gp_df[,c("word","cluster","freq","x","y")]
colnames(melted) = c("name","group","size","x","y")
melted$idx = 1:nrow(melted)
melted$size = melted$size*10
tList <- list()
for(i in 1:nrow(melted)){
    rList = "list("
    for(j in colnames(melted)){
        chr <-  ifelse(typeof(melted[i,j])=="character",paste(j,"=\"",melted[i,j],"\",",sep=""),paste(j,"=",melted[i,j],",",sep=""))
        rList <- paste(rList,chr)
    }
    rList <- paste(gsub(",$","",rList),")",sep="")
    rVect <- eval(parse(text=rList))
    tList[[i]] = rVect
}
write(toJSON(list(nodes=tList)),"intertino/data/networkSkillNode.json")
Rth = 2.
distL = list()
k=1
for(i in 1:nrow(gp_df)){
    for(j in (i+1):nrow(gp_df)){
        distR = abs(gp_df[i,"x"] - gp_df[j,"x"]) + abs(gp_df[i,"y"] - gp_df[j,"y"])
        chargeR = gp_df[i,"freq"]*gp_df[j,"freq"]
        if(is.na(distR)){next}
        if(distR > Rth){next}
        distL[[k]] = list(source=i-1,target=j-1,value=distR,distance=distR,charge=chargeR,name=gp_df[i,"word"])
        k = k + 1 
    }
}
write(toJSON(list(links=distL)),"intertino/data/networkSkillLink.json")




##-------------------------------radar-plot------------------------------

fs <- read.csv("train/skills.csv",stringsAsFactor=F)
colnames(fs) <- c("X","variable","value","label")
melted <- fs
melted = melted[order(melted$X,melted$variable),]
melted$X = melted$X %>% gsub(" ","\n",.)
polyP = list()
nCol = 1
for(chPoly in unique(melted$variable)){
    melted1 = melted[melted$variable==chPoly,]
    polyP[[chPoly]] = ggplot(melted1,aes(x=X,y=value,group=variable)) + 
        geom_polygon(aes(group=variable,color=variable,fill=variable),alpha=0.4,size=1,show.legend=TRUE) +
        RadarTheme +
        theme(axis.text.x=element_text(angle=0, hjust=1)) + 
        coord_radar() +
        scale_y_continuous(limits=c(0,1)) +
##        scale_x_discrete(labels=melted$X, expand=c(0, 0)) +
        ##facet_grid(variable ~ .,scales="free",space="free") + 
        ## facet_wrap( ~ variable,ncol=3) +
        guides(fill = guide_legend(keywidth = rel(1.3), keyheight = rel(1.3))) + 
        labs(x=gLabel[1],y=gLabel[2],title=chPoly,color=gLabel[4],fill=gLabel[5])
    polyP[[chPoly]]$color = gCol1[nCol]
    nCol = nCol + 1
}
grid.arrange(grobs=lapply(polyP,function(p,i){p + scale_color_manual(values=p$color) +  scale_fill_manual(values=p$color)}),ncol=3)


svg("intertino/fig/skillRadar.svg")
#jpeg("intertino/fig/skillRadar.jpg",width=pngWidth,height=pngHeight)
grid.arrange(grobs=lapply(polyP,function(p,i){p + scale_color_manual(values=p$color) +  scale_fill_manual(values=p$color)}),ncol=3)
dev.off()




