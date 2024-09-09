#!/usr/bin/env Rscript
setwd('~/lav/motion/')
source('src/R/graphEnv.R')
library(gridExtra)

##-----------------source-comparison---------------------
corL <- read.csv("out/activity_cor.csv")
melted = melt(corL[,c("name","cor_bon","cor_sani","cor_max")],id_vars="name")
melted = melted[order(melted$value,decreasing=TRUE),]
melted = melted[!is.na(melted$value),]
gLabel = c("location","correlation",paste("Correlation dependence"),"correlation density","convexity")
p <- ggplot(melted,aes(x=name,y=value,color=variable,group=variable)) +
#    geom_bar(stat="identity",position="dodge",alpha=0.3) + 
    geom_line(aes(fill=variable),position="identity",alpha=.5,size=2) + 
    theme(text = element_text(size = gFontSize),legend.position="bottom") +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p

fs1 = fs[fs$id_poi == 1300,]
melted = melt(fs1[,c("variable","bon","sani")],id_vars="variable")
colnames(melted) = c("time","variable","value")
gLabel = c("time","counts",paste(""),"source","convexity")
p <- ggplot(melted,aes(x=time,y=value,color=variable,group=variable)) +
    geom_line(size=2) + 
    theme(text = element_text(size = gFontSize),legend.position="bottom") +
    scale_x_discrete(breaks=unique(melted$time)[seq(1,nrow(fs1),2)]) +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p
##----------------confusion-results----------------------

fs <- corL
gs <- read.csv("out/activity_class_bon.csv")
lineG = data.frame(x=c(0.6,0.6),y=c(0.,5.))
percOver = data.frame(perc=c(
                          nrow(fs[fs$cor_bon  > .6,])/nrow(fs)
                         ,nrow(fs[fs$cor_sani > .6,])/nrow(fs)
                         ,nrow(fs[fs$cor_max  > .6,])/nrow(fs)
                         ,nrow(gs[gs$y_corr   > .6,])/nrow(gs)
                         ,nrow(gs[gs$pred>2 & gs$y_corr > .6,])/nrow(gs)) )
percOver$x = 0.1
percOver$y = c(2.5,3.0,3.5,4.0,4.5)
percOver$color = c("bon","sani","max","all cells","filtered cells")

gLabel = c("Pearson r","density",paste("Correlation density"),"correlation density","convexity")
p <- ggplot() +
    geom_density(data=gs,aes(x=y_corr,fill="all cells",color="all cells"),alpha=.3) + 
    geom_density(data=gs[gs$pred>3,],aes(x=y_corr,fill="filtered cells",color="filtered cells"),alpha=.3) + 
    geom_density(data=fs,aes(x=cor_bon,fill="bon",color="bon"),alpha=.3) +
    geom_density(data=fs,aes(x=cor_sani,fill="sani",color="sani"),alpha=.3) +
    geom_density(data=fs,aes(x=cor_max,fill="max",color="max"),alpha=.3) +
    geom_line(data=lineG,aes(x=x,y=y),color="red") +
    geom_text(data=percOver,aes(x=x,y=y,color=color,label=paste(color,":",round(perc,2),"%")),size=09) + 
    theme(text = element_text(size = gFontSize),legend.position="bottom") +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p

fs <- read.csv("out/activity_pred_bon.csv")
gs <- read.csv("out/activity_class_bon.csv")
gs <- ddply(gs,"id_poi",summarise,c_source=head(c_source,1))
fs[,'c_source'] = merge(fs,gs,by.x="id_poi",by.y="id_poi",all.x=T)['c_source']
fs$sum <- rowSums(fs[,colnames(fs) %>% grepl("X.",.)])
fs$c_sum = cut(fs$sum,breaks=c(0,2000,4000,6000,8000),labels=c(2000,4000,6000,8000))
fs$count = 1
ts <- as.data.frame.matrix(xtabs("corr ~ type + c_sum",data=fs))
cs <- as.data.frame.matrix(xtabs("count ~ type + c_sum",data=fs))
ts = ts/cs
ts$type = rownames(ts)
tabM <- melt(ts,id.vars = "type")
tabM[is.na(tabM$value),"value"] = 0
gLabel = c("type","activities",paste(""),"correlation sum","convexity")
p = ggplot(tabM,aes(x=type,y=variable,fill=value)) +
    geom_tile(alpha=1) +
    geom_text(aes(label=round(value,2)),alpha=1,size=8) +
    theme(text = element_text(size = gFontSize),legend.position="bottom") +
    scale_fill_gradient(low="white",high="steelblue") + #,limits=c(0,25)) +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p

ts <- as.data.frame.matrix(xtabs("corr ~ c_source + type",data=fs))
cs <- as.data.frame.matrix(xtabs("count ~ c_source + type",data=fs))
ts = ts/cs
ts$type = rownames(ts)
cs$type = rownames(cs)
tabM <- melt(ts,id.vars = "type")
tabM1 <- melt(cs,id.vars = "type")
tabM$count <- tabM1$value
tabM[is.na(tabM$value),"value"] = 0
gLabel = c("source number","location type",paste(""),"correlation sum","convexity")
p = ggplot(tabM,aes(x=type,y=variable,fill=value)) +
    geom_tile(alpha=1) +
    geom_text(aes(label=paste(round(value,2),count,sep="#")),alpha=1,size=8) +
    theme(text = element_text(size = gFontSize),legend.position="bottom") +
    scale_fill_gradient(low="white",high="steelblue") + #,limits=c(0,25)) +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p

##------------------------plot-dendrogram----------------------------------
##install.packages(c("heatmaply","corrplot"))
## library(heatmaply)
## heatmaply(mtcars, k_col = 2, k_row = 3) %&gt;% layout(margin = list(l = 130, b = 40))
library(rpart)
library(ggdendro)
hL = colnames(fs)[colnames(fs) %>% grepl("^X.",.)][seq(2,19)]
lL = colnames(fs)[grepl("^c_.",colnames(fs))]
model <- hclust(dist (t(fs[,lL])), "ave")
dhc <- as.dendrogram(model)
# Rectangular lines
ddata <- dendro_data(dhc, type = "rectangle")
p <- ggplot() + 
    geom_segment(data=segment(ddata),aes(x=x,y=y,xend=xend,yend=yend)) + 
    geom_text(data=ddata$labels,aes(x=x+0.2,y=y,label=label)) + 
    coord_flip() + 
    scale_y_reverse(expand = c(0.2, 0))
p
    library(corrplot)
M <- cor(fs[,lL])
corrplot.mixed(M, lower = "circle",upper="number")

if(FALSE){
    model <- rpart(y_corr ~ t_max+t_dist+t_median+t_inter+t_convex+c_tech+c_type, method = "class", data=fs)
    ddata <- dendro_data(model)
    ggplot() + 
        geom_segment(data = ddata$segments, 
                     aes(x = x, y = y, xend = xend, yend = yend)) + 
        geom_text(data = ddata$labels, 
                  aes(x = x, y = y, label = label), size = 3, vjust = 0) +
        geom_text(data = ddata$leaf_labels, 
                  aes(x = x, y = y, label = label), size = 3, vjust = 1) +
        theme_dendro()
    library(riverplot)
    sp = fs[,c("c_max","c_dist","c_median","c_inter","c_convex","c_tech","c_type","y_corr")]
    melted <- melt(sp,id.vars="y_corr")
    melted$value = as.factor(melted$value)
    
    edges = data.frame(N1 = paste0(rep(LETTERS[1:4], each = 4), rep(1:5, each = 16)),
                       N2 = paste0(rep(LETTERS[1:4], 4), rep(2:6, each = 16)),
                       Value = runif(80, min = 2, max = 5) * rep(c(1, 0.8, 0.6, 0.4, 0.3), each = 16),
                       stringsAsFactors = F)
    edges = edges[sample(c(TRUE, FALSE), nrow(edges), replace = TRUE, prob = c(0.8, 0.2)),]
    head(edges)
    nodes = data.frame(ID = unique(c(edges$N1, edges$N2)), stringsAsFactors = FALSE)
    nodes$x = as.integer(substr(nodes$ID, 2, 2))
    nodes$y = as.integer(sapply(substr(nodes$ID, 1, 1), charToRaw)) - 65
    rownames(nodes) = nodes$ID
    head(nodes)
    palette = paste0(brewer.pal(4, "Set1"), "60")
    styles = lapply(nodes$y, function(n) {list(col = palette[n+1], lty = 0, textcol = "black")})
    names(styles) = nodes$ID
    rp <- list(nodes = nodes, edges = edges, styles = styles)
    class(rp) <- c(class(rp), "riverplot")
    plot(rp, plot_area = 0.95, yscale=0.06)
}


##---------------------plot-confusion-matrix-------------------------------
nL = length(lL)
lLab = c("activity","stat prop","covariance","convexity","type")
polyP = list()
chPoly = 1
for(i in seq(nL-1)){
    for(j in seq(i+1,nL)){
        tabM <- melt(table(fs[,lL[i]],fs[,lL[j]]))
        gLabel = c("","",paste(""),"counts","convexity")
        polyP[[chPoly]] = ggplot(tabM,aes(x=Var1,y=Var2,fill=value)) +
            geom_tile(alpha=1) +
            theme(text = element_text(size = gFontSize),legend.position="bottom") +
            scale_fill_gradient(low="white",high="steelblue",limits=c(0,25)) +
            labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
        chPoly = chPoly + 1
    }
}
clustL <- fs[,lL]
colnames(clustL) <- lLab
tabM <- melt(cor(clustL))
gLabel = c("","",paste(""),"correlation","convexity")
pCor <- ggplot(tabM,aes(x=Var1,y=Var2,fill=value)) +
            geom_tile(alpha=1) +
            theme(text = element_text(size = gFontSize),legend.position="right") +
            scale_fill_gradient(low="white",high="steelblue") +
            labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])

#gtable_filter(ggplot_gtable(ggplot_build(p)), "guide-box")
tmp <- ggplot_gtable(ggplot_build(polyP[[1]]))
leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
legend <- tmp$grobs[[leg]]
lheight <- sum(legend$height)
lwidth <- sum(legend$width)
xax <- lapply(lLab[2:length(lLab)],function(l) textGrob(l))
yax <- lapply(lLab[1:(length(lLab)-1)],function(l) textGrob(l))
lay <- rbind(c(1,2,3,4),c(NA,5,6,7),c(11,11,8,9),c(11,11,NA,10))
gl = lapply(polyP,function(p,i){p + theme(legend.position="none",plot.margin=unit(c(0,0,0.2,0.2),"cm"))+ labs(x=NULL, y=NULL)})
extL = append(gl,list(pCor))
ga <- grid.arrange(legend,grobs=extL,legend,ncol=4,layout_matrix=lay)
gax <- grid.arrange(grobs=xax,ncol=4)
gay <- grid.arrange(grobs=yax,ncol=1)
gaa <- grid.arrange(gax,ga,ncol=1,heights=c(1,20))
gap <- grid.arrange(gaa,gay,ncol=2,top=textGrob("confusion matrices",gp=gpar(fontsize=20,font=3)),widths=c(10,1))
combined = arrangeGrob(gap,legend,ncol=1,heights = unit.c(unit(1, "npc") - lheight, lheight))
grid.newpage()
ggsave(file="fig/confusion.jpg", plot=combined, width=2*gWidth, height=2*gHeight)





select_grobs <- function(lay) {
  id <- unique(c(t(lay))) 
  id[!is.na(id)]
} 
select_grobs(lay)
grid.arrange(grobs=gs[select_grobs(lay)], layout_matrix=lay)

gs <- lapply(1:9, function(ii) 
  grobTree(rectGrob(gp=gpar(fill=ii, alpha=0.5)), textGrob(ii)))
grid.arrange(grobs=gs, ncol=4, 
               top="top label", bottom="bottom\nlabel", 
               left="left label", right="right label")
grid.rect(gp=gpar(fill=NA))







