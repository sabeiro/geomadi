#!/usr/bin/env Rscript
setwd('~/lav/motion/')
source('src/R/graphEnv.R')

fs <- read.csv("raw/procTimeSe.csv")
fs$Data.day <- as.Date(fs$Data.day)
for(i in colnames(fs)[2:ncol(fs)]){
    ##fs[,i] = as.POSIXlt(fs[,i],format="%Y-%m-%d %H:%M")
    fs[,i] = as.Date(fs[,i],format="%Y-%m-%d %H:%M")
}
fs <- fs[rowSums(!is.na(fs)) > 1,]
fs %>% head(.,10)
melted <- melt(fs,id.vars="Data.day")
melted %>% head
#melted = melted[!is.na(melted)]

gs = fs[,c("Data.day","ma")]
gs = gs[!is.na(gs$ma),]
gs$Data.day <- as.numeric(gs$Data.day)
gs$ma <- as.numeric(gs$ma)
fit <- lm("ma ~ Data.day",gs[gs$Data.day > as.numeric(as.Date("2017-10-01")),])
fitD = data.frame(t=seq( as.Date("2017-10-01"), by=1, len=80))
fitD$y = fit$coefficients[[1]] + fit$coefficients[[2]]*as.numeric(fitD$t)
fitD$y = as.Date(fitD$y,origin="1970-01-01")

print(fit$coefficients[[2]]*24)
print(fitD[fitD$t -fitD$y-2 >1 & fitD$t -fitD$y-2 < 3,])

gLabel = c("comp day","proc day",paste("processing time"),"metric","metric")
p <- ggplot(melted,aes(x=Data.day,y=value,color=variable)) +
    geom_line(alpha=.7,size=1.1) +
    geom_line(data=fitD,aes(x=fitD$t,y=fitD$y,color="fit")) + 
    theme(legend.position="bottom",
              text = element_text(size = gFontSize)) + 
#    scale_color_manual(values=gCol1) +
    labs(x=gLabel[1],y=gLabel[2],title=gLabel[3],fill=gLabel[4],color=gLabel[4],linetype=gLabel[5])
p

