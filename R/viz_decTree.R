#!/usr/bin/env Rscript
setwd('~/lav/motion/')
source('src/R/graphEnv.R')
require(stats)
library(grid)
library('jsonlite')
library('party')

fs <- read.csv("out/activity_score.csv")
fs = fs[!is.na(fs$y_bin),]
summary(fs)
iris_ctree <- ctree(y_corr ~ t_median + t_dist + t_ampl + t_tech + t_conv + t_type, data=fs)
iris_ctree <- ctree(y_bin ~ c_median + c_dist + c_ampl + c_tech + c_conv + c_type, data=fs)
plot(iris_ctree)


fs <- read.csv("out/activity_score.csv")
p <- ggplot(fs,aes(x=y_reg)) +
    geom_histogram() 
p
p <- ggplot(fs,aes(x=z_reg)) +
    geom_histogram() 
p
iris_ctree <- ctree(z_reg ~ c_dist + c_type + c_sum + c_median + c_max + c_inter + c_slope, data=fs)
plot(iris_ctree, type="simple")
