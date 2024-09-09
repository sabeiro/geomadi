#!/usr/bin/env Rscript
setwd('~/lav/motion/')
source('src/R/graphEnv.R')
require(stats)
library(grid)

fs <- read.csv('out/junction_country.csv')
gs <- read.csv('out/junction_motorway.csv')


