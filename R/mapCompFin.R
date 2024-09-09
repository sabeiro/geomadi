#!/usr/bin/env Rscript
setwd('~/lav/motion/')
source('src/R/graphEnv.R')
library('svglite')
require(stats)
library(grid)
library('rjson')
library('jsonlite')
library('RJSONIO')
library(RCurl)


fs <- read.csv("raw/imsiCountFin.csv")
fs1 <- read.csv("raw/popCountFin.csv")
