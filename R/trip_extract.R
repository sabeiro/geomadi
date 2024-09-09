library(stringr)
library(geosphere)
`%nin%` = Negate(`%in%`)
setwd('~/lav/motion/')
source('src/R/graphEnv.R')
source('src/R/trip_extractLib.R')
library(sqldf)


input_dir <- "raw/trips"
output_dir <- "out/"
output_file <- "auto_trips1.csv"
output_file_rda <- "df_res_all_20171110.rda"
date_tc = substr(list.dirs(input_dir),nchar(input_dir)+2,nchar(input_dir)+11)
date_tc = unique(date_tc[nchar(date_tc) == 10])
devL <- read.csv("raw/testrun_devicemap.csv")
dev1 <- read.csv("raw/testrun_device.csv")
devRun = melt(dev1,na.rm = TRUE)
devRun = merge(devRun,devL,by.x="value",by.y="dev_num",all.x=TRUE,sort=FALSE)
imsis <- devL['imsi'] 
imsis_lte <-devL[devL$network == "LTE","imsi"]
imsis_2g3g <- devL[devL$network == "2G/3G","imsi"]
underground_cilacs = read.csv("raw/testrun_cilac.csv")

testR <- read.csv("raw/testrun.csv")

melted <- sqldf("SELECT * FROM testR AS t LEFT JOIN devRun AS s ON (t.imsi = s.imsi) AND (t.day = s.date)")
write.csv(melted,"raw/testruns.csv")

dateL = unique(testR$day)
tripL <- testR[testR$day==dateL[1],]
imsiL <- unique(tripL$imsi)
tripL <- tripL[tripL$imsi==imsiL[1],]

max_time_diff <- 2*60*60
min_speed <- 8
max_time_jitter <- 60*3
min_dist <- 0

tmp <- 0
date = dates_tc[1]
for (date in dates_tc){
    df_raw_csv_input <- read.csv(paste(input_dir, "/", date, "/", "part-00000", sep=""), sep="(", header = FALSE)
    if(nrow(df_raw_csv_input) == 0){next;}
    pos1 = as.integer(gregexpr(pattern ='-> ',df_raw_csv_input$V2)) + 3
    pos2 = as.integer(gregexpr(pattern =')',df_raw_csv_input$V2)) -1
    df_raw_csv_input$IMSI <- substr(df_raw_csv_input$V2,pos1,pos2)
    all_imsis_in_file <- unique(df_raw_csv_input$IMSI)
    
    
    for (imsi in imsis){
        print(paste("date: ", date, "imsi: ", imsi))
        if (imsi %nin% all_imsis_in_file){next}
        df_raw <- df_raw_csv_input[df_raw_csv_input$IMSI==imsi,]
        df_raw$events <- strsplit(as.character(df_raw$V1), split = "[[]")
        df <- extract_online(df_raw)
        df$time <- format(as.POSIXct(df$timestamp, origin="1970-01-01"), "%H:%M")
        if(nrow(df)==0){next}
        else if(nrow(df)==1){
            df_one_online <- extract_imsi_date(imsi, date)
            if(exists('df_one_online_all') && is.data.frame(get('df_one_online_all'))){
                df_one_online_all <- rbind(df_one_online_all, df_one_online)
            }
            else{df_one_online_all <- df_one_online}
            next
        }
        nr_events <- nrow(df)
        df <- group_online(df)
        if(nrow(df)==1){
            df_one_online <- extract_imsi_date(imsi, date)
            if(exists('df_one_online_all') && is.data.frame(get('df_one_online_all'))){
                df_one_online_all <- rbind(df_one_online_all, df_one_online)
            }
            else{df_one_online_all <- df_one_online}
            next
        }
        df_res <- match_trips(df, imsi_lte, imsi_2g3g, dates_only_tc, max_time_diff, min_speed, min_dist)
        df_res_min_speed5 <- match_trips(df, imsi_lte, imsi_2g3g, dates_only_tc, max_time_diff, min_speed = 5, min_dist)
        if(identical(df_res, df_res_min_speed5)){
            df_res$speed_flag5 <- "ok"
        }
        else{
            df_res$speed_flag5 <- "check"
        }
        nr_possible_trips <- count_possible_trips(df, max_time_diff, min_speed, min_dist)
        if(nr_possible_trips!=nrow(df_res)){
            df_res$flag_different_match <- "check"
        }
        else{
            df_res$flag_different_match <- "ok"
        }
        print(paste("Nr of Trips: ", nrow(df_res), sep = ""))
        print(paste("Nr of Online-Events: ", nr_events, sep=""))
        print(paste("Nr of Grouped Online-Events: ", nrow(df), sep=""))
        if(exists('df_res_all') && is.data.frame(get('df_res_all'))){
            df_res_all <- rbind(df_res_all, df_res)
        }
        else{
            df_res_all <- df_res
        }
    }
}
df_res_all$cilac <- 
    ifelse((df_res_all$From_Ci_LAC %in% underground_cilacs | 
            df_res_all$To_Ci_LAC %in% underground_cilacs),
           "underground_contained", "all_overground")
df_res_all$any_flag <- 
    ifelse((df_res_all$nr_unmatched_event_groups>0 | 
            df_res_all$speed_flag5=="check" | 
            df_res_all$flag_different_match=="check"|
            df_res_all$cilac=="underground_contained"),
           "check", "ok")
df_res_all$distance_in_km <- gsub(".", ",", as.character(df_res_all$distance_in_km), fixed = T)
nrow(df_res_all[!is.na(df_res_all$From_Ci_LAC),])
nrow(df_res_all[!is.na(df_res_all$From_Ci_LAC)&df_res_all$any_flag=="ok",])
write.table(df_res_all, paste(output_dir, output_file, sep = "/"), row.names=F,na="NA",append=F, quote= FALSE, sep=";", col.names=T)
