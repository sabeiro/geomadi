# extract online events
extract_online <- function(df_raw){
  df <- data.frame(index = (1:length(df_raw$events[[1]])), lat = NA, lon = NA, BSE = NA, timestamp = NA, Type = NA, split_event = NA)
  df$split_event <- strsplit(df_raw$events[[1]], split = ",")
  for (i in (2:nrow(df))){
    df[i,"lon"] <- df$split_event[[i]][1]
    df[i, "lat"] <- df$split_event[[i]][2]
    df[i, "BSE"] <- df$split_event[[i]][3]
    df[i, "timestamp"] <- df$split_event[[i]][4]
    df[i, "Type"] <- df$split_event[[i]][5]
  }
  df <- df[df$Type=="Online]",]
  df <- df[!is.na(df$Type),]
  if(nrow(df)!=0){
    # sort by timestamp
    df <- df[with(df, order(timestamp)), ]
    # convert data types
    df$timestamp <- as.numeric(df$timestamp)
    df$lat <- as.numeric(df$lat)
    df$lon <- as.numeric(df$lon)  
  }
  return(df)
}

# group jittering online events
group_online <- function(df_input){
  df <- df_input
  df[1, "index"] <- 1
  for (i in (2:nrow(df))){
    if(((df[i, "timestamp"])-(df[i-1, "timestamp"]))<max_time_jitter){
      df[i,"index"] <- df[i-1, "index"]
      df[i,"BSE"] <- paste(df[i-1, "BSE"], df[i, "BSE"], sep = ", ")
    }
    else{
      df[i,"index"] <- df[i-1, "index"]+1
    }
  }
  # delete redundant jittering events
  df$keep_del <- NA
  for (i in (1:(nrow(df)-1))){
    if(df[i,"index"]==df[i+1, "index"]){
      df[i,"keep_del"] <- "delete"
    }
    else{
      df[i,"keep_del"] <- "keep"
    }
  }
  df[nrow(df), "keep_del"] <- "keep"
  df <- df[df$keep_del=="keep",]
  return(df)
}

# extract trips
match_trips <- function(df, imsi_lte, imsi_2g3g, dates_only_tc, max_time_diff, min_speed, min_dist){
  # create result df
  df_res <- data.frame(index = (1:(nrow(df)/2)),
                       Date = NA, IMSI = NA, From_Ci_LAC = NA,
                       To_Ci_LAC= NA, from_time = NA, To_time = NA,
                       Trip_Type = NA, Trips = NA,
                       Network = NA, Checked_by = NA,
                       timestamp_from = NA, timestamp_to = NA,
                       distance_in_km = NA, nr_unmatched_event_groups = NA,
                       speed_flag5 = NA, flag_different_match = NA)
  
  # network
  if(imsi %in% imsis_lte){
    df_res$Network <- "LTE"
  }
  else if(imsi %in% imsis_2g3g){
    df_res$Network <- "2G/3G"
  }
  # trip type (tc vs. subway)
  if(date %in% dates_only_tc){
    df_res$Trip_Type <- "Traffic_Cell_ODM"
  }
  else{
    df_res$Trip_Type <- "Unknown"
  }
  # trip#
  df_res$Trips <- 1
  # initialize helper objects
  non_match <- 0
  df_res_iter <- 0
  tmp_skip <- FALSE
  for (i in (1:(nrow(df)-1))){
    if(tmp_skip==TRUE){
      tmp_skip<-FALSE
      next
    }
    # time_diff in hours
    tmp_timediff <- (df[i+1, "timestamp"] - df[i, "timestamp"])/(60*60)
    # distance in km
    tmp_dist <- (distm(c(df[i, "lon"], df[i, "lat"]), c(df[i+1, "lon"], df[i+1, "lat"]), fun = distHaversine)/1000)
    tmp_speed <- tmp_dist/tmp_timediff
    # condition 1 max timediff = 2 hours
    # condition 2 min speed = 5 km/h
    # condition 3 distance > 0
    if((tmp_timediff>max_time_diff)|(tmp_speed<min_speed)|(tmp_dist<=min_dist)){
      non_match <- non_match + 1
      tmp_skip <- FALSE
      next
    }
    # case: conditions hold
    else{
      df_res_iter <- df_res_iter + 1
      df_res[df_res_iter, "Date"] <- date
      df_res[df_res_iter, "IMSI"] <- imsi
      df_res[df_res_iter, "From_Ci_LAC"] <- df[i, "BSE"]
      df_res[df_res_iter, "To_Ci_LAC"] <- df[i+1, "BSE"]
      df_res[df_res_iter, "from_time"] <- format(as.POSIXct(df[i, "timestamp"], origin="1970-01-01"), "%H:%M")
      df_res[df_res_iter, "To_time"] <- format(as.POSIXct(df[i+1, "timestamp"], origin="1970-01-01"), "%H:%M")
      df_res[df_res_iter, "Checked_by"] <- "automatic"
      df_res[df_res_iter, "timestamp_from"] <- df[i, "timestamp"]
      df_res[df_res_iter, "timestamp_to"] <- df[i+1, "timestamp"]
      df_res[df_res_iter, "distance_in_km"] <- tmp_dist
      tmp_skip <- TRUE
    }
    # case: last event not matched
  }
  # add IMSI and date to unmatched trips
  for (i in (1:nrow(df_res))){
    if(is.na(df_res[i, "Date"])){
      df_res[i,"Date"] <- date
      df_res[i, "IMSI"] <- imsi
    }
  }
  if(is.null(df_res$To_Ci_LAC)){
    detected_trips <- 0
  }
  else{
    detected_trips <- nrow(df_res[!is.na(df_res$To_Ci_LAC), ])
  }
  df_res$nr_unmatched_event_groups <- (nrow(df)) - (detected_trips * 2)
  
  return(df_res)
}

# count nr of possible trips in order to detect ambiguous trips
count_possible_trips <- function(df, max_time_diff, min_speed, min_dist){
  # initialize number of trips
  nr_trips <- 0
  for (i in (1:(nrow(df)-1))){
    # time_diff in hours
    tmp_timediff <- (df[i+1, "timestamp"] - df[i, "timestamp"])/(60*60)
    # distance in km
    tmp_dist <- (distm(c(df[i, "lon"], df[i, "lat"]), c(df[i+1, "lon"], df[i+1, "lat"]), fun = distHaversine)/1000)
    tmp_speed <- tmp_dist/tmp_timediff
    # condition 1 max timediff = 2 hours
    # condition 2 min speed = 5 km/h
    # condition 3 distance > 0
    
    if((tmp_timediff<=max_time_diff)&(tmp_speed>=min_speed)&(tmp_dist>min_dist)){
      nr_trips <- nr_trips + 1
    }
  }
  return(nr_trips)
}  

# extract imsis/days with only one online event
extract_imsi_date <- function(imsi, date){
  imsi_date <- data.frame(imsi = NA, date = NA, BSE = NA)
  imsi_date$imsi <- imsi
  imsi_date$date <- date
  imsi_date$BSE <- df$BSE
  
  return(imsi_date)  
}


