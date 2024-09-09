#!/bin/bash

# variables
RUN_SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ACTIVITY_HOME="$RUN_SCRIPTS_DIR/../"
COMMONS_HOME="$RUN_SCRIPTS_DIR/../../commons/"

today=`date '+%Y_%m_%d_%H_%M_%S'`;
PY_FILES="/tmp/activity_report_util_scripts_$today.zip"
ZIP_OUTPUT_SRC="$( cd "${ACTIVITY_HOME}src/" && zip -r "$PY_FILES" . )"
ZIP_OUTPUT_COMMONS_SRC="$( cd "${COMMONS_HOME}src/" && zip -ur "$PY_FILES" . )"
ZIP_OUTPUT_COMMONS_CONFIG="$( cd "${COMMONS_HOME}configs/" && zip -ur "$PY_FILES
" . )"

#PY_FILES="$RUN_SCRIPTS_DIR/util_scripts.zip"
#ZIP_OUTPUT="$( cd "${METRIC_AGG_HOME}src/" && zip -r "$RUN_SCRIPTS_DIR/util_scripts.zip" util_scripts )"
LOG_FILE="$( ${RUN_SCRIPTS_DIR}/setup_logging.sh activity_report)"

# set env
export ACTIVITY_HOME=${ACTIVITY_HOME}
export COMMONS_HOME=${COMMONS_HOME}

# run spark
spark-submit --driver-memory 10g --conf spark.driver.maxResultSize=2g --packages com.stratio.datasource:spark-mongodb_2.10:0.11.2 --py-files ${PY_FILES} /home/isturm/jll_activities.py $1 $2 2>&1 | tee $LOG_FILE
SPARK_RESULT_CODE=${PIPESTATUS[0]}

mv ${LOG_FILE} ${LOG_FILE}.done

# restore env
unset ACTIVITY_HOME
unset COMMONS_HOME

exit ${SPARK_RESULT_CODE}
