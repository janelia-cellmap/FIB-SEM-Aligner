#!/bin/bash

ABSOLUTE_SCRIPT=`readlink -m $0`
SCRIPT_DIR=`dirname ${ABSOLUTE_SCRIPT}`

# ==========================================================================
# Change these parameters for each run ...

export BILL_TO="hessh"

SERVICE_HOST="10.40.3.162:8080"      # use IP address for tem-services until DNS issue is resolved

OWNER="hessh"
PROJECT="20161213_CS2_Cell13"
STACK="v1_align"

BOX_ROOT_DIR="/nrs/hess/rendered_boxes"
WIDTH="18750"
HEIGHT="3000"
FORMAT="tif" # options are: png, jpg, tif

# leave empty to process all layers
#Z_ARGS="--minZ 1 --maxZ 10"
Z_ARGS=""

NUMBER_OF_SPARK_NODES=30

# ==========================================================================
# You should be able to leave everything under here as is ...

# ==========================================================================
# setup output directory
# (NOTE: you can leave this as is if output can go to the standard location)
# ==========================================================================

YEAR_MONTH=`date +"%Y%m"`
DAY=`date +"%d"`
TIME=`date +"%H%M%S"`

export SPARK_OUTPUT_DIR="/groups/hess/hesslab/render/spark_output/${USER}/${YEAR_MONTH}/${DAY}/${TIME}_$$"
export MASTER_ENV_FILE="${SPARK_OUTPUT_DIR}/master_env.sh"

export LOG_DIR="${SPARK_OUTPUT_DIR}/logs"
export TMP="${SPARK_OUTPUT_DIR}/tmp"

echo """
  creating ${SPARK_OUTPUT_DIR}
"""

mkdir -p ${SPARK_OUTPUT_DIR}

# ==========================================================================
# build java args and launch spark
# (NOTE: you should not need to modify anything below)
# ==========================================================================

echo """
  launching Spark job
"""

JAR="/groups/flyTEM/flyTEM/render/lib/render-ws-spark-client-0.3.0-SNAPSHOT-standalone.jar"
CLASS="org.janelia.render.client.spark.BoxClient"

ARGV="--baseDataUrl http://${SERVICE_HOST}/render-ws/v1"
ARGV="${ARGV} --owner ${OWNER} --project ${PROJECT} --stack ${STACK}"
ARGV="${ARGV} --height ${HEIGHT} --width ${WIDTH} --maxLevel 0 --maxOverviewWidthAndHeight 0"
ARGV="${ARGV} --format ${FORMAT}"
ARGV="${ARGV} ${Z_ARGS} --rootDirectory ${BOX_ROOT_DIR}"

/groups/flyTEM/flyTEM/render/spark/bin/inflame.sh ${NUMBER_OF_SPARK_NODES} ${JAR} ${CLASS} ${ARGV}

# from /usr/local/spark-current/conf/spark-env.sh
LIKELY_SPARK_LOG_DIR=~/.spark/logs/$(date +%H-%F)

sleep 2

SPARK_LAUNCH_LOG="spark_launch.log"

qstat | grep spark | tee -a ${SPARK_LAUNCH_LOG}

echo """
  cluster logs will be written to ${LOG_DIR}

  internal spark logs will likely be written to ${LIKELY_SPARK_LOG_DIR}

""" | tee -a ${SPARK_LAUNCH_LOG}
