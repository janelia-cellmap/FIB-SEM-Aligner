#!/bin/bash

#--------------------------------------
# set up global parameters
#--------------------------------------

LOCATION_OWNER="hessh"                             # owner of location render stack
LOCATION_PROJECT="20161213_CS2_Cell13"   # project of render stack used for tile locations
LOCATION_STACK="v1_acquire"                         # render stack used for tile locations
MIN_Z="0"                                           # minimum z value for layers to include in potential tile pairs 
MAX_Z="100000"                                       # maximum z value for layers to include in potential tile pairs

RENDER_OWNER="${LOCATION_OWNER}"                    # owner of render stack used for rendering during point match derivation
RENDER_PROJECT="${LOCATION_PROJECT}"                # project of render stack used for rendering during point match derivation
RENDER_STACK="${LOCATION_STACK}"                    # render stack used for rendering during point match derivation

# base command for running the tile pair client
BASE_CMD="/groups/flyTEM/flyTEM/render/pipeline/bin/run_ws_client.sh 1G org.janelia.render.client.TilePairClient"
PAIR_GEN_LOG=logs/tile_pairs-`date +"%Y%m%d_%H%M%S"`.log

mkdir -p logs

#--------------------------------------
# genrate outside-layer potential pairs
#--------------------------------------

DISTANCE="20"                              # distance in z from each layer to look for potential tile pairs 

# xyNeighborFactor is used to determine radial distance from tile center to look for potential pairs
FILTER_OPTS="--xyNeighborFactor 0.4 --excludeCornerNeighbors false --excludeSameLayerNeighbors true --excludeCompletelyObscuredTiles true"

P1="--baseDataUrl http://10.40.3.162:8080/render-ws/v1 --owner ${LOCATION_OWNER} --project ${LOCATION_PROJECT}"
P2="--baseOwner ${RENDER_OWNER} --baseProject ${RENDER_PROJECT} --baseStack ${RENDER_STACK}"
P3="--stack ${LOCATION_STACK} --minZ ${MIN_Z} --maxZ ${MAX_Z}"
OUTSIDE_JSON="tile_pairs_${LOCATION_STACK}_z_${MIN_Z}_to_${MAX_Z}_dist_${DISTANCE}.json.gz"

${BASE_CMD} ${P1} ${P2} ${P3} ${FILTER_OPTS} --zNeighborDistance ${DISTANCE} --toJson ${OUTSIDE_JSON} | tee -a ${PAIR_GEN_LOG}
