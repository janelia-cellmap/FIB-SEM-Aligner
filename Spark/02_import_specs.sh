#!/bin/bash

PROJECT_PARAMS="--baseDataUrl http://tem-services:8080/render-ws/v1 --owner hessh --project 20161213_CS2_Cell13"
STACK_PARAMS="--stackResolutionX 8.0 --stackResolutionY 8.0 --stackResolutionZ 8.0"

STACK="v1_acquire"

MANAGE_CMD="/groups/flyTEM/flyTEM/render/bin/manage-stack.sh ${PROJECT_PARAMS}"

${MANAGE_CMD} --action DELETE --stack ${STACK}
${MANAGE_CMD} --action CREATE --stack ${STACK} ${CYCLE_PARAMS} ${STACK_PARAMS}

#${MANAGE_CMD} --action SET_STATE --stackState LOADING --stack ${STACK}

IMPORT_CMD="/groups/flyTEM/flyTEM/render/pipeline/bin/run_ws_client.sh 1G org.janelia.render.client.ImportJsonClient ${PROJECT_PARAMS}"

${IMPORT_CMD} --stack ${STACK} sections.json

${MANAGE_CMD} --action SET_STATE --stackState COMPLETE --stack ${STACK}
