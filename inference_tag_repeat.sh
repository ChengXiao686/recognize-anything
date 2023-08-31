#!/bin/bash

#############################################################################
# Copyright 2023 Autra.Tech. All Rights Reserved.
#############################################################################

RECORD_ARR="A007_20230829_150006 A007_20230829_181256 A007_20230830_152923 A007_20230830_175551 A007_20230830_191528"
DIR="/data/mine_data"
for RECORD_PATH in $RECORD_ARR
    do
    sh ./inference_tag.sh "$DIR/$RECORD_PATH"
    done
