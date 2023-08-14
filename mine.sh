#!/bin/bash

#############################################################################
# Copyright 2023 Autra.Tech. All Rights Reserved.
#############################################################################

RECORD_PATH=$1

python batch_infer_mine.py \
  --model-type ram \
  --backbone swin_l \
  --checkpoint pretrained/ram_swin_large_14m.pth \
  --threshold-file ram/data/ram_tag_list_threshold.txt \
  --record-path $RECORD_PATH
