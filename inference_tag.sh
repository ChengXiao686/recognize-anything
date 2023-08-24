#!/bin/bash

#############################################################################
# Copyright 2023 Autra.Tech. All Rights Reserved.
#############################################################################

RECORD_PATH=$1

python3 batch_infer_mine.py \
  --backbone swin_l \
  --checkpoint pretrained/ram_swin_large_14m.pth \
  --threshold-file ram/data/ram_tag_list_threshold.txt \
  --save-tags True \
  --is-prod False \
  --batch-size 64 \
  --record-path $RECORD_PATH
