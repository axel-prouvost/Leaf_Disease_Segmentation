#!/bin/bash


python generate_groundtruth_masks.py \
  --images_dir ./Feuilles_test_full/Feuilles \
  --groundtruth_dir ./groundtruth_clement_more \
  --color_map_path ./color_map_clement_more.json \
  --annotations_json_path ./annotations_30.json