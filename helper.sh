#!/usr/bin/env bash

if [[ $# -eq 0  ]]; then
	echo "No commands specified. Please tell me how I can help."
fi

if [[ "$1" == "test" ]]; then
	python start_inference.py \
  		--input_dir_path ../data/data_obj_parent_directory \
  		--split_txt_path ../data/base_name_test_fold.txt \
  		--save_path ../testing_results \
  		--model_name "tgnet" \
  		--checkpoint_path ../tgnet_fps \
  		--checkpoint_path_bdl ../tgnet_bdl
fi

if [[ $1 == "clean" ]]; then
	rm -rf ../testing_results/*
	rm -rf ../data/data_obj_parent_directory/*
fi