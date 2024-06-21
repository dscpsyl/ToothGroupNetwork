#!/usr/bin/env bash 

echo "This is a helper script to envoke render.py more easily by giving it simply the data-hash file name. We require the dataset to be store in ../data as specified in the origional project readme and the predicted tests stored in ../testing_results."
echo "No safety is guarenteeded with this script."
echo "The first one that comes out is ground truth. The second one is predicted"


if [ $# -eq 0 ]
  then
    echo "No arguments supplied. ./render.sh [file name of predicted test] [upper or lower jaw]"
    exit
fi


# python render.py \
#   --mesh_path ../data/data_obj_parent_directory/$1/$1_$2.obj \
#   --gt_json_path ../data/data_json_parent_directory/$1/$1_$2.json \
#   --pred_json_path ../testing_results/$1_$2.json

python render.py \
  --mesh_path ../data/data_obj_parent_directory/$1/$1_$2.obj \
  --pred_json_path ../testing_results/$1_$2.json

