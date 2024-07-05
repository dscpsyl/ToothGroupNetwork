#!/usr/bin/env bash


# /home/user/Downloads/3DToothSegmentation/data/data_obj_parent_directory/p1/p1_lower.obj
# /home/user/Downloads/3DToothSegmentation/testing_results/p1_lower.json

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 -i <input_obj_file> -j <input_json_file>"
    exit 1
fi

python3 postProcess.py -i "$1" -j "$2