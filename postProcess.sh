#!/usr/bin/env bash


# /home/user/Downloads/3DToothSegmentation/data/data_obj_parent_directory/p1/p1_lower.obj
# /home/user/Downloads/3DToothSegmentation/testing_results/p1_lower.json

# if [[ $# -lt 2 ]]; then
#     echo "Usage: $0 -i <input_obj_file> -j <input_json_file>"
#     exit 1 
# fi

if [[ $1 == "-h" ]]; then
    echo "Usage: $0 -i <input_obj_file> -j <input_json_file>"
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "Uitilizing default options"
    python3 post_processor.py -i "/home/user/Downloads/3DToothSegmentation/data/data_obj_parent_directory/p1/p1_lower.obj" -j "/home/user/Downloads/3DToothSegmentation/testing_results/p1_lower.json" -r "huber"
else
    python3 post_processor.py -i "$1" -j "$2" -r "3"
fi