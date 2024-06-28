#!/usr/bin/env bash


if [ $# -eq 0 ]; then
    echo "No arguments supplied. Tell me if I should transfer up or down."
fi

if [[ "$1" == "up" ]]; then
    sudo docker cp ../data/data_obj_parent_directory 7a768fe93fc7:/workspace/3DToothSegmentation/data
    sudo docker cp ../data/base_name_test_fold.txt 7a768fe93fc7:/workspace/3DToothSegmentation/data
fi

if [[ "$1" == "down" ]]; then
    sudo docker cp 7a768fe93fc7:/workspace/3DToothSegmentation/testing_results ../
    sudo chown -R $USER:$USER ../testing_results
fi
