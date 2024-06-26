#!/usr/bin/env bash

read -p "The two dir arguments are required to be in relative path format starting from home (i.e. ~/data/path ~/new/path). Press any key to contine or CTL-c to quit."

echo "This script will find all .ply files in a given directory and process them accoringly"
echo "It will also anom the files and save them into a secified directory"
echo "We expect the jaw scans to be of '*LowerJawScan.ply' and '*UpperJawScan.ply'"

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi

input_dir=$1
output_dir=$2

input_dir="${input_dir/#~/$HOME}"
output_dir="${output_dir/#~/$HOME}"

if [[ ! -d $input_dir ]]; then
    echo "Error: $input_dir is not a directory"
    exit 1
fi


datas=()

find $input_dir -name "*LowerJawScan.ply" -print0 | xargs -0 -I {} echo {} > /tmp/tmp.txt

while read file; do
    echo "Processing lower jaw $file"
    newfile=$(date +%s | sha256sum | base64 | head -c 21)
    mkdir -p $output_dir/$newfile
    output_file="$output_dir/$newfile/${newfile}_lower.obj"
    
    python3 test_preprocessor.py -p "$file" -o "$output_file" -j l

    fileBaseName=$(basename -s LowerJawScan.ply "$file")
    upperJaw="$input_dir/${fileBaseName}UpperJawScan.ply"   
    if [[ -f "$upperJaw" ]]; then
        echo "Found upper jaw counterpart $upperJaw"
        echo "Processing upper jaw $upperJaw"
        output_file=$output_dir/$newfile/${newfile}_upper.obj
        python3 test_preprocessor.py -p "$upperJaw" -o "$output_file" -j u
    else
        echo "Could not find upper jaw counterpart for $fileBaseName"
    fi
    datas+=("$newfile")
done < /tmp/tmp.txt

for d in "${datas[@]}"; do
    echo "$d" >> ../data/base_name_test_fold.txt
done