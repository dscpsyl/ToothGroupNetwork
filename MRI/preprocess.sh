#!/usr/bin/env bash

while read p; do
    python3 preprocess.py -s ../../data/mri_data/$p.png -m ../../data/mri_data/$p.nii -o ../../data/mri_data/$p
done < ../../data/mri_data_fold.txt