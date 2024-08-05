#!/usr/bin/env bash

if [[ $1 == 'convert-obj' ]]; then

	while read f; do
		python3 test_preprocessor.py -p "$HOME/Downloads/3DToothSegmentation/data/data_obj_parent_directory/$f/${f}_lower.obj" -o "$HOME/Downloads/3DToothSegmentation/data/data_obj_parent_directory/$f/t${f}_lower.obj" -j l
		python3 test_preprocessor.py -p "$HOME/Downloads/3DToothSegmentation/data/data_obj_parent_directory/$f/${f}_upper.obj" -o "$HOME/Downloads/3DToothSegmentation/data/data_obj_parent_directory/$f/t${f}_upper.obj" -j u

		mv ../data/data_obj_parent_directory/$f/${f}_lower.obj ../data/data_obj_parent_directory/$f/${f}_lower.obj.bkup
		mv ../data/data_obj_parent_directory/$f/${f}_upper.obj ../data/data_obj_parent_directory/$f/${f}_upper.obj.bkup

		mv ../data/data_obj_parent_directory/$f/t${f}_lower.obj ../data/data_obj_parent_directory/$f/${f}_lower.obj
		mv ../data/data_obj_parent_directory/$f/t${f}_upper.obj ../data/data_obj_parent_directory/$f/${f}_upper.obj

	done < ../data/base_name_test_fold.txt

fi

if [[ $1 == 'revert-obj' ]]; then

	cd ../data/data_obj_parent_directory
	find . | grep '.obj$' | xargs -I {} rm {}

	for f in $(find . | grep '.bkup$'); do
		mv -- "$f" "${f%.bkup}"
	done

fi

if [[ $1 == 'convert-ply' ]]; then

	while read f; do
		python3 test_preprocessor.py -p "$HOME/Downloads/3DToothSegmentation/data/data_obj_parent_directory/$f/${f}_lower.ply" -o "$HOME/Downloads/3DToothSegmentation/data/data_obj_parent_directory/$f/${f}_lower.obj" -j l
		python3 test_preprocessor.py -p "$HOME/Downloads/3DToothSegmentation/data/data_obj_parent_directory/$f/${f}_upper.ply" -o "$HOME/Downloads/3DToothSegmentation/data/data_obj_parent_directory/$f/${f}_upper.obj" -j u

	done < ../data/base_name_test_fold.txt

fi

if [[ $1 == 'revert-ply' ]]; then

	cd ../data/data_obj_parent_directory
	find . | grep '.obj$' | xargs -I {} rm {}

fi