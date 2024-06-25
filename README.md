# ToothGroupNetwork
Please see the origional repo for specifications and details. This README will contain steps adn information I believe are necessary to get this up and running for a basic use.

## Installiation
You will need a specific docker image for this. You will need to clone this repo into both the docker image as well as your own local drive to use.

Install [this](https://hub.docker.com/layers/pytorch/pytorch/1.7.1-cuda11.0-cudnn8-devel/images/sha256:f0d0c1b5d4e170b4d2548d64026755421f8c0df185af2c4679085a7edc34d150) docker image into your local machine with the following command to allow for GPU access.

```sh
sudo docker run -it --gpus all pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel bash
```

This will install the docker and give you shell access into it. If you ever exit the shell, you can always reenter it with 

```sh
sudo docker start {dockerID} -i 
```

Next, install some essentials that you will need.

```sh
sudo apt get -y git vim
```

Now, clone this repo.

```sh
git clone https://github.com/dscpsyl/ToothGroupNetwork
cd ToothGroupNetwork
```

Next, you will need to install the other python dependencies after activating a new conda environment

```sh
conda create -p ./env
conda activate ./env

pip install --ignore-installed PyYAML
pip install open3d wandb multimethod termcolor trimesh easydict
```

Just make sure that you are on Python8.5.

New then, here is the buggy part...You will need to customize the custom CUDA extension to work with your GPU. First, know what kind if Nvidia GPU you have. Then, follow [this](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) table and edit `external_libs/pointops/setup.py` near the bottm to compile with the correct machine code for your GPU. Personally, I've had experince where the correct instruction set produces an error so if all else fails, then try every one of them.

After that, you can setup and compile the CUDA extension for your GPU.

```sh
cd external_libs/pointops && python setup.py install
```

## Usage

Cool! That is the setup done! Now, it's time to format the data correctly. I will not go over how to train the model as that will take too long. For now, I have provided the pre-trained model for you to work with.

To format the data, it needs to be a .obj file with the naming convention {ID}_lower.obj and {ID}_upper.obj. These two files should be put into a directory with the same name as {ID}. Great! You have the data properly formatted.

In addition, you will need to make sure that the data is in the correct orientation. (See the origional repo for mroe details.) Contained here is `test_data_process.py`, which will orient the data given a specific axis setting that we are currently using.

In the docker, I assume you have installed this repo at /workspace/ToothGroupNetwork. In that case, create the following data folder and metadata file.

```sh
mkdir /workspace/data/data_obj_parent_directory
touch /workspace/data/base_name_test_fold.txt
```

Now, from your local drive, transfer the data files into the docker container.

```sh
sudo docker cp path/to/ID/folder {dockerID}:/workspace/data/data_obj_parent_directory
```

Awesome, you're almost there! In the metadata file, each line is the {ID} of the data you want to test on. Add the ID of your new data into here. If you have more than one, make sure each {ID} is on its own line.

```sh
vi /workspace/data/base_name_test_fold.txt
```

Now, go back to the root of this repo and run the following command

```sh
python start_inference.py \
  --input_dir_path ../data/data_obj_parent_directory \
  --split_txt_path ../data/base_name_test_fold.txt \
  --save_path ../data/test_results \
  --model_name "tgnet" \
  --checkpoint_path models/tgnet_fps \
  --checkpoint_path_bdl models/tgnet_bdl
```

There might be some warnings that come up but they are fine and can be explicitly supressed.

The predicted .json files are located in `/workspace/data/test_results`. You will need to copy this to your local drive as docker cannot use openGL and X11.

```sh
sudo docker cp {dockerID}:/workspace/data/test_results path/to/parent/directory/of/local/repo
```

Here, we need both the data and the testing_results folder to be on the same level as where you installed this repo. In addition, we will need a copy of the data folder file strcture from the docker to be on the local drive too. So the file structure should look like this once you're all said and done on both your docker and local file system. NOTE: The local file system does not need the metadata file so it is in parenthesies below.

```
parentFolder
--ToothGroupNetwork
----...
--data
----data_obj_parent_directory
----(base_name_test_fold.txt)
--testing_results
----*.json...
```

Also note that you will need to have a conda environment installed on your local system with the some of the dependencies installed.

```sh
conda create -p ./env
conda activate ./env

conda install python=3.8
pip install trimesh numpy open3d
```

Finally, to view the predicted results, you can simply run, from the repo root,

```sh
./render {ID} {upper|lower}
```

## Suggestions

I recommend taking a look at `render.sh`, `transfer.sh`, and `helper.sh` as they are all scripts I've used to make my life easier.