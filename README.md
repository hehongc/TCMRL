# Code for Intra-level and Inter-level Contexts for Generalization in Offline Meta-Reinforcement Learning, NeurIPS 2024

## Installation
To install locally, you will need to first install [MuJoCo](https://www.roboti.us/index.html). 
To handle task distributions with varying reward functions, such as those found in the Cheetah and Ant environments, it is recommended to install MuJoCo150 or a more recent version.
Set `LD_LIBRARY_PATH` to point to both the MuJoCo binaries (`/$HOME/.mujoco/mujoco200/bin`) as well as the gpu drivers (something like `/usr/lib/nvidia-390`, you can find your version by running `nvidia-smi`).

To set up the remaining dependencies, create a conda environment using the following steps:
```
conda env create -f environment.yaml
```
Install the wandb:
```
pip install wandb 
```

**For Walker environments**, MuJoCo131 is required.
To install it, follow the same procedure as for MuJoCo200. To switch between different MuJoCo versions, you can use the following steps:
```
export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mjpro${VERSION_NUM}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro${VERSION_NUM}/bin
```

In addition to the aforementioned steps, you will also need to download the metaworld.

## Data Generation
Example of training policies and generating trajectories on multiple tasks:
For point-robot and cheetah-vel:
```
CUDA_VISIBLE_DEVICES=0 python policy_train.py ./configs/sparse-point-robot.json 
CUDA_VISIBLE_DEVICES=0 python policy_train.py ./configs/cheetah-vel.json
```

For Meta-World ML1 tasks:
```
python data_collection_ml1.py  ./configs/ml1.json
```
you can modify the task in `./configs/ml1.json`

data will be saved in `./data/`

## Offline RL Experiments
For Meta-World ML1 experiment, run: 
```
run_ml1_1.sh
```
To run different tasks, modify "env_name" in `./configs/cpearl-ml1-1.json` as well as "datadirs" in `run_ml1_1.sh`.

For point-robot and cheetah-vel:
```
run_point_1.sh
run_cheetah_1.sh
```
