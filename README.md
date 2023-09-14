# Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations

[[Arxiv]](https://arxiv.org/abs/2303.01497) [[Project page and videos]](https://fast-imitation.github.io/)

This is a repository containing the code for the paper "Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations".

![main_figure](https://user-images.githubusercontent.com/25313941/222460948-cb72d5bb-12bc-4ce1-bf2b-cf90f542a344.png)

## Instructions

- Set up Conda Environment

  ```
  conda env create -f conda_env.yml
  conda activate fish
  ```
  The PyTorch suite is not installed when you create the environment; you must go to the PyTorch website [here](https://pytorch.org/) and manually install PyTorch. You **MUST** ensure that the CUDA version of PyTorch matches that of your machine. For example, if you use CUDA 12, you must install the preview, not the stable version of PyTorch because it is the only version that supports CUDA 12 (as of September 2023).
- Install [ROS 1](https://www.ros.org/). Notice that you MUST use ROS 1; ROS 2 is not supported.
- Install [Holo-Dex](https://github.com/SridharPandian/Holo-Dex)
and [DIME-Controllers](https://github.com/NYU-robot-learning/DIME-Controllers).
- Train BC agent - We provide three different commands for running the code on the Hello Stretch robot, Allegro Hand, and xArm.

  ```
  python train_hand.py agent=bc suite=hellogym suite/hellogym_task=dooropen num_demos=1
  ```

  ```
  python train_hand.py agent=bc suite=handgym suite/handgym_task=cardslide num_demos=1
  ```

  ```
  python train_robot.py agent=bc suite=robotgym suite/robotgym_task=flipbagel num_demos=2
  ```
- Train FISH - We provide three different commands for running the code on the Hello Stretch robot, Allegro Hand, and xArm.

  ```
  python train_hand.py agent=potil_vinn_offset suite=hellogym suite/hellogym_task=dooropen load_bc=true num_demos=1
  ```

  ```
  python train_hand.py agent=potil_openloop_offset suite=handgym suite/handgym_task=cardslide load_bc=true num_demos=1
  ```

  ```
  python train_robot.py agent=potil_openloop_offset suite=robotgym suite/robotgym_task=flipbagel load_bc=true num_demos=2
  ```
- Monitor results

```
tensorboard --logdir exp_local
```

- To use pre-trained encoders, you must install [MVP](https://github.com/ir413/mvp) and [R3M](https://github.com/facebookresearch/r3m) using instructions provided in the respective repositories.

## Instructions to set up simulation environment

- Install [Mujoco](http://www.mujoco.org/) and mujoco-py based on the instructions given [here](https://github.com/facebookresearch/drqv2).
- Install the following libraries:

```
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf
```

- Download the Meta-World benchmark suite from [here](https://osf.io/4w69f/?view_only=e29b9dc9ea474d038d533c2245754f0c). Install the simulation environment using the following command -
  ```
  pip install -e /path/to/dir/metaworld
  ```
- Download the expert demonstrations from [here](https://osf.io/4w69f/?view_only=e29b9dc9ea474d038d533c2245754f0c) and extract the downloaded `expert_demos.zip` to `FISH/FISH`.

- To run experiments on the Meta-World Benchmark, you may use the above commands with the suite name replaced by `metaworld`.
For example, to run the `door_open` task, you first run behavior cloning with:
  ```
  python train_robot.py agent=bc suite=metaworld suite/metaworld_task=door_open num_demos=1
  ```
  Then, train FISH with:
  ```
  python train_robot.py agent=potil_vinn_offset suite=metaworld suite/metaworld_task=door_open load_bc=true num_demos=1
  ```

## Troubleshooting

There are several things to notice:

1. You must source your ROS setup script (e.g., `source /opt/ros/noetic/setup.bash`) and then source the DIME-Controllers setup script (e.g., `source /path/to/DIME-Controllers/devel/setup.bash`), in that order, before running scripts to train behavior cloning or FISH. You must activate the conda environment **AFTER** sourcing the two setup scripts.
2. The `train_hand.py` script does not work with certain tasks, an example of which is `door_open` from `metaworld`.
If you are experiencing errors with `train_hand.py`, you may need to use `train_robot.py` instead (and vice versa).
3. The expert demo files provide a limited amount of data.
For certain tasks, there might be only one expert demonstration. If you are experiencing errors such as "`list index out of range`", you may need to lower the number of demonstrations to use (e.g., changing `python train_robot.py agent=potil_vinn_offset suite=metaworld suite/metaworld_task=door_open load_bc=true num_demos=2` to `python train_robot.py agent=potil_vinn_offset suite=metaworld suite/metaworld_task=door_open load_bc=true num_demos=1`).
4. Certain base policies may not work with certain tasks.
For example, `potil_vinn_offset` does not work with `door_open` from `metaworld`.

## Bibtex

```
@article{haldar2023teach,
         title={Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations},
         author={Haldar, Siddhant and Pari, Jyothish and Rai, Anant and Pinto, Lerrel},
         journal={arXiv preprint arXiv:2303.01497},
         year={2023}
}
```
