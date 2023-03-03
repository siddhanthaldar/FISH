# Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations
[[Arxiv]](https://arxiv.org/abs/2303.01497) [[Project page and videos]](https://fast-imitation.github.io/)

This is a repository containing the code for the paper "Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations".

![main_figure](https://user-images.githubusercontent.com/25313941/222460948-cb72d5bb-12bc-4ce1-bf2b-cf90f542a344.png)

## Instructions
- Set up Environment
  ```
  conda env create -f conda_env.yml
  conda activate fish
  ```
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
- Install [Mujoco](http://www.mujoco.org/) based on the instructions given [here](https://github.com/facebookresearch/drqv2).
- Install the following libraries:
```
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
- Download the Meta-World benchmark suite and its demonstrations from [here](https://osf.io/4w69f/?view_only=e29b9dc9ea474d038d533c2245754f0c). Install the simulation environment using the following command - 
  ```
  pip install -e /path/to/dir/metaworld
  ```
- To run experiments on the Meta-World Benchmark, you may use the above commands with the suite name replaced by `metaworld`.

## Bibtex
```
@article{haldar2023teach,
         title={Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations},
         author={Haldar, Siddhant and Pari, Jyothish and Rai, Anant and Pinto, Lerrel},
         journal={arXiv preprint arXiv:2303.01497},
         year={2023}
}
```
