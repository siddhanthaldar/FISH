# Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations

This is a repository containing the code for the paper "Teach a Robot to FISH: Versatile Imitation from One Minute of Demonstrations".

![main_figure](https://user-images.githubusercontent.com/25313941/221458118-4c6f7ea5-abda-40db-be99-b02b0828b06d.png)

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

## Instructions to set up simulation environment
- Install [Mujoco](http://www.mujoco.org/) based on the instructions given [here](https://github.com/facebookresearch/drqv2).
- Install the following libraries:
```
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
- Install [Meta-World](https://github.com/Farama-Foundation/Metaworld)
  ```
  pip install -e /path/to/dir/metaworld
  ```
- To run experiments on the Meta-World Benchmark, you may use the above commands with the suite name replaced by `metaworld`.
