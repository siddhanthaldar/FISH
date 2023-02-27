import pickle as pkl 
import torch

suite = "metaworld"

env_names = {
        "metaworld": ["bin-picking-v2", "door-unlock-v2", "button-press-topdown-v2",
                      "drawer-close-v2", "door-open-v2", "hammer-v2"]
}

demo_paths = {
    "metaworld": "/path/FISH/expert_demos/metaworld"
}

for env in env_names[suite]:
    path = f"{demo_paths[suite]}/{env}/expert_demos.pkl"

    with open(path, 'rb') as f:
        obses, _, actions, _ = pkl.load(f)

    obses = torch.tensor(obses[:,:,-3:], device='cuda').float()
    obses = obses.view(3, -1)
    print(f"{env}: Mean = {obses.mean(dim=-1).cpu().data / 255.0}\t Std = {obses.std(dim=-1).cpu().data / 255.0}")