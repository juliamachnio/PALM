import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_train_dataset, get_val_dataset, get_val_dataloader, get_val_transformations
from termcolor import colored
from models.moco_v3 import MoCoV3

# Parser
parser = argparse.ArgumentParser(description='MoCo')
parser.add_argument('--config_env', help='Config file for the environment')
parser.add_argument('--config_exp', help='Config file for the experiment')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()


def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp, args.seed)
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve MoCo v3 model for feature extraction', 'blue'))
    model = MoCoV3(backbone=p['backbone'], dim=256, pred_dim=512)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()  # Ensure evaluation mode

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    transforms = get_val_transformations(p)
    train_dataset = get_train_dataset(p, transforms)
    val_dataset = get_val_dataset(p, transforms)
    train_dataloader = get_val_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Load MoCo v3 checkpoint
    print(colored('Downloading MoCo v3 checkpoint', 'blue'))
    os.system('wget -L https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar')
    moco_state = torch.load('r-50-1000ep.pth.tar', map_location='cpu')

    # Transfer MoCo v3 weights
    print(colored('Transfer MoCo v3 weights to model', 'blue'))
    new_state_dict = {}
    state_dict = moco_state['state_dict']

    for k in list(state_dict.keys()):
        # Ignore momentum encoder weights (we don't use them)
        if k.startswith('module.momentum_encoder'):
            print(f"Skipping momentum encoder key: {k}")
            continue  # Skip loading momentum encoder

        # Rename backbone weights to match MoCoV3 model
        elif k.startswith('module.base_encoder'):
            new_k = k.replace("module.base_encoder", "module.backbone")
            new_state_dict[new_k] = state_dict[k]

        # Keep only projector weights
        elif k.startswith('module.projector'):
            new_state_dict[k] = state_dict[k]

        # Ignore predictor weights (used only for training, not inference)
        elif k.startswith('module.predictor'):
            print(f"Skipping predictor layer: {k}")

        else:
            print(f"Skipping unexpected key: {k}")

    # Load the modified weights
    model.load_state_dict(new_state_dict, strict=False)
    os.system('rm -rf r-50-1000ep.pth.tar')

    # Feature Extraction Function
    def extract_features(dataloader, model, use_projector=False):
        features = []
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # print("Batch structure:", type(batch), batch.keys())  # Debugging statement
                images = batch['image'].cuda(non_blocking=True)  # Move images to GPU


                if use_projector:
                    feat = model.projector(model.backbone(images))  # Extract projector features
                else:
                    feat = model.module.backbone(images)  # Extract backbone features

                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    # Extract features
    train_features = extract_features(train_dataloader, model, use_projector=False)
    val_features = extract_features(val_dataloader, model, use_projector=False)

    # Save features
    np.save(p['pretext_features'], train_features)
    np.save(p['pretext_features'].replace('features', 'test_features'), val_features)

    print("Train Features Shape:", train_features.shape)
    print(train_features)
    print("Validation Features Shape:", val_features.shape)
    print(val_features)


if __name__ == '__main__':
    main()