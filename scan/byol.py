import argparse
import os
import torch
import numpy as np
import dill  
from utils.config import create_config
from utils.common_config import get_train_dataset, get_val_dataset, get_val_dataloader, get_val_transformations
from termcolor import colored
from models.byol import BYOL

# Parser
parser = argparse.ArgumentParser(description='BYOL Feature Extraction')
parser.add_argument('--config_env', help='Config file for the environment')
parser.add_argument('--config_exp', help='Config file for the experiment')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp, args.seed)
    print(colored(p, 'red'))

    def download_weights(url, filename):
        """Download pretrained weights if not already present."""
        if not os.path.exists(filename):
            print(colored(f"Downloading {filename}...", 'blue'))
            os.system(f'wget -L {url} -O {filename}')
        else:
            print(colored(f"{filename} already exists. Skipping download.", 'green'))

    def load_pretrained_weights(model, checkpoint_path):
        """Load BYOL pretrained weights using dill."""
        with open(checkpoint_path, "rb") as f:
            checkpoint = dill.load(f)  # Using dill to correctly load the DeepMind checkpoint

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint  

        new_state_dict = {}
        for k in state_dict.keys():
            new_k = k.replace("encoder", "online_encoder")  
            new_state_dict[new_k] = state_dict[k]

        model.load_state_dict(new_state_dict, strict=False)
        print(colored("Pretrained BYOL weights loaded successfully.", 'green'))

    # Download Pretrained Weights
    WEIGHTS_URL = "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_batchsize_512.pkl"
    CHECKPOINT_PATH = "res50x1_batchsize_512.pkl"

    download_weights(WEIGHTS_URL, CHECKPOINT_PATH)

    # Model Initialization
    print(colored("Initializing BYOL model...", 'blue'))
    model = BYOL(feature_dim=128)  
    load_pretrained_weights(model, CHECKPOINT_PATH)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

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
    print(f'Dataset contains {len(train_dataset)}/{len(val_dataset)} train/val samples')

    # Feature Extraction Function
    def extract_features(dataloader, model, use_projector=False):
        """Extract features using the BYOL online encoder."""
        features = []
        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].cuda(non_blocking=True)

                if use_projector:
                    feat = model.module.projector(model.module.online_encoder(images))  
                else:
                    feat = model.module.online_encoder(images)  

                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    print(colored('Extracting features with BYOL', 'blue'))
    train_features = extract_features(train_dataloader, model, use_projector=False)  # Extract backbone features (2048D)
    val_features = extract_features(val_dataloader, model, use_projector=False)

    # Save extracted features
    np.save(p['pretext_features'], train_features)
    np.save(p['pretext_features'].replace('features', 'test_features'), val_features)

    print(colored(f"Train Features Shape: {train_features.shape}", 'green'))
    print(colored(f"Validation Features Shape: {val_features.shape}", 'green'))

if __name__ == '__main__':
    main()
