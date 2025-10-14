"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored

def log_feature_stats(feature_path, name):
    """
    Logs statistics of the feature matrix.
    Args:
        feature_path (str): Path to the feature matrix .npy file.
        name (str): Name of the feature matrix (e.g., "features" or "test_features").
    """
    if os.path.exists(feature_path):
        features = np.load(feature_path)
        print(f"Statistics for {name}:")
        print(f"  Shape: {features.shape}")
        print(f"  Min: {features.min()}")
        print(f"  Max: {features.max()}")
        print(f"  Contains NaN: {np.isnan(features).any()}")
        print(f"  Number of NaN values: {np.isnan(features).sum()}")
    else:
        print(f"{name} not found at {feature_path}")


# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--seed', type=int, default=1, help='Random seed')

args = parser.parse_args()


def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp, args.seed)
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
                                      split='train+unlabeled')  # Split is for stl-10
    val_dataset = get_val_dataset(p, val_transforms)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # # Loop through batches in the train_dataloader
    # for batch_idx, batch in enumerate(train_dataloader):  # Assuming the dataloader returns (images, labels)
    #     # `images` is a tensor of shape (batch_size, channels, height, width)
    #     # `labels` is a tensor of shape (batch_size,)
    #
    #     print(f"Batch {batch_idx + 1}:")
    #
    #     # Iterate through images in the batch
    #     for img_idx, image in enumerate(images):
    #         min_val = image.min().item()
    #         max_val = image.max().item()
    #         print(f"  Image {img_idx + 1}: Min={min_val}, Max={max_val}")
    #
    #     # Optionally, break the loop to inspect only the first few batches
    #     if batch_idx >= 2:  # Inspect only the first 3 batches
    #         break

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, split='train')  # Dataset w/o augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset)

    # if p['train_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
    #
    #     memory_bank_base = MemoryBank(len(base_dataset),feature_dim=2048,
    #                                    p['num_classes'], p['criterion_kwargs']['temperature'])
    #     memory_bank_base.cuda()
    #     memory_bank_val = MemoryBank(len(val_dataset), feature_dim=2048,
    #                                  p['num_classes'], p['criterion_kwargs']['temperature'])
    #     memory_bank_val.cuda()

    memory_bank_base = MemoryBank(len(base_dataset),
                                      p['model_kwargs']['features_dim'],
                                      p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                     p['model_kwargs']['features_dim'],
                                     p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        simclr_train(train_dataloader, model, criterion, optimizer, epoch)

        # Fill memory bank
        print('Fill memory bank for kNN...')
        fill_memory_bank(base_dataloader, model, memory_bank_base)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base)
        print('Result of kNN evaluation is %.2f' % (top1))

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, p['pretext_checkpoint'])

        topk = 20
        print('Mine the nearest neighbors (Top-%d)' % (topk))
        indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
        np.save(p['topk_neighbors_train_path'], indices)
        np.save(p['pretext_features'], memory_bank_base.pre_lasts.cpu().numpy())
        fill_memory_bank(val_dataloader, model, memory_bank_val)
        indices, acc = memory_bank_val.mine_nearest_neighbors(topk)

        print("accuracy val: %.2f" % (acc))
        np.save(p['topk_neighbors_train_path'], indices)
        # np.save(p['pretext_features'], memory_bank_val.pre_lasts.cpu().numpy())
        np.save(p['pretext_features'].replace('features', 'test_features'), memory_bank_val.pre_lasts.cpu().numpy())

        log_feature_stats(p['pretext_features'], "features")
        log_feature_stats(p['pretext_features'].replace('features', 'test_features'), "test_features")

    # Save final model
    torch.save(model.state_dict(), p['pretext_model'])

    log_feature_stats(p['pretext_features'], "features")
    log_feature_stats(p['pretext_features'].replace('features', 'test_features'), "test_features")


    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' %(topk))
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_train_path'], indices)
    # save features
    np.save(p['pretext_features'], memory_bank_base.pre_lasts.cpu().numpy())
    np.save(p['pretext_features'].replace('features', 'test_features'), memory_bank_val.pre_lasts.cpu().numpy())


    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' %(topk))
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], indices)

 
if __name__ == '__main__':
    main()
