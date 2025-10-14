import numpy as np

global_mean = 0
global_std = 1

def load_features(ds_name, seed=1, train=True, normalized=True, method='simclr', val_ind=None):
    " load pretrained features for a dataset "

    global global_mean, global_std

    dataset_features_dict = {
        'train':
            {
                'CIFAR10': f'../../scan/results/cifar-10/pretext/{method}/features_seed{seed}.npy',
                'CIFAR100': f'../../scan/results/cifar-100/pretext/{method}/features_seed{seed}.npy',
                'TINYIMAGENET': f'../../scan/results/tiny-imagenet/pretext/{method}/features_seed{seed}.npy',
                'IMAGENET50': f'../../scan/results/imagenet_50/pretext/{method}/features_seed{seed}.npy',
                'IMAGENET100': f'../../scan/results/imagenet_100/pretext/{method}/features_seed{seed}.npy',
                'IMAGENET200': f'../../scan/results/imagenet_200/pretext/{method}/features_seed{seed}.npy',
            },
        'test':
            {
                'CIFAR10': f'../../scan/results/cifar-10/pretext/{method}/test_features_seed{seed}.npy',
                'CIFAR100': f'../../scan/results/cifar-100/pretext/{method}/test_features_seed{seed}.npy',
                'TINYIMAGENET': f'../../scan/results/tiny-imagenet/pretext/{method}/test_features_seed{seed}.npy',
                'IMAGENET50': f'../../scan/results/imagenet_50/pretext/{method}/test_features_seed{seed}.npy',
                'IMAGENET100': f'../../scan/results/imagenet_100/pretext/{method}/test_features_seed{seed}.npy',
                'IMAGENET200': f'../../scan/results/imagenet_200/pretext/{method}/test_features_seed{seed}.npy',
            },

    }

    split = "train" if train else "test"
    print(split)
    print("Method:", method)
    fname = dataset_features_dict[split][ds_name].format(seed=seed)
    print("fname", fname)
    if fname.endswith('.npy'):
        features = np.load(fname)
    elif fname.endswith('.pth'):
        features = torch.load(fname)
    else:
        raise Exception("Unsupported filetype")

    if val_ind is not None:
        val_ind = np.array(val_ind, dtype=int)
        # print("Indices to remove:", val_ind)
        mask = np.ones(features.shape[0], dtype=bool)
        mask[val_ind] = False
        features_tr = features[mask]
        global_mean = np.mean(features_tr, axis=0)
        global_std = np.std(features_tr, axis=0)

    # print("global mean", np.shape(global_mean))
    # print("global mean", np.shape(global_std))

    contains_nan, num_nan = np.isnan(features).any(), np.isnan(features).sum()
    print("num_nan", num_nan)


    features = np.nan_to_num(features, nan=0.0)
    features = (features - global_mean) / (global_std + 1e-8)


    print(features)
    return features