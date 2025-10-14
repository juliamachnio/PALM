import numpy as np
import pickle
import os


def load_cifar10_batch(file_path):
    """
    Load a single CIFAR-10 batch.
    Args:
        file_path (str): Path to the CIFAR-10 batch file.
    Returns:
        images (numpy.ndarray): Array of images (shape: [batch_size, 32, 32, 3]).
        labels (list): List of corresponding labels.
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    # Extract images and labels
    images = batch[b'data']
    labels = batch[b'labels']
    # Reshape images into (batch_size, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def compute_cifar10_mean_std(data_dir):
    """
    Compute the mean and standard deviation for the CIFAR-10 dataset.
    Args:
        data_dir (str): Path to the CIFAR-10 dataset directory.
    Returns:
        mean (numpy.ndarray): Mean pixel value for each channel (shape: [3]).
        std (numpy.ndarray): Standard deviation for each channel (shape: [3]).
    """
    # Get all batch files in the CIFAR-10 directory
    batch_files = [os.path.join(data_dir, f'data_batch_{i}') for i in range(1, 6)]
    all_images = []

    # Load all batches and concatenate images
    for batch_file in batch_files:
        images, _ = load_cifar10_batch(batch_file)
        all_images.append(images)

    # Concatenate all images into a single array (shape: [num_samples, 32, 32, 3])
    all_images = np.concatenate(all_images, axis=0)

    # Compute mean and std for each channel (RGB)
    mean = np.mean(all_images, axis=(0, 1, 2))  # Mean over height, width, and samples
    std = np.std(all_images, axis=(0, 1, 2))  # Std over height, width, and samples

    return mean, std


if __name__ == "__main__":
    # Path to the directory containing CIFAR-10 batches
    data_dir = './data/cifar-10-batches-py'

    # Compute mean and std
    mean, std = compute_cifar10_mean_std(data_dir)

    # Print results
    print("CIFAR-10 Mean (per channel):", mean)
    print("CIFAR-10 Std (per channel):", std)
