import os
import numpy as np
import torch
from src import config as cfg
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

# Function to load MNIST data
def load_mnist() -> tuple:
    """
    Load the MNIST dataset.

    Returns:
        tuple: A tuple containing the images and labels.
    """
    # Load the training and test images
    train_images = pd.read_csv(os.path.join(cfg.DATA_DIR, 'fashion-mnist_train.csv'))
    test_images = pd.read_csv(os.path.join(cfg.DATA_DIR, 'fashion-mnist_test.csv'))
    all_images = pd.concat([train_images, test_images])  # Combine training and test datasets

    # Extract the labels from the dataset
    labels = all_images['label'].values

    # Remove the labels from the dataset
    images = all_images.drop('label', axis=1).values

    # Reshape each row into a 28x28 matrix
    images = images.reshape(-1, 28, 28)

    # Convert the images and labels to float32 for compatibility with PyTorch
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    # Return the images and labels
    return images, labels

#Class for Fashion MNIST dataset
class FashionMNISTDataset(Dataset):
    """
    Dataset class for Fashion MNIST.
    """

    def __init__(self, images: np.array, labels: np.array, transform: transforms) -> None:
        """
        Initialize the dataset.

        Args:
            images (np.array): Array of images.
            labels (np.array): Array of labels.
            transform (transforms): Data transformation.

        Returns:
            None
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """
        Get the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get the image and label at the given index.

        Args:
            idx (int): Index of the image.

        Returns:
            tuple: Tuple containing the image and label.
        """
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    #Klases konstruktorius
    def __init__(self,
                 images: np.array,
                 labels: np.array,
                 transform: transforms) -> None:

        self.images = images
        self.labels = labels
        self.transform = transform

    #Grazina bendra nuotrauku skaiciu
    def __len__(self) -> int:
        return len(self.images)

    #Grazina nuotrauka pagal jos indeksa
    def __getitem__(self,
                    idx: int) -> tuple:

        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Function to create data split for training, validation, and testing
def create_data_split(images_idx: np.array,
                      label_idx: np.array,
                      split_val: float = 0.2,
                      split_test: float = 0.25) -> tuple:
    """
    Create data split for training, validation, and testing.

    Args:
        images_idx (np.array): Array of image indices.
        label_idx (np.array): Array of label indices.
        split_val (float): Validation split ratio (default: 0.2).
        split_test (float): Test split ratio (default: 0.25).

    Returns:
        tuple: A tuple containing the training, validation, and test samplers.
    """

    # Split the training data into three parts: train, validation, and test
    images_train, images_test, masks_train, masks_test = train_test_split(images_idx, label_idx, test_size=split_test,
                                                                          random_state=42)

    images_train, images_val, masks_train, masks_val = train_test_split(images_train, masks_train, test_size=split_val,
                                                                        random_state=42)

    # Shuffle the data within each split
    train_sampler = SubsetRandomSampler(images_train)
    val_sampler = SubsetRandomSampler(images_val)
    test_sampler = SubsetRandomSampler(images_test)

    return train_sampler, val_sampler, test_sampler


#
def create_data_loader(images: np.array,
                       labels: np.array,
                       transforms: transforms,
                       batch_size: int,
                       split: float=0.2) -> tuple:
    """
    Create data loaders for training, validation, and testing.

    Args:
        images (np.array): Array of images.
        labels (np.array): Array of labels.
        transforms (transforms): Data transformation.
        batch_size (int): Batch size.
        split (float): Validation split ratio (default: 0.2).

    Returns:
        tuple: A tuple containing the training, validation, and test data loaders.
    """

    # Initialize the FashionMNISTDataset class
    dataset = FashionMNISTDataset(images, labels, transforms)

    # Create arrays of indices for images and labels
    images_idx = np.arange(len(dataset))
    label_idx = np.arange(len(dataset))

    # Split the indices into training, validation, and test sets
    train_sampler, val_sampler, test_sampler = create_data_split(images_idx, label_idx)

    # Create data loaders for training, validation, and test sets
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


def plot_image_and_label(image: np.array,
                         label: torch.tensor) -> None:
    """
    Plot the image and corresponding label.

    Args:
        image (np.array): Image array.
        label (torch.tensor): Label tensor.

    Returns:
        None
    """

    image = np.squeeze(image)  # Reduce the dimensions of the image array to 2D
    label = label.item()  # Convert the label tensor to a scalar value

    plt.imshow(image, cmap='gray')
    plt.title(cfg.LABEL_DICT[label])
    plt.show()

    if __name__ == '__main__':
        # Load MNIST data
        images, labels = load_mnist()

        # Define data transformations
        transforms = transforms.Compose([transforms.ToTensor()])

        # Create data loaders for training and validation
        train_loader, val_loader, _ = create_data_loader(images, labels, transforms, batch_size=32)

        # Display sample images
        for i, (image, label) in enumerate(train_loader):
            plot_image_and_label(image[0], label[0])
            if i == 0:
                break