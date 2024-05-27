import os
from src import config as cfg
import numpy as np
from mnist_loader import load_mnist, FashionMNISTDataset, create_data_loader
from src.NeuralNet import CNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch import optim, nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Function to select all classes
def select_all_classes(test_dataset: FashionMNISTDataset) -> tuple:
    """
    Selects all unique classes from the test dataset and saves the indices of objects belonging to each class.
    Args:
        test_dataset (FashionMNISTDataset): The test dataset.

    Returns:
        tuple: A tuple containing the unique classes and the dictionary of class indices.
    """
    # Get the unique classes in the test dataset
    classes = np.unique(test_dataset.labels)
    
    # Create a dictionary to store the indices of objects belonging to each class
    class_indices = {int(class_): np.where(test_dataset.labels == class_)[0] for class_ in classes}
    
    return classes, class_indices

# Function to sample 30 images
def sample_30_images(test_dataset: FashionMNISTDataset,
                     classes: np.array,
                     class_indices: dict) -> tuple:
    """
    Samples 30 images from the test dataset, 3 images per class.
    Args:
        test_dataset (FashionMNISTDataset): The test dataset.
        classes (np.array): The unique classes.
        class_indices (dict): The dictionary of class indices.

    Returns:
        tuple: A tuple containing the sampled images and labels.
    """
    # Initialize empty lists to store sampled images and labels
    sampled_images = []
    sampled_labels = []
    
    # Iterate over each class
    for class_ in classes:
        # Get the indices of objects belonging to the current class
        indices = class_indices[class_]
        
        # Randomly sample 3 indices without replacement
        sampled_indices = np.random.choice(indices, 3, replace=False)
        
        # Append the sampled images and labels to the respective lists
        sampled_images.extend(test_dataset.images[sampled_indices])
        sampled_labels.extend(test_dataset.labels[sampled_indices])

    # Return the sampled images and labels
    return sampled_images, sampled_labels

# Function to predict samples
def predict_samples(model: nn.Module,
                    sampled_images: np.array,
                    sampled_labels: np.array) -> np.array:
    """
    Predicts the labels of the sampled images using the specified model.
    Args:
        model (nn.Module): The neural network model.
        sampled_images (np.array): The sampled images.
        sampled_labels (np.array): The corresponding labels.

    Returns:
        np.array: The predicted labels.
    """
    predictions = []
    truth = []

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over each sampled image and label
        for image, label in zip(sampled_images, sampled_labels):
            # Convert the image and label to tensors and move them to the device
            image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(cfg.DEVICE)
            label = torch.tensor(label).unsqueeze(0).unsqueeze(0).to(cfg.DEVICE, dtype=torch.long)
            
            # Forward pass through the model
            output = model(image)
            
            # Calculate the probabilities using softmax
            prob = torch.softmax(output, dim=1)
            
            # Get the predicted label
            prediction = torch.argmax(prob, dim=1)
            
            # Append the predicted label and true label to the respective lists
            predictions.append(prediction.item())
            truth.append(label.item())

    return predictions

# Function to plot images
def plot_images(sampled_images: np.array,
                sampled_labels: np.array,
                predictions: np.array) -> None:
    """
    Plots the sampled images along with their true labels and predicted labels.
    Args:
        sampled_images (np.array): The sampled images.
        sampled_labels (np.array): The corresponding true labels.
        predictions (np.array): The predicted labels.

    Returns:
        None
    """
    # Create subplots for displaying images
    fig, axes = plt.subplots(10, 3, figsize=(10, 20))
    
    # Iterate over each image and corresponding label
    for i, ax in enumerate(axes.flat):
        # Display the image
        ax.imshow(sampled_images[i], cmap='gray')
        
        # Set the title of the subplot with the true and predicted labels
        ax.set_title(f"True: {cfg.LABEL_DICT[sampled_labels[i]]}\nPred: {cfg.LABEL_DICT[predictions[i]]}")
        
        # Turn off the axis labels
        ax.axis('off')

    # Adjust the layout of the subplots
    plt.tight_layout()
    
    # Display the plot
    plt.show()

if __name__ == '__main__':
    # Load MNIST dataset
    images, labels = load_mnist()

    # Define data transformations
    transforms = transforms.Compose([transforms.ToTensor()])

    # Create data loader for test dataset
    _, _, test_loader = create_data_loader(images, labels, transforms, 30)

    # Create CNN model
    model = CNN(in_channels=1,
                num_classes=10,
                num_layers=2,
                dropout_rate=0.2,
                kernel_size=3,
                pool_kernel_size=2,
                pool_stride=2,
                stride=1,
                padding=1).to(cfg.DEVICE)

    # Load pre-trained model weights
    model.load_state_dict(torch.load(os.path.join(cfg.PROJECT_DIR, 'models', 'final_1.pt')))

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Sample 30 images from test dataset
    sampled_images, sampled_labels = sample_30_images(test_loader.dataset, *select_all_classes(test_loader.dataset))

    # Predict labels for sampled images
    predictions = predict_samples(model, sampled_images, sampled_labels)

    # Plot sampled images with true and predicted labels
    plot_images(sampled_images, sampled_labels, predictions)