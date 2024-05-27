import os
from src import config as cfg
import numpy as np
from mnist_loader import load_mnist, FashionMNISTDataset, create_data_loader
from src.NeuralNet import CNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim, nn
import subprocess
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#Model training function
def train_model(model: nn.Module,
                train_dataset: FashionMNISTDataset,
                val_dataset: FashionMNISTDataset,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                writer: SummaryWriter,
                num_epochs: int = 10) -> None:
    """
    Trains the model using the given training dataset and evaluates it on the validation dataset.
    
    Args:
        model (nn.Module): The neural network model to be trained.
        train_dataset (FashionMNISTDataset): The training dataset.
        val_dataset (FashionMNISTDataset): The validation dataset.
        optimizer (optim.Optimizer): The optimizer used for updating the model parameters.
        criterion (nn.Module): The loss function.
        writer (SummaryWriter): The TensorBoard writer for logging.
        num_epochs (int, optional): The number of training epochs. Defaults to 10.
    """
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        
        # Set the model to training mode
        model.train()
        
        # Iterate over each batch in the training dataset
        for batch_idx, (image, label) in enumerate(train_dataset):
            predictions = []  # Array to store predictions
            truth = []  # Array to store true labels
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Move images and labels to the appropriate device
            image = image.to(cfg.DEVICE)
            label = label.to(cfg.DEVICE, dtype=torch.long)
            
            # Forward pass
            output = model(image)
            
            # Compute the loss
            loss = criterion(output, label)
            
            # Backward pass
            loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            # Apply softmax to get probabilities and find the predicted class
            prob = torch.softmax(output, dim=1)
            prediction = torch.argmax(prob, dim=1)
            
            # Store predictions and true labels
            predictions.extend(prediction.detach().cpu().numpy().flatten())
            truth.extend(label.detach().cpu().numpy().flatten())
            
            # Compute accuracy and other metrics
            accuracy = accuracy_score(truth, predictions)
            precision = precision_score(truth, predictions, average='macro', zero_division=0)
            recall = recall_score(truth, predictions, average='macro', zero_division=0)
            f1 = f1_score(truth, predictions, average='macro', zero_division=0)
            
            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataset) + batch_idx)
            writer.add_scalar('Accuracy/train', accuracy, epoch)
            writer.add_scalar('Precision/train', precision, epoch)
            writer.add_scalar('Recall/train', recall, epoch)
            writer.add_scalar('F1/train', f1, epoch)
        
        # Set the model to evaluation mode
        model.eval()
        total_loss = 0
        predictions = []
        truth = []
        
        # Disable gradient computation
        with torch.no_grad():
            # Iterate over each batch in the validation dataset
            for batch_idx, (image, label) in enumerate(val_dataset):
                image = image.to(cfg.DEVICE)
                label = label.to(cfg.DEVICE, dtype=torch.long)
                
                # Forward pass
                output = model(image)
                
                # Compute the loss
                loss = criterion(output, label)
                total_loss += loss.item()
                
                # Apply softmax to get probabilities and find the predicted class
                prob = torch.softmax(output, dim=1)
                prediction = torch.argmax(prob, dim=1)
                
                # Store predictions and true labels
                predictions.extend(prediction.detach().cpu().numpy().flatten())
                truth.extend(label.detach().cpu().numpy().flatten())
        
        # Compute accuracy and other metrics
        accuracy = accuracy_score(truth, predictions)
        precision = precision_score(truth, predictions, average='macro')
        recall = recall_score(truth, predictions, average='macro')
        f1 = f1_score(truth, predictions, average='macro')
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/val', total_loss / len(val_dataset), epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('Recall/val', recall, epoch)
        writer.add_scalar('F1/val', f1, epoch)

#Function to evaluate the model with the test dataset
def evaluate_model(model: nn.Module,
                   test_dataset: FashionMNISTDataset,
                   criterion: nn.Module) -> tuple:
    """
    Evaluates the model using the given test dataset and returns the evaluation metrics.

    Args:
        model (nn.Module): The trained neural network model.
        test_dataset (FashionMNISTDataset): The test dataset.
        criterion (nn.Module): The loss function.

    Returns:
        tuple: A tuple containing the evaluation metrics.
    """

    # Set the model to evaluation mode
    model.eval()
    total_loss = 0
    predictions = []
    truth = []

    # Disable gradient computation
    with torch.no_grad():
        # Iterate over each batch in the test dataset
        for batch_idx, (image, label) in enumerate(test_dataset):
            image = image.to(cfg.DEVICE)
            label = label.to(cfg.DEVICE, dtype=torch.long)

            # Forward pass
            output = model(image)
            loss = criterion(output, label)
            total_loss += loss.item()

            # Apply softmax to get probabilities and find the predicted class
            prob = torch.softmax(output, dim=1)
            prediction = torch.argmax(prob, dim=1)
            predictions.extend(prediction.detach().cpu().numpy().flatten())
            truth.extend(label.detach().cpu().numpy().flatten())

    # Compute evaluation metrics
    accuracy = accuracy_score(truth, predictions)
    precision = precision_score(truth, predictions, average='macro')
    recall = recall_score(truth, predictions, average='macro')
    f1 = f1_score(truth, predictions, average='macro')
    matrix = confusion_matrix(predictions, truth)

    # Prepare evaluation metrics for TensorBoard
    score_table = f"""
            | Metric     | Score  | 
            |------------|--------|
            | Accuracy   | {accuracy:.3f} |
            | Precision  | {precision:.3f} |
            | Recall     | {recall:.3f} |
            | F1         | {f1:.3f} |
        """
    score_table = '\n'.join(l.strip() for l in score_table.splitlines())
    writer.add_text("Score Table", score_table, 0)

    # Create and log the confusion matrix
    writer.add_figure('Confusion Matrix', plot_confusion_matrix(matrix), 0)

#Function to create a confusion matrix
def confusion_matrix(predictions: list,
                     truth: list) -> np.array:
    """
    Computes the confusion matrix based on the predicted and true labels.

    Args:
        predictions (list): The predicted labels.
        truth (list): The true labels.

    Returns:
        np.array: The confusion matrix.
    """
    num_classes = len(cfg.LABEL_DICT)
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Update the confusion matrix based on the predictions and truth labels
    for i in range(len(predictions)):
        confusion_matrix[truth[i]][predictions[i]] += 1

    return confusion_matrix

#Function to plot the confusion matrix
def plot_confusion_matrix(confusion_matrix: np.array) -> plt.Figure:
    """
    Plots the confusion matrix.

    Args:
        confusion_matrix (np.array): The confusion matrix.

    Returns:
        plt.Figure: The plotted confusion matrix.
    """
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Convert the confusion matrix to float32
    confusion_matrix = confusion_matrix.astype(np.float32)
    # Display the confusion matrix
    ax.matshow(confusion_matrix, cmap='Blues')

    # Set the axis labels and ticks
    labels = list(cfg.LABEL_DICT.values())
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # Add the values to the matrix cells
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center')

    # Adjust the layout and return the figure
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # Load the data
    images, labels = load_mnist()
    transforms = transforms.Compose([transforms.ToTensor()])
    # Prepare the data for training, validation, and testing
    train_loader, val_loader, test_loader = create_data_loader(images, labels, transforms, 32)

    # Create the model
    model = CNN(in_channels=1,
                num_classes=10,
                num_layers=2,
                dropout_rate=0.2,
                kernel_size=3,
                pool_kernel_size=2,
                pool_stride=2,
                stride=1,
                padding=1).to(cfg.DEVICE)

    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Specify the loss function
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter()
    # Train the model
    train_model(model, train_loader, val_loader, optimizer, criterion, writer, 50)
    # Evaluate the model
    evaluate_model(model, test_loader, criterion)

    # Get a name for this experiment from the user
    comment = input("Enter a name for this experiment: ")
    writer.add_text('Experiment', comment)
    writer.close()

    # Save the model parameters if requested by the user
    save_model = input("Do you want to save the model parameters? (yes/no): ")
    if save_model.lower() == 'yes':
        model_path = os.path.join(cfg.PROJECT_DIR, 'models', f'{comment}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")
    # Launch TensorBoard
    subprocess.run(["tensorboard", "--logdir", "runs"])
