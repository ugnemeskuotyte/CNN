import torch
from torch import nn
import numpy as np

class ConvulationBlock(nn.Module):
    """
    A class representing a convolutional block in a neural network.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        dropout_rate (float, optional): The dropout rate. Default is 0.2.
        kernel_size (int, optional): The size of the convolutional kernel. Default is 3.
        stride (int, optional): The stride of the convolution. Default is 1.
        padding (int, optional): The padding size. Default is 1.
        activation (nn.Module, optional): The activation function. Default is nn.ReLU(inplace=True).
        pool_kernel_size (int, optional): The size of the pooling kernel. Default is 2.
        pool_stride (int, optional): The stride of the pooling operation. Default is 2.
        batch_norm (bool, optional): Whether to use batch normalization. Default is True.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout_rate: float = 0.2,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 pool_kernel_size: int = 2,
                 pool_stride: int = 2,
                 batch_norm: bool = True) -> None:
        super(ConvulationBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Function that specifies how data should flow through the layers.

        Args:
            x (torch.tensor): The input tensor.

        Returns:
            torch.tensor: The output tensor.
        """
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x

# Define the CNN class
class CNN(nn.Module):
    """
    A class representing a Convolutional Neural Network.

    Args:
        in_channels (int): The number of input channels.
        num_classes (int): The number of output classes.
        num_layers (int): The number of convolutional layers.
        dropout_rate (float, optional): The dropout rate. Default is 0.2.
        kernel_size (int, optional): The size of the convolutional kernel. Default is 3.
        pool_kernel_size (int, optional): The size of the pooling kernel. Default is 2.
        pool_stride (int, optional): The stride of the pooling operation. Default is 2.
        stride (int, optional): The stride of the convolution. Default is 1.
        padding (int, optional): The padding size. Default is 1.
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 num_layers: int,
                 dropout_rate: float = 0.2,
                 kernel_size: int = 3,
                 pool_kernel_size: int = 2,
                 pool_stride: int = 2,
                 stride: int = 1,
                 padding: int = 1) -> None:
        super(CNN, self).__init__()
        self.channels_in = in_channels
        self.num_classes = num_classes

        self.layers = nn.ModuleList()
        channels_out = 16
        self.flattened_size = 28  # Image size is 28 x 28

        # Create convolutional blocks
        for i in range(num_layers):
            self.layers.append(ConvulationBlock(self.channels_in,
                                                channels_out,
                                                dropout_rate,
                                                kernel_size,
                                                stride,
                                                padding))
            self.channels_in = channels_out
            channels_out *= 2

            # Update image size
            self.flattened_size = ((self.flattened_size - kernel_size + 2 * padding) / stride) + 1
            self.flattened_size = np.round(self.flattened_size / pool_kernel_size, 0)

            # If it's the last convolutional block, flatten the data to a single column
            if i == num_layers - 1:
                self.layers.append(nn.Flatten())

        # Create the last linear layer
        self.last_layer = nn.Linear(int(self.channels_in * self.flattened_size * self.flattened_size), self.num_classes)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Function that specifies how data should flow through the layers.

        Args:
            x (torch.tensor): The input tensor.

        Returns:
            torch.tensor: The output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        x = self.last_layer(x)
        return x

def test_cnn():
    # Create an instance of the CNN model
    model = CNN(1, 10, 2)

    # Generate random input data
    x = torch.randn(64, 1, 28, 28)

    # Perform forward pass through the model
    output = model(x)

    # Check the shape of the output tensor
    assert output.shape == torch.Size([64, 10])

    # Print a success message if the test passed
    print("Test passed")

if __name__ == "__main__":
    # Run the test function
    test_cnn()