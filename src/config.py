import os
import torch

# Define project directory and data directory paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Check if CUDA is available and set device accordingly
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define label dictionary for classification values
LABEL_DICT = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}