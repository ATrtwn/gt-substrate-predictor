import torch.nn as nn

def save_model(model, path="../../experiments"):
    """Save model checkpoint to experiments folder """
    pass

def load_model(path, device='cpu'):
    """Load model checkpoint """
    pass

class GT_NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # TODO: define layers

    def forward(self, x):
        # TODO: define forward pass
        pass