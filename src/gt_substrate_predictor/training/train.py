# training

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, dataloader):
        """Train model for one epoch"""
        pass

    def validate(self, dataloader):
        """Compute validation loss/metrics"""
        pass

    def fit(self, train_loader, val_loader, epochs=10):
        """Full training loop"""
        pass