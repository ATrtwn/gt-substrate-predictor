# training
from sklearn.metrics import mean_squared_error

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



class SklearnTrainer(Trainer):
    def __init__(self, model, loss_fn=None):
        super().__init__(model)
        self.loss_fn = loss_fn if loss_fn is not None else mean_squared_error

    def train_epoch(self, X, y):
        """For sklearn models, one epoch is just fit once"""
        self.model.fit(X, y)
        loss = self.loss_fn(y, self.model.predict(X))
        return loss

    def validate(self, X, y):
        loss = self.loss_fn(y, self.model.predict(X))
        return loss

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=1):
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            train_loss = self.train_epoch(X_train, y_train)
            history["train_loss"].append(train_loss)

            val_loss = self.validate(X_val, y_val) if X_val is not None and y_val is not None else None
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss if val_loss is not None else 'N/A'}")

        return history
