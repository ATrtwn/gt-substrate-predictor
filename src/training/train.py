# training
import logging
from sklearn.metrics import mean_squared_error, log_loss

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
    def predict(self, X):
        """Predict using the trained model"""
        pass



class SklearnTrainer(Trainer):
    def __init__(self, model, loss_fn=log_loss):
        super().__init__(model,optimizer=None, loss_fn=loss_fn, device='cpu')
        self.loss_fn = loss_fn # e.g. log_loss (BCE) for classification

    def train_epoch(self, X, y):
        """For sklearn models, one epoch is just fit once"""
        self.model.fit(X, y)
        loss = self.loss_fn(y, self.model.predict(X))
        return loss

    def validate(self, X, y):
        if hasattr(self.model, "predict_proba"):
            y_pred = self.model.predict_proba(X)[:, 1]
            loss = log_loss(y, y_pred, labels=[0,1])
        else:
            loss = None
        return loss

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=1):
        history = {"train_loss": [], "val_loss": []}
        logging.info(f"type of X_train: {type(X_train)}")
        logging.info(f"type of y_train: {type(y_train)}")
        self.model.fit(X_train, y_train)
        loss = self.loss_fn(y_train, self.model.predict(X_train))
        history["train_loss"].append(loss)

        val_loss = self.validate(X_val, y_val) if X_val is not None and y_val is not None else None
        history["val_loss"].append(val_loss)


        return history
    def predict(self, X):
        if hasattr(self.model, "predict_proba"):
            y_pred_prob = self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "decision_function"):
            y_pred_prob = self.model.decision_function(X)   # AUROC OK
        else:
            y_pred_prob = self.model.predict(X)
        return y_pred_prob
