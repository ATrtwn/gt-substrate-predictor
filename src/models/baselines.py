
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from src.training.evaluation import compute_accuracy, compute_roc_auc, compute_f1_score, compute_mcc

class BaselineWrapper:
    #TODO ESP, EZSpecificity, GT-predict (for GT1 only)

    def __init__(self, model_type='majority', **kwargs):
        """
        model_type: 'majority', 'knn', 'rf'
        kwargs: additional parameters for the model
        """
        self.model_type = model_type.lower()
        self.kwargs = kwargs
        self.model = self._init_model()

    # RF, KNN, GNB, SV
    def _init_model(self):
        if self.model_type == 'majority':
            return DummyClassifier(strategy='most_frequent', **self.kwargs)
        elif self.model_type == 'knn':
            return KNeighborsClassifier(**self.kwargs)
        elif self.model_type == 'rf':
            return RandomForestClassifier(**self.kwargs)
        elif self.model_type == 'other':
            # TODO: implement baseline from literature
            return None
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X_train, y_train):
        """Train the baseline model"""
        if self.model_type in ['majority', 'knn', 'rf']:
            self.model.fit(X_train, y_train)
        elif self.model_type == 'other':
            pass  # TODO

    def predict(self, X):
        """Return predictions"""
        if self.model_type in ['majority', 'knn', 'rf']:
            return self.model.predict(X)
        elif self.model_type == 'other':
            return None  # TODO

    def predict_proba(self, X):
        """Return probability estimates"""
        if self.model_type in ['majority', 'knn', 'rf']:
            return self.model.predict_proba(X)
        elif self.model_type == 'other':
            return None  # TODO

    def evaluate(self, X, y_true):
        """Compute evaluation metrics"""
        y_pred = self.predict(X)
        y_probs = self.predict_proba(X)
        # TODO: add task-specific metrics if needed
        metrics = {
            'accuracy': compute_accuracy(y_true, y_pred),
            'f1': compute_f1_score(y_true, y_pred, average='macro'),
            'auc': compute_roc_auc(y_true, y_probs),
            'mcc': compute_mcc(y_true, y_pred)
        }
        return metrics