import numpy as np
from sklearn.ensemble import RandomForestClassifier


def random_classifier(X_tests):
    """
    Make a random probability and turn it into a class label using 0.5 as the threshold.
    """
    np.random.seed(42)

    results = {}

    for split_name, X in X_tests.items():
        N = X.shape[0]

        # random probability
        proba = np.random.randint(0, 1001, size=N) / 1000

        # prediction based on probability
        pred = (proba > 0.5).astype(int)

        results[split_name] = {
            "pred": pred.tolist(),
            "proba": proba.tolist(),
        }

    return results



# ESP

def majority_classifier(y_train, X_tests):
    """
    Predict the class that appears most often in y_train.
    """
    # find majority class
    unique, counts = np.unique(y_train, return_counts=True)
    majority_class = unique[np.argmax(counts)]

    # probability = fraction of majority class in training
    prob_majority = counts[np.argmax(counts)] / len(y_train)

    results = {}

    for split_name, X in X_tests.items():
        N = X.shape[0]

        pred = np.full(N, majority_class, dtype=int)
        proba = np.full(N, prob_majority, dtype=float)

        results[split_name] = {
            "pred": pred.tolist(),
            "proba": proba.tolist(),
        }

    return results



def random_forest_classifier(X_train, y_train, X_tests, n_estimators=200):
    """
    Train a RandomForest model on concatenated embeddings.
    """
    # initialize RF
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight="balanced"
    )

    # train
    rf.fit(X_train, y_train)

    results = {}

    for split_name, X in X_tests.items():
        # predict class
        pred = rf.predict(X)

        # predict probability for class 1
        proba = rf.predict_proba(X)[:, 1]

        results[split_name] = {
            "pred": pred.tolist(),
            "proba": proba.tolist(),
        }

    return results