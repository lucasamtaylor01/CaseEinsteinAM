from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score


def train_model(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=0)
    model.fit(X)
    return model

def save_model(model):
    joblib.dump(model, "model/model.joblib")