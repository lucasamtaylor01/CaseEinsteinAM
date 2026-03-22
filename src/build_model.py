from pathlib import Path
import joblib
from sklearn.cluster import KMeans


def train_kmeans(X, k=3, random_state=0, output_path=None):
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10
    )
    
    model.fit(X)
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, output_path)
    
    return model