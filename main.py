from pathlib import Path
import pandas as pd
from src.utils import build_temporal_dict, preprocess, clean_data, data_clustering
from src.build_model import train_arima_by_cluster, train_kmeans
from src.predict import predict


# PATH DEFINITIONS
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'data_raw'
PROCESSED_DATA_DIR = DATA_DIR / 'data_processed'
DATA_PATH = RAW_DATA_DIR / 'SUPERSTORE.csv'
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_DIR = BASE_DIR / 'model'

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)


# INITIAL DATA PROCESSING
df = pd.read_csv(DATA_PATH, sep=',')
df_processed = clean_data(df)
df_processed.to_csv(PROCESSED_DATA_DIR / 'SUPERSTORE_PROCESSED.csv', index=False)

print('[1/4] Processed data exported successfully.\n')


# DATA PROCESSING FOR MODELING
df_modeling = preprocess(df_processed)
df_modeling.to_csv(PROCESSED_DATA_DIR / 'SUPERSTORE_MODELING.csv', index=False)

print('[2/4] Modeling data exported successfully.\n')


# CLUSTERING MODEL (K-MEANS)
X_clustering, df_clustering = data_clustering(df_modeling)
model_clustering = train_kmeans(
    X=X_clustering,
    k=3,
    random_state=0,
    output_path=OUTPUT_DIR / 'models_predict' / 'kmeans_clustering.joblib'
)

df_clustering['CLUSTER'] = predict(model_clustering, X_clustering)
df_clustering.to_csv(
    MODEL_DIR / 'SUPERSTORE_CLUSTERING.csv',
    index=False
)
print('[3/4] Clustering model trained and results exported successfully.\n')

# TIME SERIES MODEL BY CLUSTER
df_temporal_dict = build_temporal_dict(df_clustering)
arima_results = train_arima_by_cluster(df_temporal_dict, output_dir=OUTPUT_DIR / 'models_predict')

print('[4/4] ARIMA models trained successfully.\n')

print('[!] Pipeline completed successfully.\n')
