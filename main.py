from pathlib import Path
import pandas as pd

from src.utils import preprocess, clean_data, data_clustering
from src.build_model import (
    train_model,
    save_model
)
from src.predict import predict
from src.analytics import cluster_summary


# DEFINIÇÃO DE PATHS
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "data_raw"
PROCESSED_DATA_DIR = DATA_DIR / "data_processed"

DATA_PATH = RAW_DATA_DIR / "SUPERSTORE_DATASET.csv"

OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "model"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)


# TRATAMENTO DE DADOS INICIAL
df = pd.read_csv(DATA_PATH, sep=",")
df_tratado = clean_data(df)
df_tratado.to_csv(PROCESSED_DATA_DIR / "SUPERSTORE_DATASET_TRATADO.csv", index=False)

print("Dados tratados exportados com sucesso!")

# TRATAMENTO DE DADOS PARA MODELAGEM
df_modelagem = preprocess(df)
df_modelagem.to_csv(PROCESSED_DATA_DIR / "SUPERSTORE_DATASET_MODELAGEM.csv", index=False)

print("Dados para modelagem exportados com sucesso!")


# MODELO DE CLUSTERING (K-MEANS)
X_scaled, df_clustering = data_clustering(df_modelagem)
model_clustering = train_model(X_scaled, n_clusters=3)
df_clustering["CLUSTER"] = predict(model_clustering, X_scaled)
df_clustering.to_csv(MODEL_DIR / "SUPERSTORE_DATASET_CLUSTERING.csv", index=False)

print("Dados de clustering exportados com sucesso!")



print("Pipeline finalizado com sucesso!")
