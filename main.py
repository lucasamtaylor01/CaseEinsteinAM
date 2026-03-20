from pathlib import Path
import pandas as pd

from src.utils import preprocess, clean_data
from src.build_model import (
    evaluate_k,
    choose_best_k,
    train_model,
    save_model
)
from src.predict import predict
from src.analytics import cluster_summary


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

df = pd.read_csv(DATA_PATH, sep=",")
df_tratado = clean_data(df)
df_tratado.to_csv(PROCESSED_DATA_DIR / "SUPERSTORE_DATASET_TRATADO.csv", index=False)
print("Dados tratados exportados com sucesso!")

df_modelagem = preprocess(df)
df_modelagem.to_csv(PROCESSED_DATA_DIR / "SUPERSTORE_DATASET_MODELAGEM.csv", index=False)
print("Dados para modelagem exportados com sucesso!")


"""
X = df.select_dtypes(include="number")
X, scaler = scale_data(X, X.columns)
df_metrics = evaluate_k(X)
k, df_metrics = choose_best_k(df_metrics)
model = train_model(X, n_clusters=k)
df["CLUSTER"] = predict(model, X)
df.to_csv(OUTPUT_DIR / "predicted.csv", index=False)
df_metrics.to_csv(OUTPUT_DIR / "k_metrics.csv")
summary = cluster_summary(df)
summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")
save_model(model)

print(f"Pipeline finalizado com sucesso! Melhor k = {k}")
"""

