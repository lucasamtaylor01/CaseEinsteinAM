# SuperstoreDS

## 📊 Description
Analysis and modeling project using the Superstore dataset, focused on EDA, clustering, and profit forecasting.


## ⚙️ Install dependencies

   **Linux/macOS:**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```
   **Windows (PowerShell)**
   ```bash
   python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
   ```

## 📂 Structure

```plaintext
.
├── 📂 data
│   ├── 📂 data_processed
│   │   ├── 📄 SUPERSTORE_MODELING.csv
│   │   └── 📄 SUPERSTORE_PROCESSED.csv
│   └── 📂 data_raw
│       └── 📄 SUPERSTORE.csv
├── 🐍 main.py
├── 📂 model
│   └── 📄 SUPERSTORE_CLUSTERING.csv
├── 📓 notebooks
│   ├── 📘 01_initial_exploration.ipynb
│   ├── 📘 02_outliers.ipynb
│   ├── 📘 03_kmeans.ipynb
│   └── 📘 04_temporal_series.ipynb
├── 📂 output
│   └── 📂 models_predict
│       ├── 🤖 arima_cluster_0.joblib
│       ├── 🤖 arima_cluster_1.joblib
│       ├── 🤖 arima_cluster_2.joblib
│       └── 🤖 kmeans_clustering.joblib
├── 📝 README.md
├── 📊 report
│   ├── 📂 notebook_01
│   ├── 📂 notebook_03
│   └── 📂 notebook_04
├── 📦 requirements.txt
└── 📂 src
    ├── ⚙️ build_model.py
    ├── 🔮 predict.py
    └── 🧰 utils.py
```

## 💾 Dataset

Available on [Github](https://raw.githubusercontent.com/WuCandice/Superstore-Sales-Analysis/refs/heads/main/dataset/Superstore%20Dataset.csv)
