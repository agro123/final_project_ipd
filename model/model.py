import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn
from joblib import dump
from urllib.parse import urlparse

import os
import random
import tempfile

from pre_process import pre_process
from s3_helpers import download_from_s3, upload_to_s3

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

VERSION = os.getenv("MODEL_VERSION", "v1")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

print(f"Training model version: {VERSION}")

# Configuración del experimento MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("movies_revenue_prediction")

with mlflow.start_run():
    # ======================== 1. Carga y limpieza de Datos ========================
    movies_key = f'train_data/{VERSION}/Movies.csv'
    #local_movies_path = f'./temp/Movies_{VERSION}.csv'
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_movies:
        download_from_s3(movies_key, tmp_movies.name)
        movies = pd.read_csv(tmp_movies.name)

    film_details_key = f'train_data/{VERSION}/FilmDetails.csv'
    #local_film_details_path = f'./temp/FilmDetails_{VERSION}.csv'
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_film_details:
        download_from_s3(film_details_key, tmp_film_details.name)
        film_details = pd.read_csv(tmp_film_details.name)

    # Combinar tablas
    data = pd.merge(movies, film_details, on='id')
    data = pre_process(data, True)

    # ======================== 2. División de Datos ========================
    X = data.drop(columns=['revenue_usd'])
    y = data['revenue_usd']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])

    # ======================== 3. Modelo y Entrenamiento ========================
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Predicción y Métricas
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Mediana_EA", medae)
    mlflow.log_metric("Explained_Variance", explained_var)

    print("MAE: ", mae)
    print("R2: ", r2)
    print("Mediana_EA: ", medae)
    print("Explained_Variance: ", explained_var)

    # ======================== 4. Visualización y Artefactos ========================
    # Gráfico: Predicción vs Real
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_test)), y_test, label="Real", alpha=0.6)
    plt.scatter(range(len(y_test)), y_pred, label="Predicción", alpha=0.6)
    plt.legend()
    plt.title("Predicción vs Valores Reales de revenue_usd")
    plt.savefig("pred_vs_real.png")
    mlflow.log_artifact("pred_vs_real.png")

    # Gráfico: Importancia de características
    importances = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(14, 7))
    plt.barh(features, importances)
    plt.title("Importancia de las características")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    # ======================== 5. Registro del Modelo ========================
    input_example = X_train.iloc[0].to_dict()

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"movies_revenue_prediction_{VERSION}",
            registered_model_name=f"revenue_prediction_{VERSION}",
            input_example=input_example
        )
    else:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"movies_revenue_prediction_{VERSION}",
            input_example=input_example
        )

    print('Guardando model en .joblib')
    model_key = f'models/revenue_prediction_{VERSION}.joblib'
    #local_model_path = f'./temp/revenue_prediction_{VERSION}.joblib'
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_model:
        dump(model, tmp_model.name)
        upload_to_s3(model_key, tmp_model.name)

    if os.path.exists(tmp_model.name):
        os.remove(tmp_model.name)
    if os.path.exists(tmp_movies.name):
        os.remove(tmp_movies.name)
    if os.path.exists(tmp_film_details.name):
        os.remove(tmp_film_details.name)

    print(f"Modelo {VERSION} y artefactos registrados en MLflow y S3")

#os.system('mlflow ui --host 0.0.0.0 --port 5000') #local
#os.system('mlflow server -h 0.0.0.0 --default-artifact-root s3://final-project-ipd-2024') #server