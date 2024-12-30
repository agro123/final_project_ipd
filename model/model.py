import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn
import os
import random
import sys
import boto3
from joblib import dump
from pre_process import pre_process

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

VERSION = 'v1'

print(f"Training model version: {VERSION}")

# Configuración del experimento MLflow
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("movies_revenue_prediction")

# Configura cliente S3
s3 = boto3.client('s3')
BUCKET_NAME = "final-project-ipd-2024"

# Función para descargar archivos desde S3
def download_from_s3(key, local_path):
    s3.download_file(BUCKET_NAME, key, local_path)

# Función para subir archivos a S3
def upload_to_s3(key, local_path):
    s3.upload_file(local_path, BUCKET_NAME, key)

with mlflow.start_run():
    # ======================== 1. Carga y limpieza de Datos ========================
    movies_key = f'train_data/{VERSION}/Movies.csv'
    film_details_key = f'train_data/{VERSION}/FilmDetails.csv'
    local_movies_path = f'./temp/Movies_{VERSION}.csv'
    local_film_details_path = f'./temp/FilmDetails_{VERSION}.csv'

    # Descargar los archivos desde S3
    download_from_s3(movies_key, local_movies_path)
    download_from_s3(film_details_key, local_film_details_path)

    # Leer los archivos CSV
    movies = pd.read_csv(local_movies_path)
    film_details = pd.read_csv(local_film_details_path)

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

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=f"movies_revenue_prediction_{VERSION}",
        #registered_model_name=f"revenue_prediction_{VERSION}",
        input_example=input_example
    )

    # Guardar el modelo localmente
    local_model_path = f'./temp/revenue_prediction_{VERSION}.joblib'
    dump(model, local_model_path)

    # Subir el modelo a S3
    model_key = f'models/revenue_prediction_{VERSION}.joblib'
    upload_to_s3(model_key, local_model_path)

    print(f"Modelo {VERSION} y artefactos registrados en MLflow y S3")

#os.system('mlflow ui --host 0.0.0.0 --port 5000')
#os.system('mlflow server -h 0.0.0.0 --default-artifact-root s3://final-project-ipd-2024')