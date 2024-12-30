import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, median_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn
from joblib import dump

import os
import random
import sys

from pre_process import pre_process

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

VERSION = "v1"

print(f"Training model version: {VERSION}")

debug = True

if len(sys.argv) == 2:
    debug = bool(sys.argv[1])

# Configuración del experimento MLflow
mlflow.set_experiment("movies_revenue_prediction")

with mlflow.start_run():
    # ======================== 1. Carga y limpieza de Datos ========================
    movies = pd.read_csv(f'/app/data/train_data/{VERSION}/Movies.csv')
    film_details = pd.read_csv(f'/app/data/train_data/{VERSION}/FilmDetails.csv')

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
    rmse = root_mean_squared_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("Mediana_EA", medae) #Ayuda a reducir el impacto de valores atípicos (outliers).
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
    input_example = X_train.iloc[0].to_dict() # Genera un ejemplo basado en los datos de entrenamiento

    const = model.predict(pd.DataFrame([input_example]))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=f"movies_revenue_prediction_{VERSION}",
        registered_model_name=f"revenue_prediction_{VERSION}",
        input_example=input_example
    )

    os.system('sh -c mlflow server --host 0.0.0.0 --port 5000 --artifacts-destination s3://final-project-ipd-2024')

    dump(model, f'../models/revenue_prediction_{VERSION}.joblib')

    print(f"Model {VERSION} & artifacts registered on MLflow")

#os.system('mlflow ui --host 0.0.0.0 --port 5000')