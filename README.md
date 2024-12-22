# final_project_ipd

**Descripción:**
Proyecto final Infraestructuras paralelas y distribuidas.
Tiene como objetivo demostrar la implementación, despliegue y
monitoreo de pipelines de aprendizaje automático en entornos distribuidos, utilizando servicios
de la nube como AWS.

**Cómo ejecutar:**

- **Jupyter notebook:**
   ```bash
   docker run -it --name jupyter-all --rm -p 8888:8888 -v "$(pwd)/pip":/pip -v "$(pwd)/notebook":/home/jovyan -v "$(pwd)/data":/home/jovyan/data -v "$(pwd)/model":/home/jovyan/model jupyter/datascience-notebook


- **MLFlow:**
   ```bash
      docker build -t movie_mlflow .
      docker run -it --name movies_revenue_mlflow --rm -p 5000:5000 -v "$(pwd)/data":/app/data -v "$(pwd)/model/mlruns":/app/mlruns -v "$(pwd)/model/service.py":/app/service.py -v "$(pwd)/model/model.py":/app/model.py -v "$(pwd)/model/pre_process.py":/app/pre_process.py movie_mlflow
